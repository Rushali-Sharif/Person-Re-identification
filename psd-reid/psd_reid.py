"""Platonic Self-Distillation for person re-ID. Self-contained: CLIP ViT-B/16 from timm + BNNeck, no COPE.

  baseline : ID (CE) + batch-hard triplet
  PSD      : baseline + EMA teacher + referee agreement + distillation of CONFIRMED neighbourhoods
             (1) teacher kNN of each training image is checked against a frozen referee (DINOv2);
             (2) neighbours both agree on are "confirmed"; images with none are "pending" (no distillation yet,
                 re-checked every epoch as the teacher improves);
             (3) the student is pulled toward the teacher's feature (weighted by agreement) and toward the
                 teacher features of the confirmed neighbours (multi-positive contrastive relation loss).
Usage:
  python psd_reid.py --make-dino OUT.pt --root MARKET_ROOT
  python psd_reid.py --root MARKET_ROOT --name base --epochs 40
  python psd_reid.py --root MARKET_ROOT --name psd  --epochs 40 --psd --dino DINO.pt
"""
import os, re, glob, json, math, time, copy, random, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MEAN, STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)   # CLIP statistics


# ------------------------------------------------------------------ data
def read_split(root, split):
    items = []
    for p in sorted(glob.glob(f"{root}/{split}/*.jpg")):
        m = re.match(r"([-\d]+)_c(\d)", os.path.basename(p)); pid, cam = int(m.group(1)), int(m.group(2)) - 1
        if pid != -1: items.append((p, pid, cam))            # junk images (pid -1) are ignored, distractors (pid 0) stay
    return items


class ImgDS(Dataset):
    def __init__(self, items, tf): self.items, self.tf = items, tf
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, pid, cam = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), pid, cam, i


class PKBatches:
    """P identities x K images per batch; the number of batches per epoch = N // (P*K)."""
    def __init__(self, labels, P, K, seed=0):
        self.by = {}
        for i, y in enumerate(labels): self.by.setdefault(int(y), []).append(i)
        self.pids, self.P, self.K, self.n = sorted(self.by), P, K, len(labels) // (P * K); self.rng = random.Random(seed)
    def __len__(self): return self.n
    def __iter__(self):
        order = self.pids[:]; self.rng.shuffle(order); ptr = 0
        for _ in range(self.n):
            if ptr + self.P > len(order): self.rng.shuffle(order); ptr = 0
            batch = []
            for pid in order[ptr:ptr + self.P]:
                pool = self.by[pid]; batch += self.rng.sample(pool, self.K) if len(pool) >= self.K else self.rng.choices(pool, k=self.K)
            ptr += self.P; yield batch


def transforms(size):
    h, w = size
    train = T.Compose([T.Resize((h, w), interpolation=T.InterpolationMode.BICUBIC), T.RandomHorizontalFlip(), T.Pad(10), T.RandomCrop((h, w)),
                       T.ToTensor(), T.Normalize(MEAN, STD), T.RandomErasing(p=0.5, value="random")])
    test = T.Compose([T.Resize((h, w), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(MEAN, STD)])
    return train, test


# ------------------------------------------------------------------ model
class ReID(nn.Module):
    def __init__(self, num_classes, backbone, size, pretrained=True):
        super().__init__()
        import timm
        self.enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0, img_size=size)   # CLS feature, pos-embed resized to 16x8
        d = self.enc.num_features
        self.bn = nn.BatchNorm1d(d); self.bn.bias.requires_grad_(False)                                # BNNeck
        self.cls = nn.Linear(d, num_classes, bias=False); nn.init.normal_(self.cls.weight, std=0.001)
    def forward(self, x):
        f = self.enc(x); fb = self.bn(f)
        return f, fb, self.cls(fb)


def triplet_softmargin(f, y):
    d = torch.cdist(f, f); same = y[:, None] == y[None, :]
    ap = (d * same).max(1).values; an = (d + 1e5 * same).min(1).values
    return F.softplus(ap - an).mean()


@torch.no_grad()
def ema_update(teacher, student, m):
    for pt, ps in zip(teacher.parameters(), student.parameters()): pt.mul_(m).add_(ps.detach(), alpha=1 - m)
    for bt, bs in zip(teacher.buffers(), student.buffers()):
        if bt.dtype.is_floating_point: bt.mul_(m).add_(bs, alpha=1 - m)      # BN statistics follow the same slow average as the weights
        else: bt.copy_(bs)


# ------------------------------------------------------------------ features, agreement, evaluation
@torch.no_grad()
def extract(model, items, tf, bs=128, workers=4):
    model.eval(); out = []
    for x, _, _, _ in DataLoader(ImgDS(items, tf), batch_size=bs, shuffle=False, num_workers=workers):
        with torch.autocast(DEV, enabled=DEV == "cuda"): _, fb, _ = model(x.to(DEV))
        out.append(F.normalize(fb.float(), dim=1))
    return torch.cat(out)


@torch.no_grad()
def confirm_neighbours(feats, ref, k=20, full=0.5, use_ref=True):
    """nbr[i]: the model's k nearest neighbours of image i. conf[i,a]: neighbour a is ALSO among i's k nearest under the referee.
    agreement a_i = mean(conf[i]); weight w_i = clip((a_i - chance) / (full - chance), 0, 1), chance = k / (N - 1)."""
    f = F.normalize(feats.float(), dim=1); r = F.normalize(ref.float().to(f.device), dim=1) if use_ref else None
    N = f.shape[0]; nbr = torch.zeros(N, k, dtype=torch.long, device=f.device); conf = torch.zeros(N, k, dtype=torch.bool, device=f.device)
    for s in range(0, N, 1024):
        e = min(s + 1024, N); rows = torch.arange(e - s, device=f.device); cols = torch.arange(s, e, device=f.device)
        a = f[s:e] @ f.T; a[rows, cols] = -2; na = a.topk(k, 1).indices; nbr[s:e] = na
        if use_ref:
            b = r[s:e] @ r.T; b[rows, cols] = -2; nb = b.topk(k, 1).indices
            conf[s:e] = (na.unsqueeze(2) == nb.unsqueeze(1)).any(2)
        else: conf[s:e] = True
    agree = conf.float().mean(1); chance = k / (N - 1)
    w = ((agree - chance) / (full - chance)).clamp(0, 1) if use_ref else torch.ones(N, device=f.device)
    return nbr, conf, w


def rank_eval(dist, qp, gp, qc, gc, max_rank=10):
    idx = np.argsort(dist, 1); cmc_sum = np.zeros(max_rank); aps = []; valid = 0
    for i in range(dist.shape[0]):
        order = idx[i]; keep = ~((gp[order] == qp[i]) & (gc[order] == qc[i]))
        match = (gp[order][keep] == qp[i]).astype(np.float32)
        if match.sum() == 0: continue
        valid += 1; cmc = match.cumsum(); cmc[cmc > 1] = 1; cmc = np.pad(cmc, (0, max(0, max_rank - len(cmc))), mode="edge"); cmc_sum += cmc[:max_rank]
        pos = np.where(match == 1)[0]; aps.append(((np.arange(len(pos)) + 1) / (pos + 1)).mean())
    return cmc_sum / valid, float(np.mean(aps))


def evaluate(model, query, gallery, tf, workers=4):
    t0 = time.time(); qf = extract(model, query, tf, workers=workers); gf = extract(model, gallery, tf, workers=workers)
    dist = (2 - 2 * qf @ gf.T).cpu().numpy()
    cmc, mAP = rank_eval(dist, np.array([q[1] for q in query]), np.array([g[1] for g in gallery]),
                         np.array([q[2] for q in query]), np.array([g[2] for g in gallery]))
    return {"mAP": 100 * mAP, "R1": 100 * cmc[0], "R5": 100 * cmc[4], "R10": 100 * cmc[9], "eval_seconds": time.time() - t0}


# ------------------------------------------------------------------ DINOv2 referee features
def make_dino(root, out, size=(252, 126)):
    from transformers import AutoModel
    items = read_split(root, "bounding_box_train")
    m = AutoModel.from_pretrained("facebook/dinov2-small").to(DEV).eval()
    tf = T.Compose([T.Resize(size, interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    feats = []
    with torch.no_grad():
        for x, _, _, _ in DataLoader(ImgDS(items, tf), batch_size=128, shuffle=False, num_workers=4):
            with torch.autocast(DEV, enabled=DEV == "cuda"): feats.append(m(pixel_values=x.to(DEV)).pooler_output.float().cpu())
    f = torch.cat(feats); torch.save(f, out); print("saved", tuple(f.shape), "->", out)


# ------------------------------------------------------------------ training
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--name", default="run"); ap.add_argument("--out", default="/kaggle/working/psd_reid_out")
    ap.add_argument("--make-dino", default=""); ap.add_argument("--dino", default="")
    ap.add_argument("--backbone", default="vit_base_patch16_clip_224.openai"); ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--P", type=int, default=16); ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--head-lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=3); ap.add_argument("--eval-every", type=int, default=20); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--psd", action="store_true"); ap.add_argument("--no-gate", action="store_true", help="distil every teacher neighbour, no referee")
    ap.add_argument("--psd-start", type=int, default=5); ap.add_argument("--ramp", type=int, default=3, help="epochs to ramp the distillation weight 0 -> 1"); ap.add_argument("--knn", type=int, default=20); ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--lam-feat", type=float, default=1.0); ap.add_argument("--lam-rel", type=float, default=0.5); ap.add_argument("--ema", type=float, default=0.998)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--size", type=int, nargs=2, default=[256, 128]); ap.add_argument("--max-train", type=int, default=0)
    a = ap.parse_args()
    if a.make_dino: return make_dino(a.root, a.make_dino)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); os.makedirs(a.out, exist_ok=True)

    train = read_split(a.root, "bounding_box_train"); query = read_split(a.root, "query"); gallery = read_split(a.root, "bounding_box_test")
    if a.max_train: train = train[:a.max_train]
    pids = sorted({t[1] for t in train}); remap = {p: i for i, p in enumerate(pids)}; train = [(p, remap[y], c) for p, y, c in train]
    labels = [t[1] for t in train]; N = len(train); print(f"train {N} images, {len(pids)} ids | query {len(query)} | gallery {len(gallery)}")
    ttf, etf = transforms(tuple(a.size))
    dl = DataLoader(ImgDS(train, ttf), batch_sampler=PKBatches(labels, a.P, a.K, a.seed), num_workers=a.workers)
    dino = torch.load(a.dino) if a.psd and a.dino and not a.no_gate else None
    if dino is not None: assert len(dino) == N, f"DINOv2 features ({len(dino)}) must match the training set ({N})"

    model = ReID(len(pids), a.backbone, tuple(a.size), pretrained=not a.no_pretrained).to(DEV)
    teacher = copy.deepcopy(model).eval() if a.psd else None
    if teacher is not None:
        for p in teacher.parameters(): p.requires_grad_(False)
    head = [p for n, p in model.named_parameters() if not n.startswith("enc.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": a.lr}, {"params": head, "lr": a.head_lr}], weight_decay=a.wd)
    total, warm = a.epochs * len(dl), a.warmup * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: min(1.0, (it + 1) / max(1, warm)) * (0.5 * (1 + math.cos(math.pi * min(it, total) / total))))
    scaler = torch.amp.GradScaler(enabled=DEV == "cuda"); log = {"args": vars(a), "epochs": []}; t_start = time.time()

    for ep in range(1, a.epochs + 1):
        t0 = time.time(); use_psd = a.psd and ep >= a.psd_start; stats = {}
        ramp = min(1.0, (ep - a.psd_start + 1) / max(1, a.ramp)) if use_psd else 0.0
        if use_psd:                                                         # teacher bank + referee agreement, refreshed every epoch
            bank = extract(teacher, train, etf, workers=a.workers)
            nbr, conf, w = confirm_neighbours(bank, dino, k=a.knn, use_ref=dino is not None)
            stats = {"mean_w": w.mean().item(), "frac_w_pos": (w > 0).float().mean().item(), "frac_confirmed_nbrs": conf.float().mean().item(),
                     "frac_pending": (conf.sum(1) == 0).float().mean().item()}
        model.train(); run_loss = 0.0
        for it, (x, y, cam, idx) in enumerate(dl):
            x, y, idx = x.to(DEV), y.to(DEV), idx.to(DEV)
            with torch.autocast(DEV, enabled=DEV == "cuda"):
                f, fb, logit = model(x)
                loss = F.cross_entropy(logit.float(), y, label_smoothing=0.1) + triplet_softmargin(f.float(), y)
                if use_psd:
                    with torch.no_grad(): _, tb, _ = teacher(x)
                    sb, tb = F.normalize(fb.float(), dim=1), F.normalize(tb.float(), dim=1); wb = w[idx]
                    l_feat = ((1 - (sb * tb).sum(1)) * wb).mean()                                           # follow the teacher where referees agree
                    lg = sb @ bank.T / a.tau; lg[torch.arange(len(idx)), idx] = -1e4                       # never use the image itself
                    lp = F.log_softmax(lg, 1).gather(1, nbr[idx]); cf = conf[idx].float(); cnt = cf.sum(1)
                    l_rel_i = -(lp * cf).sum(1) / cnt.clamp(min=1); has = (cnt > 0).float()                   # pending images (cnt == 0) are skipped
                    l_rel = (l_rel_i * has).sum() / has.sum().clamp(min=1)
                    loss = loss + ramp * (a.lam_feat * l_feat + a.lam_rel * l_rel)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); scaler.step(opt); scaler.update(); sched.step()
            if use_psd: ema_update(teacher, model, a.ema)
            run_loss += loss.item()
        rec = {"epoch": ep, "loss": run_loss / len(dl), "seconds": time.time() - t0, **stats}
        if ep % a.eval_every == 0 or ep == a.epochs:
            rec["eval"] = evaluate(model, query, gallery, etf, a.workers)
            if teacher is not None: rec["eval_teacher"] = evaluate(teacher, query, gallery, etf, a.workers)    # is the teacher ahead of the student?
        log["epochs"].append(rec); print(json.dumps(rec), flush=True)
    log["final"] = log["epochs"][-1].get("eval"); log["train_minutes"] = (time.time() - t_start) / 60
    json.dump(log, open(f"{a.out}/{a.name}.json", "w"), indent=1)
    os.makedirs(f"{a.out}/ckpt", exist_ok=True); torch.save(model.state_dict(), f"{a.out}/ckpt/{a.name}.pth")   # one file per run only
    print("FINAL", json.dumps(log["final"]))


if __name__ == "__main__":
    main()
