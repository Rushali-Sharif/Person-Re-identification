# ---- Video re-ID demo with psd_iter1 (standalone: no COPE, no masks) ----
import os, sys, glob, base64, subprocess, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.io import loadmat
import torchvision.transforms as T
for p in ("imageio", "imageio-ffmpeg", "timm"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", p], check=False)
import timm, imageio.v2 as imageio

QUERY_CAM, RENDER_CAM = "c1", "c2"   # query person taken from one camera, searched for in another
MAXFRAME, FPS, THRESH, WIDTH = 300, 6, 0.55, 960
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MEAN, STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)

# ---------- checkpoint ----------
cands = sorted(glob.glob("/kaggle/working/psd_reid_out/ckpt/*.pth")) + sorted(glob.glob("/kaggle/input/**/*.pth", recursive=True))
cands = [c for c in cands if os.path.getsize(c) > 1e8]
CK = next((c for c in cands if "psd_iter1" in os.path.basename(c)), None) or next((c for c in cands if "iter1" in c), None)
assert CK, f"psd_iter1 checkpoint not found. Candidates seen:\n" + "\n".join(cands)
print("checkpoint:", CK, f"({os.path.getsize(CK)/1e6:.0f} MB)")

class ReID(nn.Module):
    def __init__(self, nid, dim=768, bias=False):
        super().__init__()
        self.enc = timm.create_model("vit_base_patch16_clip_224.openai", pretrained=False, num_classes=0, img_size=(256, 128))
        self.bn = nn.BatchNorm1d(dim); self.bn.bias.requires_grad_(False)
        self.cls = nn.Linear(dim, nid, bias=bias)
    def forward(self, x):
        f = self.enc(x); fb = self.bn(f); return f, fb, self.cls(fb)

sd = torch.load(CK, map_location="cpu")
model = ReID(sd["cls.weight"].shape[0], sd["bn.weight"].shape[0], "cls.bias" in sd)
model.load_state_dict(sd); model = model.to(DEV).eval()
tf = T.Compose([T.Resize((256, 128), interpolation=3), T.ToTensor(), T.Normalize(MEAN, STD)])
print("model loaded,", sd["cls.weight"].shape[0], "training identities")

# ---------- PRW ----------
PRW = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "frames" in dirs and "annotations" in dirs: PRW = root; break
assert PRW, "PRW not found under /kaggle/input - add it with 'Add Input'"
print("PRW at:", PRW)

def read_anno(fp):
    m = loadmat(fp); k = [k for k in m if not k.startswith("__")]
    return np.asarray(m[k[0]], dtype=float).reshape(-1, 5) if k else np.zeros((0, 5))   # [id, x, y, w, h]

def collect(cam, n):
    out = []
    for f in sorted(glob.glob(f"{PRW}/frames/{cam}*.jpg"))[:n]:
        ap = f"{PRW}/annotations/{os.path.basename(f)}.mat"
        if not os.path.exists(ap): continue
        for pid, x, y, w, h in read_anno(ap):
            if w > 10 and h > 20: out.append((f, int(pid), int(x), int(y), int(w), int(h)))
    return out

ex = sorted(glob.glob(f"{PRW}/frames/{QUERY_CAM}*.jpg"))[0]
print("anno keys:", [k for k in loadmat(f"{PRW}/annotations/{os.path.basename(ex)}.mat") if not k.startswith('__')])
qb, rb = collect(QUERY_CAM, MAXFRAME), collect(RENDER_CAM, MAXFRAME)
shared = {b[1] for b in qb if b[1] > 0} & {b[1] for b in rb if b[1] > 0}
assert shared, f"no identity appears on both {QUERY_CAM} and {RENDER_CAM}; raise MAXFRAME or change cameras"
QID = max(shared, key=lambda i: sum(b[1] == i for b in rb))
print(f"{len(qb)} boxes on {QUERY_CAM}, {len(rb)} on {RENDER_CAM}; query id {QID} appears "
      f"{sum(b[1]==QID for b in rb)} times on {RENDER_CAM}")

@torch.no_grad()
def embed(items, bs=96):
    out = []
    for s in range(0, len(items), bs):
        ims = torch.stack([tf(Image.open(f).convert("RGB").crop((x, y, x+w, y+h))) for f, _, x, y, w, h in items[s:s+bs]])
        with torch.autocast(DEV, enabled=DEV == "cuda"):
            _, fb, _ = model(ims.to(DEV))
        out.append(F.normalize(fb.float(), dim=1).cpu())
    return torch.cat(out)

qbox = next(b for b in qb if b[1] == QID)
qfeat = embed([qbox])[0]
sims = (embed(rb) @ qfeat).numpy()
print("embedded gallery:", len(rb), "| score range %.2f-%.2f" % (sims.min(), sims.max()))

# ---------- render ----------
byframe = {}
for i, b in enumerate(rb): byframe.setdefault(b[0], []).append(i)
f0, _, qx, qy, qw, qh = qbox
thumb = Image.open(f0).convert("RGB").crop((qx, qy, qx+qw, qy+qh)).resize((80, 160))
OUT = "/kaggle/working/reid_demo_psd_iter1.mp4"
w = imageio.get_writer(OUT, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
for f in sorted(glob.glob(f"{PRW}/frames/{RENDER_CAM}*.jpg"))[:MAXFRAME]:
    im = Image.open(f).convert("RGB"); d = ImageDraw.Draw(im)
    for i in byframe.get(f, []):
        _, pid, x, y, bw, bh = rb[i]; s = float(sims[i]); hit = s >= THRESH
        d.rectangle([x, y, x+bw, y+bh], outline=(0, 220, 0) if hit else (95, 95, 95), width=5 if hit else 2)
        if hit:
            d.rectangle([x, y-20, x+120, y], fill=(0, 220, 0)); d.text((x+4, y-16), f"MATCH {s:.2f}", fill=(0, 0, 0))
    im.paste(thumb, (12, 12)); d = ImageDraw.Draw(im)
    d.rectangle([12, 12, 92, 172], outline=(255, 200, 0), width=4)
    d.text((12, 178), f"QUERY id {QID} (from {QUERY_CAM})", fill=(255, 200, 0))
    d.text((12, 194), f"searching {os.path.basename(f)}", fill=(255, 255, 255))
    sc = WIDTH / im.width
    w.append_data(np.array(im.resize((WIDTH, int(im.height * sc)))))
w.close()
mb = os.path.getsize(OUT) / 1e6; print(f"wrote {OUT}  {mb:.1f} MB")

from IPython.display import HTML, display
if mb < 40:
    display(HTML(f'<video width="900" controls><source src="data:video/mp4;base64,{base64.b64encode(open(OUT,"rb").read()).decode()}" type="video/mp4"></video>'))
else:
    print("too large to embed inline - download it from the Output panel")
