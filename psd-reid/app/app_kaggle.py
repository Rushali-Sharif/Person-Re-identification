# ================= Person re-ID application (psd_iter1) =================
import os, sys, glob, subprocess, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from scipy.io import loadmat
import torchvision.transforms as T
for p in ("imageio", "imageio-ffmpeg", "timm", "gradio"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", p], check=False)
import timm, gradio as gr, imageio.v2 as imageio

CK    = "/kaggle/input/datasets/lowodsneha/iter1ckpt/psd_iter1.pth"
CAMS  = ["c1", "c2", "c3", "c4"]      # cameras to index
NFRAME = 300                           # frames per camera
DEV   = "cuda" if torch.cuda.is_available() else "cpu"
MEAN, STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)

def font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()

# ---------------- model ----------------
class ReID(nn.Module):
    def __init__(self, nid, dim=768, bias=False):
        super().__init__()
        self.enc = timm.create_model("vit_base_patch16_clip_224.openai", pretrained=False, num_classes=0, img_size=(256, 128))
        self.bn = nn.BatchNorm1d(dim); self.bn.bias.requires_grad_(False)
        self.cls = nn.Linear(dim, nid, bias=bias)
    def forward(self, x):
        f = self.enc(x); fb = self.bn(f); return f, fb, self.cls(fb)

assert os.path.exists(CK), f"checkpoint missing: {CK}"
sd = torch.load(CK, map_location="cpu")
model = ReID(sd["cls.weight"].shape[0], sd["bn.weight"].shape[0], "cls.bias" in sd)
model.load_state_dict(sd); model = model.to(DEV).eval()
tf = T.Compose([T.Resize((256, 128), interpolation=3), T.ToTensor(), T.Normalize(MEAN, STD)])
print(f"model loaded from {os.path.basename(CK)} ({sd['cls.weight'].shape[0]} training identities)")

# ---------------- data ----------------
PRW = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "frames" in dirs and "annotations" in dirs: PRW = root; break
assert PRW, "PRW not found under /kaggle/input"

def read_anno(fp):
    m = loadmat(fp); k = [k for k in m if not k.startswith("__")]
    return np.asarray(m[k[0]], dtype=float).reshape(-1, 5) if k else np.zeros((0, 5))

BOX = []                                              # (frame, cam, pid, x, y, w, h)
for cam in CAMS:
    for f in sorted(glob.glob(f"{PRW}/frames/{cam}*.jpg"))[:NFRAME]:
        ap = f"{PRW}/annotations/{os.path.basename(f)}.mat"
        if not os.path.exists(ap): continue
        for pid, x, y, w, h in read_anno(ap):
            if w > 15 and h > 30: BOX.append((f, cam, int(pid), int(x), int(y), int(w), int(h)))
print(f"indexed {len(BOX)} people across {len(CAMS)} cameras")

def crop(b): return Image.open(b[0]).convert("RGB").crop((b[3], b[4], b[3]+b[5], b[4]+b[6]))

@torch.no_grad()
def embed_imgs(imgs, bs=96):
    out = []
    for s in range(0, len(imgs), bs):
        x = torch.stack([tf(i) for i in imgs[s:s+bs]]).to(DEV)
        with torch.autocast(DEV, enabled=DEV == "cuda"):
            _, fb, _ = model(x)
        out.append(F.normalize(fb.float(), dim=1).cpu())
    return torch.cat(out)

CACHE = "/kaggle/working/prw_index.pt"
_c = torch.load(CACHE) if os.path.exists(CACHE) else None
if _c is not None and _c["n"] == len(BOX):
    B = _c["B"]; print("loaded cached index")
else:
    B = torch.cat([embed_imgs([crop(b) for b in BOX[s:s+256]]) for s in range(0, len(BOX), 256)])
    torch.save({"B": B, "n": len(BOX)}, CACHE)
print("index ready", tuple(B.shape))
IDENT = [i for i, b in enumerate(BOX) if b[2] > 0]

# ---------------- drawing ----------------
def draw(im, items, thresh, qthumb, title):
    W = im.width; lw = max(3, W // 300); fs = max(14, W // 60); d = ImageDraw.Draw(im); ft = font(fs)
    for (x, y, w, h, s) in items:
        hit = s >= thresh
        col = (0, 230, 0) if hit else (0, 170, 255)
        d.rectangle([x, y, x+w, y+h], outline=col, width=lw * (2 if hit else 1))
        lab = f"{'MATCH ' if hit else ''}{s:.2f}"
        tw = d.textlength(lab, font=ft)
        d.rectangle([x, y - fs - 6, x + tw + 8, y], fill=col)
        d.text((x + 4, y - fs - 4), lab, fill=(0, 0, 0), font=ft)
    th = qthumb.resize((int(W * 0.09), int(W * 0.18)))
    im.paste(th, (12, 12)); d = ImageDraw.Draw(im)
    d.rectangle([12, 12, 12 + th.width, 12 + th.height], outline=(255, 200, 0), width=lw)
    d.text((12, 16 + th.height), title, fill=(255, 200, 0), font=ft)
    return im

# ---------------- actions ----------------
def q_from_index(i):
    return crop(BOX[IDENT[int(i)]])

def search(img, k):
    if img is None: return [], "upload a person crop or pick one with the slider"
    f = embed_imgs([img.convert("RGB")])[0]
    s = (B @ f).numpy(); order = np.argsort(-s)[:int(k)]
    out = [(crop(BOX[j]), f"{BOX[j][1]} · id {BOX[j][2]} · {s[j]:.3f}") for j in order]
    return out, f"score range over {len(BOX)} people: {s.min():.3f} to {s.max():.3f} — matches above ~{np.percentile(s,99.5):.2f} are the top 0.5%"

def trace(img, cam, thresh, nframe, width):
    if img is None: return None, "upload a person crop or pick one with the slider"
    f = embed_imgs([img.convert("RGB")])[0]
    sel = [i for i, b in enumerate(BOX) if b[1] == cam]
    s = (B[sel] @ f).numpy()
    per = {}
    for n, i in enumerate(sel): per.setdefault(BOX[i][0], []).append((i, float(s[n])))
    frames = sorted(glob.glob(f"{PRW}/frames/{cam}*.jpg"))[:int(nframe)]
    out = "/kaggle/working/trace.mp4"
    w = imageio.get_writer(out, fps=6, codec="libx264", quality=6, macro_block_size=1)
    hits = 0
    for fr in frames:
        im = Image.open(fr).convert("RGB")
        items = [(BOX[i][3], BOX[i][4], BOX[i][5], BOX[i][6], sc) for i, sc in per.get(fr, [])]
        hits += sum(1 for it in items if it[4] >= thresh)
        im = draw(im, items, thresh, img.convert("RGB"), f"QUERY — searching {cam}")
        sc = int(width) / im.width
        w.append_data(np.array(im.resize((int(width), int(im.height * sc)))))
    w.close()
    return out, f"{len(frames)} frames on {cam}, {len(sel)} people seen, {hits} boxes above {thresh:.2f} (score range {s.min():.3f}–{s.max():.3f})"

# ---------------- UI ----------------
with gr.Blocks(title="Person re-ID - PSD iteration 1") as app:
    gr.Markdown("# Person re-identification\nCLIP ViT-B/16 + BNNeck, trained on Market-1501 with Platonic Self-Distillation. "
                f"Searching {len(BOX)} people across cameras {', '.join(CAMS)} of PRW (the same camera network as Market-1501).")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Choose who to look for")
            up = gr.Image(type="pil", label="Upload a person crop", height=280)
            gr.Markdown("*or pick someone already in the footage:*")
            pick = gr.Slider(0, len(IDENT) - 1, value=0, step=1, label="person index")
            pick.release(q_from_index, pick, up)
            gr.Button("Use this person").click(q_from_index, pick, up)
        with gr.Column(scale=3):
            gr.Markdown("### 2. See the result")
            with gr.Tab("Find matches"):
                k = gr.Slider(5, 30, value=12, step=1, label="how many results")
                g = gr.Gallery(label="most similar people found", columns=6, height=440)
                m1 = gr.Markdown()
                gr.Button("Search", variant="primary").click(search, [up, k], [g, m1])
            with gr.Tab("Trace through video"):
                with gr.Row():
                    cam = gr.Dropdown(CAMS, value=CAMS[-1], label="camera to search")
                    th  = gr.Slider(0.2, 0.95, value=0.60, step=0.01, label="match threshold")
                    nf  = gr.Slider(60, NFRAME, value=200, step=20, label="frames")
                    wd  = gr.Slider(640, 1280, value=960, step=160, label="video width")
                vid = gr.Video(label="green = match, blue = other people, score on every box", height=430)
                m2 = gr.Markdown()
                gr.Button("Render video", variant="primary").click(trace, [up, cam, th, nf, wd], [vid, m2])

app.launch(share=True, allowed_paths=["/kaggle"])
