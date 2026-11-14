# ============ Build a Hugging Face Space bundle (run on Kaggle) ============
import os, sys, glob, shutil, zipfile, subprocess, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.io import loadmat
import torchvision.transforms as T
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"], check=False)
import timm

CK   = "/kaggle/input/datasets/lowodsneha/iter1ckpt/psd_iter1.pth"
CAMS = ["c1", "c2"]        # cameras to ship
NFR  = 120                 # frames per camera  (~120 x 2 x 250 KB = 60 MB)
OUT  = "/kaggle/working/hf_space"
DEV  = "cuda" if torch.cuda.is_available() else "cpu"
MEAN, STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)

shutil.rmtree(OUT, ignore_errors=True); os.makedirs(f"{OUT}/frames", exist_ok=True)

class ReID(nn.Module):
    def __init__(self, nid, dim=768, bias=False):
        super().__init__()
        self.enc = timm.create_model("vit_base_patch16_clip_224.openai", pretrained=False, num_classes=0, img_size=(256, 128))
        self.bn = nn.BatchNorm1d(dim); self.bn.bias.requires_grad_(False)
        self.cls = nn.Linear(dim, nid, bias=bias)
    def forward(self, x):
        f = self.enc(x); fb = self.bn(f); return f, fb, self.cls(fb)

sd = torch.load(CK, map_location="cpu")
model = ReID(sd["cls.weight"].shape[0], sd["bn.weight"].shape[0], "cls.bias" in sd).to(DEV).eval()
model.load_state_dict(sd)
tf = T.Compose([T.Resize((256, 128), interpolation=3), T.ToTensor(), T.Normalize(MEAN, STD)])

PRW = None
for r, d, f in os.walk("/kaggle/input"):
    if "frames" in d and "annotations" in d: PRW = r; break
assert PRW, "PRW not found"

def read_anno(fp):
    m = loadmat(fp); k = [k for k in m if not k.startswith("__")]
    return np.asarray(m[k[0]], dtype=float).reshape(-1, 5) if k else np.zeros((0, 5))

BOX = []
for cam in CAMS:
    for f in sorted(glob.glob(f"{PRW}/frames/{cam}*.jpg"))[:NFR]:
        ap = f"{PRW}/annotations/{os.path.basename(f)}.mat"
        if not os.path.exists(ap): continue
        rows = [(int(p), int(x), int(y), int(w), int(h)) for p, x, y, w, h in read_anno(ap) if w > 15 and h > 30]
        if not rows: continue
        Image.open(f).convert("RGB").save(f"{OUT}/frames/{os.path.basename(f)}", quality=82)   # recompress to save space
        for p, x, y, w, h in rows: BOX.append((os.path.basename(f), cam, p, x, y, w, h))
print(f"{len(BOX)} people in {len(set(b[0] for b in BOX))} frames")

@torch.no_grad()
def embed(imgs, bs=96):
    o = []
    for s in range(0, len(imgs), bs):
        x = torch.stack([tf(i) for i in imgs[s:s+bs]]).to(DEV)
        with torch.autocast(DEV, enabled=DEV == "cuda"): _, fb, _ = model(x)
        o.append(F.normalize(fb.float(), dim=1).cpu())
    return torch.cat(o)

feats = torch.cat([embed([Image.open(f"{OUT}/frames/{b[0]}").convert("RGB").crop((b[3], b[4], b[3]+b[5], b[4]+b[6]))
                          for b in BOX[s:s+256]]) for s in range(0, len(BOX), 256)])
torch.save({"BOX": BOX, "feats": feats}, f"{OUT}/index.pt")
shutil.copy(CK, f"{OUT}/psd_iter1.pth")
print("index", tuple(feats.shape))

open(f"{OUT}/requirements.txt", "w").write("torch\ntorchvision\ntimm\ngradio\nnumpy\npillow\nimageio\nimageio-ffmpeg\n")
open(f"{OUT}/README.md", "w").write("""---
title: Person Re-ID (Platonic Self-Distillation)
emoji: walking
colorFrom: indigo
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

Person re-identification demo. CLIP ViT-B/16 + BNNeck trained on Market-1501 with
Platonic Self-Distillation. Frames are from the PRW dataset and are included here
under research use only; this Space is private and must not be made public.
""")

APP = '''import os, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import timm, gradio as gr, imageio.v2 as imageio

HERE = os.path.dirname(os.path.abspath(__file__))
MEAN, STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)

class ReID(nn.Module):
    def __init__(self, nid, dim=768, bias=False):
        super().__init__()
        self.enc = timm.create_model("vit_base_patch16_clip_224.openai", pretrained=False, num_classes=0, img_size=(256, 128))
        self.bn = nn.BatchNorm1d(dim); self.bn.bias.requires_grad_(False)
        self.cls = nn.Linear(dim, nid, bias=bias)
    def forward(self, x):
        f = self.enc(x); fb = self.bn(f); return f, fb, self.cls(fb)

sd = torch.load(f"{HERE}/psd_iter1.pth", map_location="cpu")
model = ReID(sd["cls.weight"].shape[0], sd["bn.weight"].shape[0], "cls.bias" in sd)
model.load_state_dict(sd); model.eval()
tf = T.Compose([T.Resize((256, 128), interpolation=3), T.ToTensor(), T.Normalize(MEAN, STD)])

ix = torch.load(f"{HERE}/index.pt")
BOX, FEAT = ix["BOX"], ix["feats"]
CAMS = sorted({b[1] for b in BOX})
IDENT = [i for i, b in enumerate(BOX) if b[2] > 0]

def fpath(n): return f"{HERE}/frames/{n}"
def crop(b): return Image.open(fpath(b[0])).convert("RGB").crop((b[3], b[4], b[3]+b[5], b[4]+b[6]))

def font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",):
        if os.path.exists(p): return ImageFont.truetype(p, sz)
    return ImageFont.load_default()

@torch.no_grad()
def embed_one(img):
    with torch.no_grad():
        _, fb, _ = model(tf(img.convert("RGB")).unsqueeze(0))
    return F.normalize(fb.float(), dim=1)[0]

def q_from_index(i): return crop(BOX[IDENT[int(i)]])

def search(img, k):
    if img is None: return [], "pick or upload a person first"
    s = (FEAT @ embed_one(img)).numpy(); order = np.argsort(-s)[:int(k)]
    return ([(crop(BOX[j]), f"{BOX[j][1]} - id {BOX[j][2]} - {s[j]:.3f}") for j in order],
            f"scores over {len(BOX)} people: {s.min():.3f} to {s.max():.3f}; top 0.5% is above {np.percentile(s,99.5):.2f}")

def trace(img, cam, thresh, width):
    if img is None: return None, "pick or upload a person first"
    q = embed_one(img); sel = [i for i, b in enumerate(BOX) if b[1] == cam]
    s = (FEAT[sel] @ q).numpy()
    per = {}
    for n, i in enumerate(sel): per.setdefault(BOX[i][0], []).append((i, float(s[n])))
    out = "/tmp/trace.mp4"
    w = imageio.get_writer(out, fps=6, codec="libx264", quality=6, macro_block_size=1)
    hits = 0
    for fn in sorted(per):
        im = Image.open(fpath(fn)).convert("RGB"); W = im.width
        lw, fs = max(3, W // 300), max(14, W // 60); d = ImageDraw.Draw(im); ft = font(fs)
        for i, sc in per[fn]:
            _, _, _, x, y, bw, bh = BOX[i]; hit = sc >= thresh; hits += hit
            col = (0, 230, 0) if hit else (0, 170, 255)
            d.rectangle([x, y, x+bw, y+bh], outline=col, width=lw * (2 if hit else 1))
            lab = ("MATCH " if hit else "") + f"{sc:.2f}"
            d.rectangle([x, y-fs-6, x + d.textlength(lab, font=ft) + 8, y], fill=col)
            d.text((x+4, y-fs-4), lab, fill=(0, 0, 0), font=ft)
        th = img.convert("RGB").resize((int(W*0.09), int(W*0.18))); im.paste(th, (12, 12))
        d = ImageDraw.Draw(im); d.rectangle([12, 12, 12+th.width, 12+th.height], outline=(255, 200, 0), width=lw)
        d.text((12, 16+th.height), f"QUERY - searching {cam}", fill=(255, 200, 0), font=ft)
        sc2 = int(width) / W
        w.append_data(np.array(im.resize((int(width), int(im.height*sc2)))))
    w.close()
    return out, f"{len(per)} frames on {cam}, {len(sel)} people, {hits} above {thresh:.2f} (range {s.min():.3f}-{s.max():.3f})"

with gr.Blocks(title="Person re-ID") as app:
    gr.Markdown("# Person re-identification\\nCLIP ViT-B/16 + BNNeck, Market-1501 + Platonic Self-Distillation. "
                f"Searching {len(BOX)} people across cameras {', '.join(CAMS)}.")
    with gr.Row():
        with gr.Column(scale=1):
            up = gr.Image(type="pil", label="Upload a person crop", height=280)
            pick = gr.Slider(0, len(IDENT)-1, value=0, step=1, label="or pick someone in the footage")
            pick.release(q_from_index, pick, up)
            gr.Button("Use this person").click(q_from_index, pick, up)
        with gr.Column(scale=3):
            with gr.Tab("Find matches"):
                k = gr.Slider(5, 30, value=12, step=1, label="results")
                g = gr.Gallery(label="most similar people", columns=6, height=430); m1 = gr.Markdown()
                gr.Button("Search", variant="primary").click(search, [up, k], [g, m1])
            with gr.Tab("Trace through video"):
                with gr.Row():
                    cam = gr.Dropdown(CAMS, value=CAMS[-1], label="camera")
                    th = gr.Slider(0.2, 0.95, value=0.6, step=0.01, label="threshold")
                    wd = gr.Slider(640, 1280, value=960, step=160, label="width")
                vid = gr.Video(label="green = match, blue = others", height=430); m2 = gr.Markdown()
                gr.Button("Render video", variant="primary").click(trace, [up, cam, th, wd], [vid, m2])

app.launch()
'''
open(f"{OUT}/app.py", "w").write(APP)

z = "/kaggle/working/hf_space.zip"
with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:
    for r, _, fs in os.walk(OUT):
        for f in fs: p = os.path.join(r, f); zf.write(p, os.path.relpath(p, OUT))
tot = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(OUT) for f in fs)
print(f"\nbundle: {tot/1e6:.0f} MB total ({'OK' if tot < 9e8 else 'TOO BIG for the 1 GB Space cap - lower NFR'})")
for r, _, fs in os.walk(OUT):
    for f in sorted(fs)[:3]: print(f"  {os.path.getsize(os.path.join(r,f))/1e6:8.1f} MB  {os.path.relpath(os.path.join(r,f), OUT)}")
print(f"\nzip at {z} ({os.path.getsize(z)/1e6:.0f} MB) - download it from the Output panel")
