import os, sys, glob, time, zlib, torch, numpy as np
from PIL import Image
from multiprocessing import Pool
sys.path.insert(0, "/kaggle/working"); import psd_reid as P
try: from imagecorruptions import corrupt
except ImportError:
    os.system("pip install -q imagecorruptions"); from imagecorruptions import corrupt

SEVERITIES = [1, 3, 5]
TYPES = ['gaussian_noise','shot_noise','impulse_noise','defocus_blur','motion_blur','zoom_blur',
         'snow','frost','fog','brightness','contrast','elastic_transform','pixelate','jpeg_compression']

def _one(job):
    src, dst, sev, seed = job
    if os.path.exists(dst): return
    rng = np.random.RandomState(seed)                       # seeded by file name -> same corruption for every model
    name = TYPES[rng.randint(len(TYPES))]
    a = np.array(Image.open(src).convert("RGB"))
    try: a = corrupt(a, corruption_name=name, severity=sev)
    except Exception: pass
    Image.fromarray(a).save(dst, quality=95)

def build(market, sev):
    out = f"/kaggle/working/market_c_s{sev}"
    jobs = []
    for sub in ("query", "bounding_box_test"):
        os.makedirs(f"{out}/{sub}", exist_ok=True)
        for s in sorted(glob.glob(f"{market}/{sub}/*.jpg")):
            b = os.path.basename(s)
            jobs.append((s, f"{out}/{sub}/{b}", sev, zlib.crc32(b.encode())))
    todo = [j for j in jobs if not os.path.exists(j[1])]
    if todo:
        t = time.time()
        with Pool(os.cpu_count()) as p: p.map(_one, todo, chunksize=64)
        print(f"  severity {sev}: built {len(todo)} images in {time.time()-t:.0f}s")
    else: print(f"  severity {sev}: already built")
    return out

def load_model(path):
    sd = torch.load(path, map_location="cpu")
    m = P.ReID(sd["cls.weight"].shape[0], "vit_base_patch16_clip_224.openai", (256,128), pretrained=False)
    m.load_state_dict(sd); return m.to(P.DEV).eval()

print("building corrupted Market-1501 test sets")
SETS = {"clean (severity 0)": MARKET}
for s in SEVERITIES: SETS[f"corrupted severity {s}"] = build(MARKET, s)

CKPTS = {os.path.basename(p)[:-4]: p for p in sorted(glob.glob("/kaggle/working/psd_reid_out/ckpt/*.pth"))}
_, tf = P.transforms((256, 128)); print("\ncheckpoints:", list(CKPTS) or "NONE FOUND")
rows = []
for name, path in CKPTS.items():
    model = load_model(path)
    for sname, root in SETS.items():
        q, g = P.read_split(root, "query"), P.read_split(root, "bounding_box_test")
        mAP, r1, r5, r10, _ = P.evaluate(model, q, g, tf)
        rows.append((name, sname, mAP, r1)); print(f"{name:12s} {sname:22s} mAP {mAP:5.2f}  R1 {r1:5.2f}")
    del model; torch.cuda.empty_cache()
print("\nmodel,setting,mAP,R1")
for r in rows: print(f"{r[0]},{r[1]},{r[2]:.2f},{r[3]:.2f}")
