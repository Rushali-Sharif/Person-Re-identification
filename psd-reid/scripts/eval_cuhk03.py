import os, glob, sys, torch
from scipy.io import loadmat
sys.path.insert(0, "/kaggle/working"); import psd_reid as P

CU = "/kaggle/input/datasets/priyanagda/cuhk03/archive"

def read_cuhk03(root, mode):
    mat = loadmat(f"{root}/cuhk03_new_protocol_config_{mode}.mat")
    print(f"  {mode}: mat keys = {[k for k in mat if not k.startswith('__')]}")
    img_dir, filelist = f"{root}/images_{mode}", mat["filelist"].flatten()
    labels = mat["labels"].flatten()
    def grab(key):                                           # camera id from the 3rd filename field, as torchreid does
        out = []
        for i in mat[key].flatten() - 1:
            n = str(filelist[i][0])
            out.append((f"{img_dir}/{n}", int(labels[i]), int(n.split("_")[2]) - 1))
        return out
    q, g = grab("query_idx"), grab("gallery_idx")
    print(f"  {mode}: {len(q)} query / {len(g)} gallery, {len({p for _,p,_ in q})} query ids")
    return q, g

def ev(model, q, g, tf):
    r = P.evaluate(model, q, g, tf)
    return (r["mAP"], r["R1"], r["R5"], r["R10"]) if isinstance(r, dict) else r[:4]

def load_model(path):
    sd = torch.load(path, map_location="cpu")
    m = P.ReID(sd["cls.weight"].shape[0], "vit_base_patch16_clip_224.openai", (256,128), pretrained=False)
    m.load_state_dict(sd); return m.to(P.DEV).eval()

SPLITS = {}
for mode in ("detected", "labeled"):
    try: SPLITS[f"cuhk03-{mode}"] = read_cuhk03(CU, mode)
    except Exception as e: print(f"  skip {mode}: {type(e).__name__} {e}")

CKPTS = {os.path.basename(p)[:-4]: p for p in sorted(glob.glob("/kaggle/working/psd_reid_out/ckpt/*.pth"))}
_, tf = P.transforms((256, 128)); print("\ncheckpoints:", list(CKPTS) or "NONE FOUND")
rows = []
for name, path in CKPTS.items():
    model = load_model(path)
    for sname, (q, g) in SPLITS.items():
        mAP, r1, r5, r10 = ev(model, q, g, tf)
        rows.append((name, sname, mAP, r1, r5, r10))
        print(f"{name:12s} {sname:18s} mAP {mAP:5.2f}  R1 {r1:5.2f}  R5 {r5:5.2f}  R10 {r10:5.2f}")
    del model; torch.cuda.empty_cache()
print("\nmodel,dataset,mAP,R1,R5,R10")
for r in rows: print(f"{r[0]},{r[1]},{r[2]:.2f},{r[3]:.2f},{r[4]:.2f},{r[5]:.2f}")
