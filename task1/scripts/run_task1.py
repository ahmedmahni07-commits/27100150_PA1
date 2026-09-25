"""
Task 1 -- single end-to-end driver that regenerates every number/figure used
in the report and SAVES them (previous scripts only printed to stdout).

Run from the repo root:
    python -m task1.scripts.run_task1

Outputs (all under task1/results/):
    task1_results.json          every metric below, machine-readable
    translation_curve.png       accuracy / consistency vs displacement
    tsne_clean_vs_transformed.png
    cue_conflict_examples.png   informative agreements / disagreements
    cue_conflict_predictions.csv per-image predictions for all models

Protocol (matches the assignment):
  * frozen backbones: ResNet-50 (IMAGENET1K_V2, GAP feature), ViT-B/16
    (IMAGENET1K_V1, final CLS token), OpenCLIP ViT-B-32 'openai' (L2-normalised
    image embedding); one linear head per backbone trained on the cached
    train features (AdamW lr 1e-3, wd 1e-4, <=50 epochs, early stop patience 5
    on val accuracy, seed 6304). CLIP zero-shot uses "a photo of a {class}.".
  * every intervention is built on the SAME 224x224 RGB image (bilinear
    resize of the 96x96 STL-10 image) before each model's own normalisation,
    so all models receive identical pixels.
  * cue conflicts: the stylised AdaIN images already on disk; a pixel-level
    rejection rule (fixed below, independent of any model) is applied first.
"""
from __future__ import annotations

import copy
import csv
import glob
import json
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE
from torchvision import datasets, models, transforms

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from common.seed import set_seed  # noqa: E402

T1 = os.path.join(ROOT, "task1")
DATA = os.path.join(T1, "data")
FEAT_DIR = os.path.join(T1, "features")
OUT = os.path.join(T1, "results")
os.makedirs(OUT, exist_ok=True)

SEED = 6304
CLASSES = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]
C2I = {c: i for i, c in enumerate(CLASSES)}
MODEL_NAMES = ["resnet50", "vit_b_16", "openclip"]
PRETTY = {"resnet50": "ResNet-50", "vit_b_16": "ViT-B/16", "openclip": "CLIP (head)", "clip_zs": "CLIP (zero-shot)"}
DELTAS = [0, 8, 16, 32]

IMNET_NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
CLIP_NORM = transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])

# Cue-conflict visual rejection rule (fixed before any model is evaluated):
#   reject a stylised image if (a) it is nearly flat (pixel std < 0.04) -- the
#   decoder produced a wash-out with no usable content or texture -- or
#   (b) its mean Sobel edge magnitude is < 25% of its content image's, i.e.
#   the content/shape cue was essentially erased by stylisation.
REJ_MIN_STD = 0.04
REJ_MIN_EDGE_RATIO = 0.25


def device_pick():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEV = device_pick()


# --------------------------------------------------------------------------- data
def load_test_subset():
    splits = torch.load(os.path.join(DATA, "splits", "stl10_splits.pt"), weights_only=False)
    base = datasets.STL10(root=os.path.join(DATA, "raw"), split="test", download=False)
    idx = np.asarray(splits["test_subset_indices"])
    to_t = transforms.ToTensor()
    rs = transforms.Resize((224, 224))
    imgs, labels = [], []
    for i in idx:
        im, y = base[int(i)]
        imgs.append(to_t(rs(im)))
        labels.append(int(y))
    return torch.stack(imgs), np.array(labels), idx


def grayscale(x):
    return TF.rgb_to_grayscale(x, num_output_channels=3)


def hue_rotate(x, h=0.3):  # +0.3 of the hue circle ~ +108 degrees
    return torch.stack([TF.adjust_hue(im, h) for im in x])


def translate(x, dx, dy, delta):
    p = F.pad(x, (delta, delta, delta, delta), mode="reflect")
    sx, sy = delta - dx, delta - dy
    return p[:, :, sy:sy + 224, sx:sx + 224]


def load_patch_shuffle(n):
    files = sorted(glob.glob(os.path.join(DATA, "patch_shuffle", "*.png")))
    assert len(files) == n, f"expected {n} patch-shuffled images, found {len(files)}"
    to_t = transforms.ToTensor()
    imgs, labels = [], []
    for f in files:
        labels.append(int(re.search(r"_label_(\d+)\.png", f).group(1)))
        imgs.append(to_t(Image.open(f).convert("RGB")))
    return torch.stack(imgs), np.array(labels)


def sobel_mag(x):
    g = TF.rgb_to_grayscale(x.unsqueeze(0))
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    return torch.sqrt(F.conv2d(g, kx, padding=1) ** 2 + F.conv2d(g, ky, padding=1) ** 2).mean().item()


def load_cue_conflicts():
    pat = re.compile(r"(\d+)_shape_([a-z]+)_tex_([a-z]+)_stylized\.png")
    to_t = transforms.ToTensor()
    rs = transforms.Resize((224, 224))
    files = sorted(glob.glob(os.path.join(DATA, "cue_conflicts_stylized", "*_stylized.png")))
    kept, rejected = [], []
    for f in files:
        name = os.path.basename(f)
        m = pat.match(name)
        shape, tex = m.group(2), m.group(3)
        content_path = os.path.join(DATA, "cue_conflicts_raw", name.replace("_stylized.png", "_content.jpg"))
        sty = to_t(rs(Image.open(f).convert("RGB")))
        con = to_t(rs(Image.open(content_path).convert("RGB")))
        std = sty.std().item()
        ratio = sobel_mag(sty) / max(sobel_mag(con), 1e-8)
        rec = dict(file=name, shape=shape, texture=tex, pair="-".join(sorted([shape, tex])),
                   std=round(std, 4), edge_ratio=round(ratio, 4))
        if std < REJ_MIN_STD or ratio < REJ_MIN_EDGE_RATIO:
            rejected.append(rec)
        else:
            kept.append((rec, sty, con))
    return kept, rejected


# --------------------------------------------------------------------------- models
def build_backbones():
    import open_clip

    r = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    r.fc = nn.Identity()
    v = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    v.heads = nn.Identity()
    c, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tok = open_clip.get_tokenizer("ViT-B-32")
    return {"resnet50": r.eval().to(DEV), "vit_b_16": v.eval().to(DEV), "openclip": c.eval().to(DEV)}, tok


@torch.no_grad()
def embed(name, model, x, bs=50):
    norm = CLIP_NORM if name == "openclip" else IMNET_NORM
    out = []
    for i in range(0, len(x), bs):
        b = norm(x[i:i + bs]).to(DEV)
        f = model.encode_image(b) if name == "openclip" else model(b)
        f = f.float()
        if name == "openclip":
            f = f / f.norm(dim=-1, keepdim=True)
        out.append(f.cpu())
    return torch.cat(out)


def train_head(name):
    set_seed(SEED)
    ld = lambda s: torch.load(os.path.join(FEAT_DIR, f"{name}_{s}.pt"), weights_only=False)
    tr, va = ld("train"), ld("val")
    Xt, yt = tr["features"].flatten(1).float(), tr["labels"].long()
    Xv, yv = va["features"].flatten(1).float(), va["labels"].long()
    head = nn.Linear(Xt.shape[1], 10)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(SEED)
    best, best_w, bad, ep_used = -1.0, None, 0, 0
    for ep in range(50):
        head.train()
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), 256):
            j = perm[i:i + 256]
            opt.zero_grad()
            F.cross_entropy(head(Xt[j]), yt[j]).backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            acc = (head(Xv).argmax(1) == yv).float().mean().item()
        ep_used = ep + 1
        if acc > best:
            best, best_w, bad = acc, copy.deepcopy(head.state_dict()), 0
        else:
            bad += 1
            if bad >= 5:
                break
    head.load_state_dict(best_w)
    return head.eval(), {"best_val_acc": best, "epochs_run": ep_used}


# --------------------------------------------------------------------------- metrics
def predict(logits):
    p = logits.softmax(1)
    conf, pred = p.max(1)
    return pred.numpy(), conf.numpy()


def summary(pred, conf, y):
    return {"acc": float((pred == y).mean() * 100), "macro_f1": float(f1_score(y, pred, average="macro")),
            "mean_max_conf": float(conf.mean())}


def cosine(a, b):
    return float(F.cosine_similarity(a, b, dim=1).mean())


def main():
    set_seed(SEED)
    print(f"device: {DEV}")
    X, y, subset_idx = load_test_subset()
    n = len(y)
    per_class = np.bincount(y, minlength=10).tolist()
    print(f"test subset: {n} images, per class {per_class}")

    backbones, tok = build_backbones()
    heads, head_info = {}, {}
    for m in MODEL_NAMES:
        heads[m], head_info[m] = train_head(m)
        print(f"head {m}: {head_info[m]}")

    with torch.no_grad():
        txt = backbones["openclip"].encode_text(tok([f"a photo of a {c}." for c in CLASSES]).to(DEV)).float().cpu()
        txt = txt / txt.norm(dim=-1, keepdim=True)
        scale = backbones["openclip"].logit_scale.exp().item()

    def logits_for(m, feats):
        if m == "clip_zs":
            return scale * feats @ txt.T
        with torch.no_grad():
            return heads[m](feats)

    EVAL = MODEL_NAMES + ["clip_zs"]
    feat_of = lambda m: "openclip" if m == "clip_zs" else m

    # ---- build all intervention inputs once (identical pixels for every model)
    conds = {"clean": X, "grayscale": grayscale(X), "hue_rot": hue_rotate(X)}
    for d in DELTAS[1:]:
        for tag, (dx, dy) in {"R": (d, 0), "L": (-d, 0), "D": (0, d), "U": (0, -d)}.items():
            conds[f"shift{d}{tag}"] = translate(X, dx, dy, d)
    Xs, ys = load_patch_shuffle(n)
    assert (ys == y).all(), "patch-shuffle labels are not aligned with the test subset order"
    conds["patch_shuffle"] = Xs

    kept, rejected = load_cue_conflicts()
    print(f"cue conflicts: kept {len(kept)}, rejected {len(rejected)}")
    Xcc = torch.stack([k[1] for k in kept])
    Xcon = torch.stack([k[2] for k in kept])

    feats = {}
    for m in MODEL_NAMES:
        feats[m] = {}
        for c, xc in conds.items():
            feats[m][c] = embed(m, backbones[m], xc)
        feats[m]["cue_conflict"] = embed(m, backbones[m], Xcc)
        feats[m]["cue_content"] = embed(m, backbones[m], Xcon)
        print(f"features done: {m}")

    R = {"meta": {"device": str(DEV), "seed": SEED, "n_test": n, "test_per_class": per_class,
                  "heads": head_info, "hue_shift": 0.3, "deltas": DELTAS, "patch_grid": 4,
                  "cue_rejection_rule": {"min_std": REJ_MIN_STD, "min_edge_ratio": REJ_MIN_EDGE_RATIO}}}

    # ---- 1. clean baseline + 2. colour + 5. patch shuffle
    preds = {}
    R["clean"], R["color"], R["patch_shuffle"] = {}, {}, {}
    for m in EVAL:
        preds[m] = {}
        for c in conds:
            preds[m][c] = predict(logits_for(m, feats[feat_of(m)][c]))
        R["clean"][m] = summary(*preds[m]["clean"], y)
        pc = preds[m]["clean"][0]
        for c in ["grayscale", "hue_rot"]:
            p, cf = preds[m][c]
            s = summary(p, cf, y)
            R["color"].setdefault(c, {})[m] = {**s, "delta_acc": s["acc"] - R["clean"][m]["acc"],
                                               "consistency": float((p == pc).mean() * 100)}
        p, cf = preds[m]["patch_shuffle"]
        s = summary(p, cf, y)
        wrong = p != y
        R["patch_shuffle"][m] = {**s, "delta_acc": s["acc"] - R["clean"][m]["acc"],
                                 "consistency": float((p == pc).mean() * 100),
                                 "mean_conf_when_wrong": float(cf[wrong].mean()) if wrong.any() else None,
                                 "frac_wrong_with_conf_gt_0.8": float((cf[wrong] > 0.8).mean()) if wrong.any() else None,
                                 "pred_histogram": np.bincount(p, minlength=10).tolist()}

    # ---- 4. translation
    R["translation"] = {}
    for m in EVAL:
        pc = preds[m]["clean"][0]
        rows = []
        for d in DELTAS:
            if d == 0:
                rows.append({"delta": 0, "acc": R["clean"][m]["acc"], "consistency": 100.0, "per_dir": {}})
                continue
            per = {}
            for tag in "RLDU":
                p, _ = preds[m][f"shift{d}{tag}"]
                per[tag] = {"acc": float((p == y).mean() * 100), "consistency": float((p == pc).mean() * 100)}
            rows.append({"delta": d, "acc": float(np.mean([v["acc"] for v in per.values()])),
                         "consistency": float(np.mean([v["consistency"] for v in per.values()])), "per_dir": per})
        R["translation"][m] = rows

    # ---- 3. cue conflict
    R["cue_conflict"] = {"n_candidates": len(kept) + len(rejected), "n_kept": len(kept), "n_rejected": len(rejected),
                         "rejected": rejected, "per_direction_counts": {}, "models": {}}
    for rec, _, _ in kept:
        k = f"{rec['shape']}->{rec['texture']}"
        R["cue_conflict"]["per_direction_counts"][k] = R["cue_conflict"]["per_direction_counts"].get(k, 0) + 1
    shp = np.array([C2I[k[0]["shape"]] for k in kept])
    tex = np.array([C2I[k[0]["texture"]] for k in kept])
    pairs = np.array([k[0]["pair"] for k in kept])
    cc_pred = {}
    for m in EVAL:
        p, cf = predict(logits_for(m, feats[feat_of(m)]["cue_conflict"]))
        pcon, _ = predict(logits_for(m, feats[feat_of(m)]["cue_content"]))
        cc_pred[m] = (p, cf, pcon)
        ns, nt = int((p == shp).sum()), int((p == tex).sum())
        no = len(p) - ns - nt
        per_pair = {}
        for pr in sorted(set(pairs)):
            mk = pairs == pr
            a, b = int((p[mk] == shp[mk]).sum()), int((p[mk] == tex[mk]).sum())
            per_pair[pr] = {"n": int(mk.sum()), "shape": a, "texture": b, "other": int(mk.sum()) - a - b,
                            "shape_bias": 100 * a / (a + b) if a + b else None}
        other_hist = np.bincount(p[(p != shp) & (p != tex)], minlength=10).tolist()
        R["cue_conflict"]["models"][m] = {
            "N_shape": ns, "N_texture": nt, "N_other": no,
            "shape_bias": 100 * ns / (ns + nt) if ns + nt else None,
            "coverage": 100 * (ns + nt) / len(p),
            "content_image_acc": float((pcon == shp).mean() * 100),
            "shape_bias_given_content_correct": (lambda mk: 100 * int((p[mk] == shp[mk]).sum()) /
                                                 max(1, int((p[mk] == shp[mk]).sum() + (p[mk] == tex[mk]).sum())))(pcon == shp),
            "mean_conf": float(cf.mean()), "per_pair": per_pair, "other_pred_histogram": other_hist}
    with open(os.path.join(OUT, "cue_conflict_predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "shape", "texture"] + [f"{m}_pred" for m in EVAL] + [f"{m}_conf" for m in EVAL])
        for i, (rec, _, _) in enumerate(kept):
            w.writerow([rec["file"], rec["shape"], rec["texture"]] + [CLASSES[cc_pred[m][0][i]] for m in EVAL]
                       + [round(float(cc_pred[m][1][i]), 3) for m in EVAL])

    # ---- 6. representation stability + agreement with prediction stability
    R["representation"] = {}
    for m in MODEL_NAMES:
        fc = feats[m]["clean"]
        pc = preds[m]["clean"][0]
        out = {}

        def stab(name, ft, fref, pt, pref):
            cs = F.cosine_similarity(fref, ft, dim=1).numpy()
            same = pt == pref
            out[name] = {"I_T": float(cs.mean()), "I_T_std": float(cs.std()),
                         "I_T_pred_unchanged": float(cs[same].mean()) if same.any() else None,
                         "I_T_pred_changed": float(cs[~same].mean()) if (~same).any() else None,
                         "pred_consistency": float(same.mean() * 100)}

        stab("grayscale", feats[m]["grayscale"], fc, preds[m]["grayscale"][0], pc)
        stab("hue_rot", feats[m]["hue_rot"], fc, preds[m]["hue_rot"][0], pc)
        for d in DELTAS[1:]:
            ft = torch.cat([feats[m][f"shift{d}{t}"] for t in "RLDU"])
            pt = np.concatenate([preds[m][f"shift{d}{t}"][0] for t in "RLDU"])
            stab(f"translate_{d}", ft, torch.cat([fc] * 4), pt, np.concatenate([pc] * 4))
        stab("patch_shuffle", feats[m]["patch_shuffle"], fc, preds[m]["patch_shuffle"][0], pc)
        # cue conflict: clean counterpart = the content (shape-source) image
        stab("cue_conflict", feats[m]["cue_conflict"], feats[m]["cue_content"], cc_pred[m][0], cc_pred[m][2])
        # neighbourhood check in the ORIGINAL feature space: is the transformed
        # image's nearest clean neighbour of the same class?
        fcn = F.normalize(fc, dim=1)
        for c in ["grayscale", "hue_rot", "patch_shuffle"]:
            ftn = F.normalize(feats[m][c], dim=1)
            nn_idx = (ftn @ fcn.T).argmax(1).numpy()
            out[c]["nn_clean_is_own_counterpart"] = float((nn_idx == np.arange(n)).mean() * 100)
            out[c]["nn_clean_same_class"] = float((y[nn_idx] == y).mean() * 100)
        R["representation"][m] = out

    # CLIP zero-shot vs trained head: agreement under each intervention
    R["clip_zs_vs_head"] = {}
    for c in ["clean", "grayscale", "hue_rot", "patch_shuffle"]:
        a, b = preds["openclip"][c][0], preds["clip_zs"][c][0]
        R["clip_zs_vs_head"][c] = {"agreement": float((a == b).mean() * 100),
                                   "head_right_zs_wrong": int(((a == y) & (b != y)).sum()),
                                   "zs_right_head_wrong": int(((b == y) & (a != y)).sum())}
    a, b = cc_pred["openclip"][0], cc_pred["clip_zs"][0]
    R["clip_zs_vs_head"]["cue_conflict"] = {"agreement": float((a == b).mean() * 100)}

    with open(os.path.join(OUT, "task1_results.json"), "w") as f:
        json.dump(R, f, indent=2)
    print("wrote task1/results/task1_results.json")

    # ---------------------------------------------------------------- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for m in EVAL:
        rows = R["translation"][m]
        ax[0].plot(DELTAS, [r["acc"] for r in rows], marker="o", label=PRETTY[m])
        ax[1].plot(DELTAS, [r["consistency"] for r in rows], marker="o", label=PRETTY[m])
    for a, t in zip(ax, ["Top-1 accuracy (%)", "Prediction consistency (%)"]):
        a.set_xlabel("displacement (px, mean of 4 directions)")
        a.set_ylabel(t)
        a.set_xticks(DELTAS)
        a.grid(alpha=0.3)
    ax[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "translation_curve.png"), dpi=200)
    plt.close()

    # t-SNE: one projection per backbone per transform, fit on clean+transformed jointly
    show = [("patch_shuffle", "patch shuffle"), ("grayscale", "grayscale")]
    fig, axes = plt.subplots(len(show), 3, figsize=(12, 7.6))
    cmap = plt.get_cmap("tab10")
    for j, m in enumerate(MODEL_NAMES):
        for i, (c, title) in enumerate(show):
            Z = np.vstack([F.normalize(feats[m]["clean"], dim=1).numpy(), F.normalize(feats[m][c], dim=1).numpy()])
            E = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
                     random_state=SEED).fit_transform(Z)
            a = axes[i, j]
            a.scatter(E[:n, 0], E[:n, 1], c=[cmap(k) for k in y], s=7, marker="o", alpha=0.7)
            a.scatter(E[n:, 0], E[n:, 1], c=[cmap(k) for k in y], s=12, marker="x", alpha=0.7, linewidths=0.8)
            a.set_title(f"{PRETTY[m].replace(' (head)', '')}: clean (o) vs {title} (x)", fontsize=9)
            a.set_xticks([]); a.set_yticks([])
    handles = [plt.Line2D([], [], marker="o", ls="", color=cmap(k), label=CLASSES[k]) for k in range(10)]
    fig.legend(handles=handles, loc="lower center", ncol=10, fontsize=8, frameon=False)
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(os.path.join(OUT, "tsne_clean_vs_transformed.png"), dpi=200)
    plt.close()

    # informative cue-conflict cases: models disagree, or all pick "other"
    P = {m: cc_pred[m][0] for m in EVAL}
    tag = lambda m, i: "S" if P[m][i] == shp[i] else ("T" if P[m][i] == tex[i] else "O")
    groups = {"disagree": [], "all_other": [], "all_shape": [], "all_texture": []}
    for i in range(len(kept)):
        tags = {tag(m, i) for m in MODEL_NAMES}
        key = "disagree" if len(tags) > 1 else {"O": "all_other", "S": "all_shape", "T": "all_texture"}[tags.pop()]
        groups[key].append(i)

    def pick(idx_list, k):
        out, used = [], set()
        for i in idx_list:            # prefer distinct class pairs
            if pairs[i] not in used:
                out.append(i); used.add(pairs[i])
            if len(out) == k:
                return out
        return out + [i for i in idx_list if i not in out][: k - len(out)]

    chosen = pick(groups["disagree"], 3) + pick(groups["all_texture"], 1) + \
        pick(groups["all_other"], 1) + pick(groups["all_shape"], 1)
    R["cue_conflict"]["agreement_groups"] = {k: len(v) for k, v in groups.items()}
    fig, axes = plt.subplots(1, len(chosen), figsize=(2.3 * len(chosen), 3.3))
    for a, i in zip(np.atleast_1d(axes), chosen):
        a.imshow(Xcc[i].permute(1, 2, 0).numpy())
        a.set_xticks([]); a.set_yticks([])
        lines = [f"shape={CLASSES[shp[i]]}, tex={CLASSES[tex[i]]}"]
        for m, s in zip(EVAL, ["R50", "ViT", "CLIPh", "CLIPzs"]):
            lines.append(f"{s}: {CLASSES[P[m][i]]} ({tag(m, i)})")
        a.set_title("\n".join(lines), fontsize=6.5, loc="left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "cue_conflict_examples.png"), dpi=200)
    plt.close()
    R["cue_conflict"]["example_files"] = [kept[i][0]["file"] for i in chosen]
    with open(os.path.join(OUT, "task1_results.json"), "w") as f:
        json.dump(R, f, indent=2)

    # console summary
    print("\n=== CLEAN ===")
    for m in EVAL:
        print(f"{PRETTY[m]:18s} " + " ".join(f"{k}={v:.3f}" for k, v in R["clean"][m].items()))
    print("\n=== COLOUR / SHUFFLE (acc, dAcc, consistency) ===")
    for m in EVAL:
        g, h, s = R["color"]["grayscale"][m], R["color"]["hue_rot"][m], R["patch_shuffle"][m]
        print(f"{PRETTY[m]:18s} gray {g['acc']:.1f} {g['delta_acc']:+.1f} {g['consistency']:.1f} | "
              f"hue {h['acc']:.1f} {h['delta_acc']:+.1f} {h['consistency']:.1f} | "
              f"shuf {s['acc']:.1f} {s['delta_acc']:+.1f} {s['consistency']:.1f}")
    print("\n=== CUE CONFLICT ===")
    for m in EVAL:
        c = R["cue_conflict"]["models"][m]
        print(f"{PRETTY[m]:18s} S={c['N_shape']} T={c['N_texture']} O={c['N_other']} "
              f"bias={c['shape_bias']:.1f} cov={c['coverage']:.1f}")
    print("\n=== TRANSLATION (acc/cons) ===")
    for m in EVAL:
        print(f"{PRETTY[m]:18s} " + " ".join(f"d{r['delta']}:{r['acc']:.1f}/{r['consistency']:.1f}"
                                            for r in R["translation"][m]))
    print("\n=== REPRESENTATION I_T ===")
    for m in MODEL_NAMES:
        print(f"{PRETTY[m]:18s} " + " ".join(f"{k}={v['I_T']:.3f}" for k, v in R["representation"][m].items()))
    print("\nDone. Figures in task1/results/")


if __name__ == "__main__":
    main()
