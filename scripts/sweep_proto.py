"""Prototype: depth-window sweep for flat ink models.

For each z offset of the model's layer window, run the model on sampled tiles
and record label-based (AUC vs ink labels) and label-free scores.

usage: python sweep_proto.py VOLUME_ZARR CHECKPOINT OUT_JSON
          [--labels LABEL_ZARR --supmask MASK_ZARR] [--max-tiles N] [--drop-front K]
--drop-front K simulates a surface rendered K layers too deep by discarding the
first K source layers (known-answer test: best offset should move by -K).
"""
import argparse
import json
import logging
import time

import numpy as np
import torch
import zarr

from vesuvius.ink_detection.inference.infer import (
    configure_single_model,
    logits_to_probabilities,
    normalize_flat_patch,
    select_layer_indices,
)

logging.basicConfig(level=logging.WARNING)


def auc(scores, labels):
    """Rank-based ROC AUC; labels boolean."""
    labels = labels.astype(bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ties
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def label_free_scores(prob, onsheet):
    p = prob[onsheet]
    if p.size == 0:
        return {}
    p50, p90, p99 = np.percentile(p, [50, 90, 99])
    return {
        "mean": float(p.mean()),
        "p99_minus_p50": float(p99 - p50),
        "p90_minus_p50": float(p90 - p50),
        "std": float(p.std()),
        "frac_gt_0.5": float((p > 0.5).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("volume")
    ap.add_argument("checkpoint")
    ap.add_argument("out_json")
    ap.add_argument("--labels")
    ap.add_argument("--supmask")
    ap.add_argument("--max-tiles", type=int, default=256)
    ap.add_argument("--drop-front", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cm = configure_single_model(args.checkpoint, amp_dtype="auto")
    device = torch.device("cuda")
    model = cm.model.to(device).eval()
    P, Dm = cm.patch_size, cm.input_depth

    vol = zarr.open_array(args.volume + "/0", mode="r")
    D, H, W = vol.shape
    D_eff = D
    base = select_layer_indices(D_eff, layer_start=None, layer_end=None,
                                output_depth=Dm, direction="forward")
    offsets = [o for o in range(-D_eff, D_eff + 1)
               if base[0] + o >= 0 and base[-1] + o < D_eff]
    print(f"vol {vol.shape} model depth {Dm} patch {P} base {base[0]}..{base[-1]} "
          f"drop_front {args.drop_front} offsets {offsets[0]}..{offsets[-1]}", flush=True)

    lab = sup = None
    if args.labels:
        lab = zarr.open_array(args.labels + "/0", mode="r")
        sup = zarr.open_array(args.supmask + "/0", mode="r")

    # candidate tiles: non-overlapping PxP grid
    rng = np.random.default_rng(args.seed)
    cands = [(y, x) for y in range(0, H - P + 1, P) for x in range(0, W - P + 1, P)]
    rng.shuffle(cands)
    tiles = []
    for y, x in cands:
        if len(tiles) >= args.max_tiles:
            break
        raw = np.asarray(vol[D // 2, y:y + P, x:x + P])
        if (raw > 0).mean() < 0.9:
            continue
        if sup is not None:
            s = np.asarray(sup[:, y:y + P, x:x + P]).max(axis=0) > 0
            if s.mean() < 0.5:
                continue
            l = np.asarray(lab[:, y:y + P, x:x + P]).max(axis=0) > 0
            if l[s].mean() < 0.02 or l[s].mean() > 0.98:
                continue
        tiles.append((y, x))
    print(f"{len(tiles)} tiles", flush=True)

    # preload raw stacks (all layers) once
    # --drop-front K: new[z] = old[z + K], zero-padded at the back so depth and
    # the centred base window stay the same; the best offset should move by -K.
    K = args.drop_front
    raws = []
    for y, x in tiles:
        r = np.asarray(vol[:, y:y + P, x:x + P])
        if K:
            r = np.concatenate([r[K:], np.zeros_like(r[:K])], axis=0)
        raws.append(r)
    labs = sups = None
    if lab is not None:
        labs = [np.asarray(lab[:, y:y + P, x:x + P]).max(axis=0) > 0 for y, x in tiles]
        sups = [np.asarray(sup[:, y:y + P, x:x + P]).max(axis=0) > 0 for y, x in tiles]

    autocast = torch.autocast("cuda", dtype=cm.amp_dtype) if cm.amp_dtype else torch.autocast("cuda", enabled=False)
    results = []
    per_tile = {}
    t0 = time.time()
    for o in offsets:
        idx = base + o
        probs = []
        for b in range(0, len(tiles), args.batch_size):
            batch = []
            for r in raws[b:b + args.batch_size]:
                zyx = r[idx]  # (Dm, P, P)
                batch.append(normalize_flat_patch(zyx, cm.preprocessing))
            x = torch.from_numpy(np.stack(batch)).unsqueeze(1).to(device)
            with torch.inference_mode(), autocast:
                pr = logits_to_probabilities(model(x), image_hw=(P, P))
            probs.extend(pr.float().cpu().numpy()[:, 0])
        rec = {"offset": o}
        onsheet = [r[r.shape[0] // 2] > 0 for r in raws]
        allp = np.concatenate([p[m] for p, m in zip(probs, onsheet)])
        rec.update({f"lf_{k}": v for k, v in label_free_scores(allp, np.ones_like(allp, bool)).items()})
        if labs is not None:
            sc = np.concatenate([p[s] for p, s in zip(probs, sups)])
            lb = np.concatenate([l[s] for l, s in zip(labs, sups)])
            rec["auc"] = auc(sc, lb)
        results.append(rec)
        per_tile[o] = [label_free_scores(p, m).get("p99_minus_p50", float("nan")) for p, m in zip(probs, onsheet)]
        if labs is not None:
            per_tile.setdefault(f"auc{o}", [auc(p[s], l[s]) for p, l, s in zip(probs, labs, sups)])
        print(json.dumps(rec), f"{time.time()-t0:.0f}s", flush=True)

    out = {"volume": args.volume, "checkpoint": args.checkpoint, "drop_front": args.drop_front,
           "tiles": tiles, "offsets": offsets, "results": results,
           "per_tile": {str(k): v for k, v in per_tile.items()}}
    json.dump(out, open(args.out_json, "w"), indent=1)
    for key in [k for k in results[0] if k != "offset"]:
        best = max(results, key=lambda r: r[key] if r[key] == r[key] else -1)
        print(f"argmax {key:>18s}: offset {best['offset']:+d}")


if __name__ == "__main__":
    main()
