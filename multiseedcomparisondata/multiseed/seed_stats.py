"""Multi-seed mean±SD, McNemar and paired bootstrap for three FreqNet variants:
    original_freqnet   (fixed square high-pass)
    ring_ones_freqnet  (fixed radial mask, every band weight = 1)
    dynamic_freqnet    (radial mask with per-image learned band weights)
plus two plain CNN baselines (used only if their predictions are present for a seed):
    simple_cnn
    alexnet_scratch

Usage:   python seed_stats.py [RESULTS_DIR]        (default: ./results)

RESULTS_DIR holds one folder per seed (any name containing 'seed<number>'), each with the two
CSVs written by model_comparison_new.ipynb:
    full_predictions_long.csv
    overall_metrics_comparison.csv
Folders are found recursively, so Kaggle paths like /kaggle/working/multiseed also work.
A seed without ring_ones predictions is still used for the Original-vs-Dynamic comparison.

Outputs: per_seed_metrics.csv, mean_sd_summary.csv, paired_tests_per_seed.csv,
         paired_across_seeds.csv, passband_share.csv
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_rel
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
N_BOOT = 2000
RNG = np.random.default_rng(0)

FREQNET_MODELS = ["original_freqnet", "ring_ones_freqnet", "dynamic_freqnet"]
BASELINES = ["simple_cnn", "alexnet_scratch"]
MODELS = FREQNET_MODELS + BASELINES
PAIRS = [("dynamic_freqnet", "original_freqnet"),     # does the whole method help?
         ("dynamic_freqnet", "ring_ones_freqnet"),    # does ADAPTIVE weighting help beyond the passband?
         ("ring_ones_freqnet", "original_freqnet"),   # does the wider passband alone help?
         ("dynamic_freqnet", "alexnet_scratch"),      # FreqNet variants vs the plain CNN baselines
         ("dynamic_freqnet", "simple_cnn"),
         ("original_freqnet", "alexnet_scratch"),
         ("original_freqnet", "simple_cnn"),
         ("alexnet_scratch", "simple_cnn")]
METRICS = ["roc_auc", "average_precision", "accuracy", "f1_fake", "recall_fake", "specificity_real"]
PAIR_METRICS = ["roc_auc", "average_precision", "accuracy"]


def find_seed_dirs(root):
    found = {}
    for f in root.rglob("full_predictions_long.csv"):
        m = re.search(r"seed[_\-]?(\d+)", str(f.parent.relative_to(root)) or f.parent.name, re.I) \
            or re.search(r"seed[_\-]?(\d+)", str(f.parent), re.I)
        if m:
            found[int(m.group(1))] = f.parent
    if not found:
        sys.exit(f"No 'seed<number>' folders with full_predictions_long.csv under {root.resolve()}")
    return dict(sorted(found.items()))


def load_seed(seed, folder):
    preds = pd.read_csv(folder / "full_predictions_long.csv")
    over_path = folder / "overall_metrics_comparison.csv"
    over = pd.read_csv(over_path).set_index("model") if over_path.exists() else None
    present = [m for m in MODELS if m in set(preds["model"])]
    for need in ("original_freqnet", "dynamic_freqnet"):
        if need not in present:
            sys.exit(f"seed {seed}: model '{need}' not found. Models in file: {sorted(set(preds['model']))}")
    if "ring_ones_freqnet" not in present:
        print(f"  [seed {seed}] note: no ring_ones_freqnet predictions, skipping ring comparisons for this seed")

    frames, thr = {}, {}
    for m in present:
        p = preds[preds["model"] == m][["relative_path", "y_true", "y_prob"]].copy()
        # same file name can appear under several generators: keep rows distinguishable
        p["occ"] = p.groupby("relative_path").cumcount()
        frames[m] = p.set_index(["relative_path", "occ"])
        if over is not None and m in over.index and "threshold" in over.columns:
            thr[m] = float(over.loc[m, "threshold"])
        else:
            print(f"  [seed {seed}] WARNING: no threshold for {m}, using 0.5")
            thr[m] = 0.5

    ref = frames[present[0]].index
    for m in present[1:]:
        if len(frames[m]) != len(ref) or not frames[m].index.isin(ref).all():
            sys.exit(f"seed {seed}: '{m}' was not evaluated on the same test images as '{present[0]}'")
        frames[m] = frames[m].loc[ref]
    y = frames[present[0]]["y_true"].values.astype(int)
    for m in present[1:]:
        if not (frames[m]["y_true"].values.astype(int) == y).all():
            sys.exit(f"seed {seed}: labels do not line up for '{m}'")
    probs = {m: frames[m]["y_prob"].values.astype(float) for m in present}
    print(f"seed {seed}: {len(y)} test images | " +
          " | ".join(f"{m.replace('_freqnet', '')} thr={thr[m]:.3f}" for m in present))
    return y, probs, thr


def metrics(y, p, t):
    pred = (p >= t).astype(int)
    return {"roc_auc": roc_auc_score(y, p), "average_precision": average_precision_score(y, p),
            "accuracy": float((pred == y).mean()), "f1_fake": f1_score(y, pred),
            "recall_fake": float(pred[y == 1].mean()), "specificity_real": float(1 - pred[y == 0].mean())}


def bootstrap(y, probs, thr):
    """One set of resampled images shared by all models, so every difference is paired."""
    n = len(y)
    correct = {m: ((probs[m] >= thr[m]).astype(int) == y) for m in probs}
    auc = {m: [] for m in probs}
    acc = {m: [] for m in probs}
    for _ in range(N_BOOT):
        i = RNG.integers(0, n, n)
        yy = y[i]
        if yy.min() == yy.max():
            continue
        for m in probs:
            auc[m].append(roc_auc_score(yy, probs[m][i]))
            acc[m].append(correct[m][i].mean())
    return {m: np.array(v) for m, v in auc.items()}, {m: np.array(v) for m, v in acc.items()}


def ci(v):
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


seed_dirs = find_seed_dirs(RESULTS)
print("Seeds found:", list(seed_dirs))
metric_rows, test_rows = [], []
for s, folder in seed_dirs.items():
    y, probs, thr = load_seed(s, folder)
    res = {m: metrics(y, probs[m], thr[m]) for m in probs}
    for m in probs:
        metric_rows.append({"seed": s, "model": m, "threshold": thr[m], **res[m]})
    boot_auc, boot_acc = bootstrap(y, probs, thr)
    for a, b in PAIRS:
        if a not in probs or b not in probs:
            continue
        ok_a = (probs[a] >= thr[a]).astype(int) == y
        ok_b = (probs[b] >= thr[b]).astype(int) == y
        only_a, only_b = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
        pv = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
        a_lo, a_hi = ci(boot_auc[a] - boot_auc[b])
        c_lo, c_hi = ci(boot_acc[a] - boot_acc[b])
        test_rows.append({"seed": s, "A": a, "B": b,
                          "dAUC": res[a]["roc_auc"] - res[b]["roc_auc"], "dAUC_CI_lo": a_lo, "dAUC_CI_hi": a_hi,
                          "dAcc": res[a]["accuracy"] - res[b]["accuracy"], "dAcc_CI_lo": c_lo, "dAcc_CI_hi": c_hi,
                          "only_A_right": only_a, "only_B_right": only_b, "mcnemar_p": pv})

per_seed, tests = pd.DataFrame(metric_rows), pd.DataFrame(test_rows)
per_seed.to_csv("per_seed_metrics.csv", index=False)
tests.to_csv("paired_tests_per_seed.csv", index=False)

# ---- mean ± SD over seeds -------------------------------------------------------------
summary_rows = []
for m in MODELS:
    g = per_seed[per_seed.model == m]
    if g.empty:
        continue
    row = {"model": m, "n_seeds": len(g), "seeds": ",".join(str(x) for x in g["seed"])}
    for k in METRICS:
        row[f"{k}_mean"] = g[k].mean()
        row[f"{k}_sd"] = g[k].std(ddof=1) if len(g) > 1 else np.nan
    summary_rows.append(row)
summary = pd.DataFrame(summary_rows)
summary.to_csv("mean_sd_summary.csv", index=False)

# ---- paired comparison across seeds ----------------------------------------------------
across_rows = []
for a, b in PAIRS:
    ga = per_seed[per_seed.model == a].set_index("seed")
    gb = per_seed[per_seed.model == b].set_index("seed")
    common = sorted(set(ga.index) & set(gb.index))
    if not common:
        continue
    for k in PAIR_METRICS:
        d = np.array([ga.loc[s, k] - gb.loc[s, k] for s in common])
        row = {"A": a, "B": b, "metric": k, "n_seeds": len(d), "mean_diff": d.mean(),
               "sd_diff": d.std(ddof=1) if len(d) > 1 else np.nan, "A_better_in": int((d > 0).sum())}
        if len(d) >= 3:
            row["paired_t_p"] = float(ttest_rel([ga.loc[s, k] for s in common], [gb.loc[s, k] for s in common]).pvalue)
        across_rows.append(row)
across = pd.DataFrame(across_rows)
across.to_csv("paired_across_seeds.csv", index=False)

# ---- how much of Original -> Dynamic comes from the passband alone? ---------------------
share_rows = []
for s in sorted(per_seed["seed"].unique()):
    g = per_seed[per_seed.seed == s].set_index("model")
    if all(m in g.index for m in FREQNET_MODELS):
        total = g.loc["dynamic_freqnet", "roc_auc"] - g.loc["original_freqnet", "roc_auc"]
        passband = g.loc["ring_ones_freqnet", "roc_auc"] - g.loc["original_freqnet", "roc_auc"]
        adaptive = g.loc["dynamic_freqnet", "roc_auc"] - g.loc["ring_ones_freqnet", "roc_auc"]
        share_rows.append({"seed": s, "total_gain": total, "from_passband": passband, "from_adaptive": adaptive,
                           "passband_share": passband / total if abs(total) > 1e-9 else np.nan})
share = pd.DataFrame(share_rows)
share.to_csv("passband_share.csv", index=False)

# ---- print -------------------------------------------------------------------------------
print("\n=== Per seed ===")
print(per_seed[["seed", "model", "threshold", "roc_auc", "average_precision", "accuracy", "f1_fake"]]
      .round(4).to_string(index=False))

print(f"\n=== Mean ± SD over seeds (sample SD, n-1) ===")
for _, r in summary.iterrows():
    sd = lambda k: f"{r[k + '_sd']:.4f}" if not np.isnan(r[k + "_sd"]) else "n/a"
    print(f"{r['model']:19s} (seeds {r['seeds']}) " +
          " | ".join(f"{k}: {r[k + '_mean']:.4f} ± {sd(k)}" for k in ["roc_auc", "average_precision", "accuracy", "f1_fake"]))

print("\n=== Paired comparison per seed (A minus B, same test images) ===")
show = tests.copy()
show["mcnemar_p"] = show["mcnemar_p"].map(lambda v: f"{v:.2e}")
show["A"] = show["A"].str.replace("_freqnet", "")
show["B"] = show["B"].str.replace("_freqnet", "")
print(show.round(5).to_string(index=False))

print("\n=== Paired comparison across seeds (A minus B) ===")
for _, r in across.iterrows():
    sd = f" ± {r['sd_diff']:.4f}" if not np.isnan(r["sd_diff"]) else " (1 seed)"
    p = f"  paired t-test p={r['paired_t_p']:.3f}" if "paired_t_p" in r and not np.isnan(r.get("paired_t_p", np.nan)) else ""
    print(f"{r['A'].replace('_freqnet', ''):>9s} - {r['B'].replace('_freqnet', ''):<9s} {r['metric']:<18s} "
          f"{r['mean_diff']:+.4f}{sd}   A better in {int(r['A_better_in'])}/{int(r['n_seeds'])} seeds{p}")

if not share.empty:
    print("\n=== Where does the Original -> Dynamic ROC-AUC gain come from? ===")
    print(share.round(4).to_string(index=False))
    print(f"mean passband share: {share['passband_share'].mean():.0%}   "
          f"(passband = Original -> ring_ones, adaptive = ring_ones -> Dynamic)")
print("\nSaved: per_seed_metrics.csv, mean_sd_summary.csv, paired_tests_per_seed.csv, "
      "paired_across_seeds.csv, passband_share.csv")
print("With 3 seeds a t-test has little power: report mean ± SD, 'better in k of n seeds', and the per-seed intervals.")
