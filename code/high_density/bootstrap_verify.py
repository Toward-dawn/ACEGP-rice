# -*- coding: utf-8 -*-
"""bootstrap 的稳健性与一致性验证。

1. 用多个随机种子重跑，检查 95% 区间是否稳定（不改变主结果，仅作验证）。
2. 核对原始平均 Δr 与 review_per_repeat.csv 是否一致。
3. 核对分布文件的完整性与数值范围。
"""
import json
import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

N = {"development": 241, "validation": 151}
CN = {"development": "Primary", "validation": "Validation"}
NREP = 20
def build(cohort, roots):
    n = N[cohort]
    store = {}
    for rep in range(1, NREP + 1):
        parts = []
        for fno in range(1, 6):
            name = "%s_rep%02d_fold%d" % (cohort, rep, fno)
            p = next((os.path.join(r, name, "predictions.csv") for r in roots
                      if os.path.exists(os.path.join(r, name, "predictions.csv"))), None)
            parts.append(pd.read_csv(p))
        d = pd.concat(parts, ignore_index=True)
        for t in d.itertuples():
            store.setdefault(t.sample_id, {})[rep] = (float(t.y), float(t.genome), float(t.qtl_dense))
    ids = sorted(store)
    y = np.array([store[s][1][0] for s in ids])
    Pg = np.zeros((n, NREP)); Pq = np.zeros((n, NREP))
    for i, s in enumerate(ids):
        for r in range(1, NREP + 1):
            Pg[i, r - 1] = store[s][r][1]
            Pq[i, r - 1] = store[s][r][2]
    return y, Pg, Pq


def boot(y, Pg, Pq, seed, B=10000):
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n), dtype=np.int32)
    Y = y[idx]
    Yc = Y - Y.mean(1, keepdims=True)
    Yss = (Yc ** 2).sum(1)

    def rc(P):
        Pc = P - P.mean(1, keepdims=True)
        return (Yc * Pc).sum(1) / np.sqrt(Yss * (Pc ** 2).sum(1))

    diffs = np.empty((B, NREP))
    for j in range(NREP):
        diffs[:, j] = rc(Pq[:, j][idx]) - rc(Pg[:, j][idx])
    s = diffs.mean(1)
    return s[np.isfinite(s)]


parser = argparse.ArgumentParser(description="Verify fixed-prediction bootstrap outputs")
parser.add_argument("--results-root", required=True,
                    help="directory containing review_per_repeat.csv, bootstrap_results.json and fold outputs")
parser.add_argument("--output-dir", required=True, help="directory for the multi-seed audit JSON")
args = parser.parse_args()
D = args.results_root
os.makedirs(args.output_dir, exist_ok=True)
ROOTS = [os.path.join(D, "results_server1", "out", "folds"),
         os.path.join(D, "results_server2", "out", "folds"),
         os.path.join(D, "results_newserver", "out", "folds")]

main = json.load(open(os.path.join(D, "bootstrap_results.json"), encoding="utf-8"))
per = pd.read_csv(os.path.join(D, "review_per_repeat.csv"))
dist = pd.read_csv(os.path.join(D, "bootstrap_distribution.csv"))

print("=" * 88)
print("验证 1：原始平均 Δr 与 review_per_repeat.csv 的一致性")
for coh in ("development", "validation"):
    src = per[per.cohort == coh].delta.mean()
    got = main["results"][CN[coh]]["original_mean_delta"]
    print("  %-11s 报告 %+.12f  review CSV %+.12f  差 %.2e"
          % (CN[coh], got, src, abs(got - src)))

print()
print("验证 2：分布文件完整性")
for coh in ("Primary", "Validation"):
    s = dist[dist.cohort == coh].delta_star
    r = main["results"][coh]
    print("  %-11s 行数 %d（应为 10000），均值 %+.10f，min %+.6f，max %+.6f"
          % (coh, len(s), s.mean(), s.min(), s.max()))
    print("             重算 2.5%%/97.5%% = [%+.10f, %+.10f]；报告值 [%+.10f, %+.10f]"
          % (np.percentile(s, 2.5), np.percentile(s, 97.5),
             r["ci95_percentile"][0], r["ci95_percentile"][1]))

print()
print("验证 3：多种子稳健性（不改变主结果，仅检查区间稳定性）")
seeds = [20261112, 1, 2, 3, 42]
out = {}
for coh in ("development", "validation"):
    y, Pg, Pq = build(coh, ROOTS)
    rows = []
    for sd in seeds:
        s = boot(y, Pg, Pq, sd)
        lo, hi = np.percentile(s, [2.5, 97.5])
        rows.append((sd, float(s.mean()), float(lo), float(hi), float((s > 0).mean())))
    out[CN[coh]] = [{"seed": r[0], "mean": r[1], "ci95": [r[2], r[3]],
                     "prop_gt0": r[4]} for r in rows]
    print("  %s" % CN[coh])
    for r in rows:
        print("    种子 %-8d 均值 %+.10f  区间 [%+.10f, %+.10f]  P(>0)=%.4f"
              % (r[0], r[1], r[2], r[3], r[4]))
    lo_all = [r[2] for r in rows]; hi_all = [r[3] for r in rows]
    print("    区间下界跨度 %.2e，上界跨度 %.2e" % (max(lo_all) - min(lo_all), max(hi_all) - min(hi_all)))

with open(os.path.join(args.output_dir, "bootstrap_种子稳健性.json"), "w", encoding="utf-8") as fh:
    json.dump({"seeds": seeds, "design_note": "仅验证区间稳定性，主结果仍以种子 20261112 为准",
               "results": out}, fh, indent=1, ensure_ascii=False)
print()
print("已写 bootstrap_种子稳健性.json")
