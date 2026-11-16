# -*- coding: utf-8 -*-
"""材料层面配对 bootstrap（条件性补充分析，不重新训练）。

设计
  1. 以材料（sample）为单位，有放回抽取 n 个（Primary 241 / Validation 151）。
  2. 同一次抽样的索引同时用于两个模型、全部 20 次重复。
  3. 每次抽样内：对 20 个重复分别计算 r(QTL-informed) − r(Genome-wide)，
     再取这 20 个差值的平均，作为该次抽样的统计量 Δr*。
  4. 固定随机种子，重采样 B = 10,000 次。
  5. 报告：原始平均 Δr、bootstrap 95% 分位数区间、Δr* > 0 的比例、
     种子、有效抽样次数、异常情况。

输入：既有 200 折的 predictions.csv（逐材料折外预测），不做任何重新训练。
输出：bootstrap_results.json / bootstrap_distribution.csv / bootstrap_核查记录.md
"""
import glob
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

N = {"development": 241, "validation": 151}
CN = {"development": "Primary", "validation": "Validation"}

SEED = 20261112
B = 10000
NREP = 20


def build_matrix(cohort, roots):
    """返回 (sample_ids, y, Pg, Pq)，形状 (n,), (n,), (n,20), (n,20)。"""
    n = N[cohort]
    store = {}
    for rep in range(1, NREP + 1):
        parts = []
        for fno in range(1, 6):
            name = "%s_rep%02d_fold%d" % (cohort, rep, fno)
            p = None
            for r in roots:
                q = os.path.join(r, name, "predictions.csv")
                if os.path.exists(q):
                    p = q
                    break
            if p is None:
                raise RuntimeError("缺少折: %s" % name)
            d = pd.read_csv(p)
            d["repeat"] = rep
            parts.append(d)
        d = pd.concat(parts, ignore_index=True)
        if len(d) != n or d.sample_id.nunique() != n:
            raise RuntimeError("%s rep%02d 材料数异常: %d / %d" % (cohort, rep, len(d), d.sample_id.nunique()))
        for t in d.itertuples():
            rec = store.setdefault(t.sample_id, {})
            rec[rep] = (float(t.y), float(t.genome), float(t.qtl_dense))

    ids = sorted(store)
    y = np.array([store[s][1][0] for s in ids], float)
    Pg = np.zeros((n, NREP))
    Pq = np.zeros((n, NREP))
    bad = 0
    for i, s in enumerate(ids):
        rec = store[s]
        if len(rec) != NREP:
            raise RuntimeError("%s 只有 %d 个重复的预测" % (s, len(rec)))
        ys = {round(rec[r][0], 9) for r in rec}
        if len(ys) != 1:
            bad += 1
        for r in range(1, NREP + 1):
            Pg[i, r - 1] = rec[r][1]
            Pq[i, r - 1] = rec[r][2]
    return ids, y, Pg, Pq, bad


def rowwise_corr(Yc, Yss, P):
    """Yc: (B,n) 已中心化；Yss: (B,) 平方和；P: (B,n) 原始值。"""
    Pc = P - P.mean(axis=1, keepdims=True)
    Pss = (Pc ** 2).sum(axis=1)
    num = (Yc * Pc).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num / np.sqrt(Yss * Pss)
    return r


def main(results_root, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    roots = [os.path.join(results_root, "results_server1", "out", "folds"),
             os.path.join(results_root, "results_server2", "out", "folds"),
             os.path.join(results_root, "results_newserver", "out", "folds")]
    t0 = time.time()
    results = {}
    dist_rows = []

    for cohort in ("development", "validation"):
        ids, y, Pg, Pq, bad_y = build_matrix(cohort, roots)
        n = len(ids)
        print("=" * 84)
        print("%s：%d 份材料 × %d 个重复" % (CN[cohort], n, NREP))
        if bad_y:
            print("  警告：%d 个材料的 y 跨重复不一致" % bad_y)

        # 原始值（同一套函数，恒等索引）
        idx_id = np.arange(n)[None, :]
        Y0 = y[idx_id]
        Y0c = Y0 - Y0.mean(axis=1, keepdims=True)
        Y0ss = (Y0c ** 2).sum(axis=1)
        rg0 = np.array([rowwise_corr(Y0c, Y0ss, Pg[:, j][idx_id])[0] for j in range(NREP)])
        rq0 = np.array([rowwise_corr(Y0c, Y0ss, Pq[:, j][idx_id])[0] for j in range(NREP)])
        d0 = rq0 - rg0
        orig_mean = float(d0.mean())
        print("  原始：r(genome) %.10f  r(qtl) %.10f  Δr 均值 %+.10f  (SD %.10f)"
              % (rg0.mean(), rq0.mean(), orig_mean, d0.std(ddof=1)))
        print("  逐重复 Δr 范围 %+.10f ~ %+.10f，正向 %d / %d"
              % (d0.min(), d0.max(), int((d0 > 0).sum()), NREP))

        # bootstrap
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, n, size=(B, n), dtype=np.int32)
        Y = y[idx]
        Yc = Y - Y.mean(axis=1, keepdims=True)
        Yss = (Yc ** 2).sum(axis=1)

        diffs = np.empty((B, NREP))
        nan_flag = np.zeros(B, dtype=bool)
        for j in range(NREP):
            rg = rowwise_corr(Yc, Yss, Pg[:, j][idx])
            rq = rowwise_corr(Yc, Yss, Pq[:, j][idx])
            diffs[:, j] = rq - rg
            nan_flag |= ~np.isfinite(rg) | ~np.isfinite(rq)

        stat = np.nanmean(np.where(nan_flag[:, None], np.nan, diffs), axis=1)
        finite = np.isfinite(stat)
        eff = int(finite.sum())
        s = stat[finite]

        lo, hi = np.percentile(s, [2.5, 97.5])
        prop_pos = float((s > 0).mean())
        print("  bootstrap（种子 %d，B=%d）：有效 %d 次，异常 %d 次" % (SEED, B, eff, B - eff))
        print("    均值 %+.10f  中位数 %+.10f" % (s.mean(), np.median(s)))
        print("    95%% 分位数区间 [%+.10f, %+.10f]" % (lo, hi))
        print("    Δr* > 0 的比例 %.6f" % prop_pos)

        results[CN[cohort]] = {
            "n_materials": n,
            "n_repeats": NREP,
            "seed": SEED,
            "n_resamples_requested": B,
            "n_resamples_effective": eff,
            "n_resamples_dropped": int(B - eff),
            "original_mean_r_genome": float(rg0.mean()),
            "original_mean_r_qtl": float(rq0.mean()),
            "original_mean_delta": orig_mean,
            "original_sd_delta": float(d0.std(ddof=1)),
            "original_per_repeat_delta": [float(x) for x in d0],
            "original_n_positive": int((d0 > 0).sum()),
            "bootstrap_mean": float(s.mean()),
            "bootstrap_median": float(np.median(s)),
            "ci95_percentile": [float(lo), float(hi)],
            "proportion_delta_gt_zero": prop_pos,
            "percentile_method": "linear interpolation (numpy.percentile 默认)",
            "materials_with_inconsistent_y": int(bad_y),
        }
        for v in s:
            dist_rows.append({"cohort": CN[cohort], "delta_star": float(v)})

    pd.DataFrame(dist_rows).to_csv(os.path.join(output_dir, "bootstrap_distribution.csv"),
                                   index=False, float_format="%.12f")

    meta = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_seconds": round(time.time() - t0, 1),
        "design": {
            "resampling_unit": "材料（sample），有放回",
            "shared_index": "同一抽样索引同时用于两个模型与全部 20 次重复",
            "statistic": "每次抽样内先算 20 个重复的 r_qtl−r_genome，再取平均",
            "n_resamples": B,
            "seed": SEED,
            "rng": "numpy.random.default_rng(seed).integers",
            "input": "既有 200 折 predictions.csv（逐材料折外预测）",
            "no_retraining": True,
        },
        "results": results,
    }
    with open(os.path.join(output_dir, "bootstrap_results.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, ensure_ascii=False)

    # 核查记录
    L = []
    A = L.append
    A("# 材料层面配对 bootstrap · 运行记录")
    A("")
    A("生成时间：%s　｜　耗时 %.1f 秒" % (meta["generated_at"], meta["runtime_seconds"]))
    A("")
    A("## 一、方法")
    A("")
    A("| 项目 | 设置 |")
    A("|---|---|")
    A("| 重采样单位 | 材料（sample），有放回抽取 n 个 |")
    A("| 抽样索引 | **同一次抽样的索引同时用于两个模型与全部 20 次重复** |")
    A("| 每次抽样的统计量 | 先在 20 个重复内分别计算 r(QTL-informed) − r(Genome-wide)，再取 20 个差值的平均 |")
    A("| 重采样次数 | %d |" % B)
    A("| 随机种子 | %d（numpy default_rng） |" % SEED)
    A("| 分位数方法 | numpy.percentile 线性插值 |")
    A("| 输入 | 既有 200 折的 predictions.csv（逐材料折外预测） |")
    A("| 是否重新训练 | **否** |")
    A("")
    A("## 二、结果")
    A("")
    A("| 项目 | Primary | Validation |")
    A("|---|---|---|")
    for k, label in (("n_materials", "材料数"),):
        A("| %s | %d | %d |" % (label, results["Primary"][k], results["Validation"][k]))
    A("| 重复数 | 20 | 20 |")
    A("| 原始平均 r（Genome-wide） | %.10f | %.10f |"
      % (results["Primary"]["original_mean_r_genome"], results["Validation"]["original_mean_r_genome"]))
    A("| 原始平均 r（QTL-informed） | %.10f | %.10f |"
      % (results["Primary"]["original_mean_r_qtl"], results["Validation"]["original_mean_r_qtl"]))
    A("| **原始平均 Δr** | **%+.10f** | **%+.10f** |"
      % (results["Primary"]["original_mean_delta"], results["Validation"]["original_mean_delta"]))
    A("| bootstrap 均值 | %+.10f | %+.10f |"
      % (results["Primary"]["bootstrap_mean"], results["Validation"]["bootstrap_mean"]))
    A("| **95%% 分位数区间** | **[%+.10f, %+.10f]** | **[%+.10f, %+.10f]** |"
      % (results["Primary"]["ci95_percentile"][0], results["Primary"]["ci95_percentile"][1],
         results["Validation"]["ci95_percentile"][0], results["Validation"]["ci95_percentile"][1]))
    A("| **Δr\\* > 0 的比例** | **%.4f** | **%.4f** |"
      % (results["Primary"]["proportion_delta_gt_zero"], results["Validation"]["proportion_delta_gt_zero"]))
    A("| 有效抽样次数 | %d / %d | %d / %d |"
      % (results["Primary"]["n_resamples_effective"], B,
         results["Validation"]["n_resamples_effective"], B))
    A("| 异常（剔除）抽样数 | %d | %d |"
      % (results["Primary"]["n_resamples_dropped"], results["Validation"]["n_resamples_dropped"]))
    A("| 原始逐重复 Δr 正向数 | %d / 20 | %d / 20 |"
      % (results["Primary"]["original_n_positive"], results["Validation"]["original_n_positive"]))
    A("")
    A("## 三、异常与边界情况")
    A("")
    A("- 异常定义：某次抽样中，20 个重复里任一相关系数出现非有限值（例如抽样后 y 为常数导致分母为 0），"
      "则该次抽样整体记为异常并从统计中剔除，不参与分位数与比例计算。")
    A("- 本次结果：Primary 剔除 %d 次、Validation 剔除 %d 次。"
      % (results["Primary"]["n_resamples_dropped"], results["Validation"]["n_resamples_dropped"]))
    A("- 材料 y 跨重复一致性异常数：Primary %d、Validation %d。"
      % (results["Primary"]["materials_with_inconsistent_y"], results["Validation"]["materials_with_inconsistent_y"]))
    A("")
    A("## 四、定位与限制")
    A("")
    A("1. 本分析是**基于固定折外预测的条件性补充分析**，不改变、也不替代原报告的统计结果。")
    A("2. 它只反映「把材料视为来自更大育种群体时，Δr 的抽样不确定性」，不能据此宣称在其它群体或新环境中显著。")
    A("3. 无法消除 Primary 面板的选择偏差：QTL 面板由同一批材料定位得到，重采样不会让面板变外部。")
    A("4. 简单有放回重采样忽略材料间的亲缘/亚群结构。")
    A("5. 折外预测来自每折训练 4/5 材料的模型；区间反映的是该训练规模下的表现。")

    with open(os.path.join(output_dir, "bootstrap_核查记录.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L))

    print()
    print("已写 bootstrap_results.json / bootstrap_distribution.csv / bootstrap_核查记录.md")
    print("总耗时 %.1f 秒" % (time.time() - t0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap fixed out-of-fold predictions without retraining")
    parser.add_argument("--results-root", required=True,
                        help="directory containing results_server1/, results_server2/ and results_newserver/")
    parser.add_argument("--output-dir", required=True, help="directory for bootstrap outputs")
    args = parser.parse_args()
    main(args.results_root, args.output_dir)
