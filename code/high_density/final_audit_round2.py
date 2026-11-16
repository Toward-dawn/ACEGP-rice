# -*- coding: utf-8 -*-
"""第二批独立复核（R9–R12）：用官方原始文件反向验证每折预测，不依赖回执内容。

与 final_audit.py（R1–R8，主要核对回执自洽与内部一致性）互补：
本脚本把 predictions.csv 与**官方划分文件、官方表型文件**直接对照。

R9  每折 predictions 的 sample_id 必须等于 20x5.json 中该 repeat/fold 的 test_ids
R10 predictions 里的 y 必须等于官方表型文件中对应 sample 的值
R11 预测值落在合理区间（不超出 y 的范围 ±5 倍跨度）
R12 每个重复的 5 折合并后：n 等于材料总数、材料不重复、合并 r 落在各折 r 的 [min,max] 附近
"""
import glob
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    return float(((a - a.mean()) * (b - b.mean())).sum() /
                 np.sqrt(((a - a.mean()) ** 2).sum() * ((b - b.mean()) ** 2).sum()))


def main(results_root, inputs_root, output):
    roots = [os.path.join(results_root, "results_server1", "out", "folds"),
             os.path.join(results_root, "results_server2", "out", "folds"),
             os.path.join(results_root, "results_newserver", "out", "folds")]
    splits = {}
    pheno = {}
    for cohort in ("development", "validation"):
        splits[cohort] = json.load(open(os.path.join(inputs_root, "inputs", "%s_20x5.json" % cohort),
                                        encoding="utf-8"))
        df = pd.read_csv(os.path.join(inputs_root, "inputs", "%s_samples.csv" % cohort),
                         dtype={"sample_id": str})
        pheno[cohort] = dict(zip(df.sample_id, df.loss_SF.astype(float)))

    fail = {k: [] for k in ("R9 划分", "R10 表型", "R11 区间", "R12 合并")}
    folds = {}
    for root in roots:
        for p in glob.glob(os.path.join(root, "*")):
            name = os.path.basename(p)
            if "_rep" not in name or not os.path.exists(os.path.join(p, "predictions.csv")):
                continue
            cohort = name.split("_rep")[0]
            rep = int(name.split("_rep")[1].split("_")[0])
            fno = int(name.split("fold")[1])
            folds.setdefault((cohort, rep, fno), p)

    print("=" * 90)
    print("第二批独立复核（对照官方划分文件与表型文件）")
    print("=" * 90)
    print("折数 %d" % len(folds))

    for (cohort, rep, fno), p in sorted(folds.items()):
        name = "%s_rep%02d_fold%d" % (cohort, rep, fno)
        pred = pd.read_csv(os.path.join(p, "predictions.csv"))
        # R9
        sp = splits[cohort]["repeats"][rep - 1]["folds"][fno - 1]
        if pred.sample_id.tolist() != sp["test_ids"]:
            fail["R9 划分"].append("%s 测试样本与官方划分不一致" % name)
        # R10
        ph = pheno[cohort]
        bad = 0
        for sid, y in zip(pred.sample_id, pred.y):
            if sid not in ph or abs(ph[sid] - float(y)) > 1e-6:
                bad += 1
        if bad:
            fail["R10 表型"].append("%s 有 %d 个材料的 y 与官方表型不符" % (name, bad))
        # R11
        ys = np.array([ph[s] for s in pred.sample_id if s in ph])
        if ys.size:
            lo, hi = ys.min() - 5 * (ys.max() - ys.min()), ys.max() + 5 * (ys.max() - ys.min())
            for col in ("genome", "qtl_dense"):
                v = pred[col].to_numpy()
                if v.min() < lo or v.max() > hi:
                    fail["R11 区间"].append("%s %s 预测超出合理区间 [%.1f, %.1f]"
                                            % (name, col, lo, hi))

    # R12 合并
    by_rep = {}
    for (cohort, rep, fno), p in folds.items():
        by_rep.setdefault((cohort, rep), {})[fno] = pd.read_csv(os.path.join(p, "predictions.csv"))
    complete = 0
    for (cohort, rep), fm in sorted(by_rep.items()):
        if len(fm) < 5:
            continue
        complete += 1
        d = pd.concat([fm[f] for f in sorted(fm)], ignore_index=True)
        need = 241 if cohort == "development" else 151
        if len(d) != need or d.sample_id.nunique() != need:
            fail["R12 合并"].append("%s rep%02d 合并后 %d 行 / %d 唯一，应为 %d"
                                    % (cohort, rep, len(d), d.sample_id.nunique(), need))
            continue
        pooled = pearson(d.y, d.qtl_dense)
        per = [pearson(f.y, f.qtl_dense) for f in fm.values()]
        lo, hi = min(per), max(per)
        pad = 0.15 * (hi - lo) + 0.01
        if not (lo - pad <= pooled <= hi + pad):
            fail["R12 合并"].append("%s rep%02d 合并 r %.4f 不在各折 [%.4f, %.4f] 附近"
                                    % (cohort, rep, pooled, lo, hi))

    print("\n各轮结果")
    ok = 0
    for k in ("R9 划分", "R10 表型", "R11 区间", "R12 合并"):
        items = fail[k]
        print("  %-10s %s" % (k, "通过" if not items else "**%d 处异常**" % len(items)))
        for x in items[:6]:
            print("       - %s" % x)
        if not items:
            ok += 1
    print("\n完整重复 %d / 40" % complete)
    print("通过轮次 %d / 4" % ok)
    verdict = "全部通过" if ok == 4 else "存在异常"
    print("总判定：%s" % verdict)

    with open(output, "w", encoding="utf-8") as fh:
        json.dump({"folds": len(folds), "complete_repeats": complete,
                   "fail": fail, "verdict": verdict}, fh, indent=1, ensure_ascii=False)
    return 0 if ok == 4 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit saved predictions against fixed splits and phenotype files")
    parser.add_argument("--results-root", required=True,
                        help="directory containing results_server1/, results_server2/ and results_newserver/")
    parser.add_argument("--inputs-root", required=True,
                        help="prediction-input directory containing inputs/")
    parser.add_argument("--output", required=True, help="path for final_audit_round2.json")
    args = parser.parse_args()
    sys.exit(main(args.results_root, args.inputs_root, args.output))
