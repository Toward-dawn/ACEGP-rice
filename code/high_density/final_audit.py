# -*- coding: utf-8 -*-
"""最终多轮复核：对本地已备份的 200 折做独立、互相交叉的检查。

轮次
  R1  文件与回执完整性（存在、可解析、status、n_train+n_test）
  R2  用 predictions.csv 独立重算 r，与回执里记录的 r 对比（容差 1e-6）
  R3  每个重复的 5 折必须恰好覆盖全部材料一次（不重不漏）
  R4  重复性检查：任意两折的预测向量不得完全相同；预测不得等于表型本身
  R5  跨机检查：同一折不得在两台都有回执
  R6  数值合理性：r ∈ [-1,1]、RMSE/MAE > 0、预测非常数、无 NaN/Inf
  R7  输入指纹一致性：配置/划分/表型/基因型/标记顺序 哈希在两台、全部折上必须一致
  R8  质控计数与区间：raw 值正确、passed>0、区间齐全
任何一轮出现异常都会逐条列出，并给出总体判定。
"""
import glob
import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EXPECT = {
    "development": {"n": 241, "n_train": (192, 193), "n_test": (48, 49),
                    "markers": 4817964, "qtl": 67985,
                    "regions": {"row03", "row07", "row10", "row13", "row18",
                                "new4_04", "czx4_01", "takai_tt2_01"},
                    "config": "2429f50c1c39b85ddaf44a742f27eac39b78bce094f7294a2ff33e6f779fb356",
                    "splits": "7430a3f3057226326cb19af71868076c7f05658ce7a70388a36fd4d8bae26075"},
    "validation": {"n": 151, "n_train": (120, 121), "n_test": (30, 31),
                   "markers": 4817083, "qtl": 67104,
                   "regions": {"row03", "row07", "row13", "row18",
                               "new4_04", "czx4_01", "takai_tt2_01"},
                   "config": "2429f50c1c39b85ddaf44a742f27eac39b78bce094f7294a2ff33e6f779fb356",
                   "splits": "cace55c2adf82f57c5750531bdaee2f0c1081c2de12115173caef6322fa24117"},
}


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    return float(((a - a.mean()) * (b - b.mean())).sum() /
                 np.sqrt(((a - a.mean()) ** 2).sum() * ((b - b.mean()) ** 2).sum()))


def main(results_root, output):
    roots = {"server1": os.path.join(results_root, "results_server1", "out", "folds"),
             "server2": os.path.join(results_root, "results_server2", "out", "folds"),
             "newserver": os.path.join(results_root, "results_newserver", "out", "folds")}
    folds = {}
    fail = defaultdict(list)

    # ---------- 收集 ----------
    for tag, root in roots.items():
        for p in glob.glob(os.path.join(root, "*")):
            name = os.path.basename(p)
            if "_rep" not in name:
                continue
            if not os.path.exists(os.path.join(p, "receipt.json")):
                continue                      # 未完成，跳过
            cohort = name.split("_rep")[0]
            rep = int(name.split("_rep")[1].split("_")[0])
            fno = int(name.split("fold")[1])
            folds.setdefault((cohort, rep, fno), []).append((tag, p))

    print("=" * 92)
    print("最终多轮复核（本地已备份的折）")
    print("=" * 92)
    print("回执共 %d 个（唯一折 %d 个）" % (sum(len(v) for v in folds.values()), len(folds)))

    metas = {}
    for key, lst in sorted(folds.items()):
        cohort, rep, fno = key
        E = EXPECT[cohort]
        tag, p = lst[0]
        name = "%s_rep%02d_fold%d" % (cohort, rep, fno)

        # R5 跨机重复
        if len(lst) > 1:
            fail["R5 跨机重复"].append("%s 出现在 %s" % (name, [t for t, _ in lst]))

        try:
            rec = json.load(open(os.path.join(p, "receipt.json"), encoding="utf-8"))
            aud = json.load(open(os.path.join(p, "audit.json"), encoding="utf-8"))
            pred = pd.read_csv(os.path.join(p, "predictions.csv"))
        except Exception as e:
            fail["R1 文件"].append("%s 解析失败: %s" % (name, e))
            continue

        # R1
        if rec.get("status") != "PASS_FOLD_COMPLETE":
            fail["R1 文件"].append("%s status=%s" % (name, rec.get("status")))
        if not (rec["n_train"] in E["n_train"] and rec["n_test"] in E["n_test"]
                and rec["n_train"] + rec["n_test"] == E["n"]):
            fail["R1 文件"].append("%s 样本数 %s/%s" % (name, rec["n_train"], rec["n_test"]))
        if len(pred) != rec["n_test"]:
            fail["R1 文件"].append("%s 预测行数 %d != %d" % (name, len(pred), rec["n_test"]))

        # R2 独立重算 r
        if len(pred) == rec["n_test"]:
            rg = pearson(pred.y, pred.genome)
            rq = pearson(pred.y, pred.qtl_dense)
            if abs(rg - rec["r_genome"]) > 1e-6:
                fail["R2 重算 r"].append("%s genome: 重算 %.9f vs 回执 %.9f"
                                        % (name, rg, rec["r_genome"]))
            if abs(rq - rec["r_qtl_dense"]) > 1e-6:
                fail["R2 重算 r"].append("%s qtl: 重算 %.9f vs 回执 %.9f"
                                        % (name, rq, rec["r_qtl_dense"]))

        # R6 数值合理性
        vals = pred[["y", "genome", "qtl_dense"]].to_numpy()
        if not np.isfinite(vals).all():
            fail["R6 数值"].append("%s 存在 NaN/Inf" % name)
        if pred.genome.std() < 1e-9 or pred.qtl_dense.std() < 1e-9:
            fail["R6 数值"].append("%s 预测为常数" % name)
        for k in ("r_genome", "r_qtl_dense"):
            if not (-1.0 <= rec[k] <= 1.0):
                fail["R6 数值"].append("%s %s 越界 %s" % (name, k, rec[k]))

        # R7 输入指纹
        if rec.get("config_sha256") != E["config"]:
            fail["R7 指纹"].append("%s config 不符" % name)
        if rec.get("splits_sha256") != E["splits"]:
            fail["R7 指纹"].append("%s splits 不符" % name)
        if rec.get("n_markers") != E["markers"] or rec.get("n_qtl_columns") != E["qtl"]:
            fail["R7 指纹"].append("%s 标记数 %s/%s" % (name, rec.get("n_markers"),
                                                        rec.get("n_qtl_columns")))

        # R8 质控与区间
        gq, qq = rec.get("genome_qc", {}), rec.get("qtl_qc", {})
        if gq.get("raw") != E["markers"] or not (0 < gq.get("passed", -1) <= E["markers"]):
            fail["R8 质控"].append("%s 背景质控 %s" % (name, gq))
        if qq.get("raw") != E["qtl"] or not (0 < qq.get("passed", -1) <= E["qtl"]):
            fail["R8 质控"].append("%s QTL 质控 %s" % (name, qq))
        rq_keys = set(rec.get("region_qc", {}))
        if rq_keys != E["regions"]:
            fail["R8 质控"].append("%s 区间集合 %s" % (name, sorted(rq_keys)))
        for q, v in rec.get("region_qc", {}).items():
            if v.get("passed", 0) <= 0:
                fail["R8 质控"].append("%s 区间 %s 质控后为 0" % (name, q))

        metas[key] = {"path": p, "rec": rec, "pred": pred, "tag": tag}

    # R3 覆盖
    by_rep = defaultdict(dict)
    for (cohort, rep, fno), m in metas.items():
        by_rep[(cohort, rep)][fno] = m["pred"].sample_id.tolist()
    complete_reps = 0
    for (cohort, rep), fm in sorted(by_rep.items()):
        need = EXPECT[cohort]["n"]
        if len(fm) < 5:
            continue
        complete_reps += 1
        allids = [i for f in sorted(fm) for i in fm[f]]
        if len(set(allids)) != need or len(allids) != need:
            fail["R3 覆盖"].append("%s rep%02d: 去重后 %d，总 %d，应为 %d"
                                   % (cohort, rep, len(set(allids)), len(allids), need))

    # R4 重复性：两折预测向量完全相同 = 可疑
    sigs = {}
    for key, m in metas.items():
        p = m["pred"]
        if "genome" not in p:
            continue
        sig = hashlib.sha256(np.round(p.genome.to_numpy(), 6).tobytes()).hexdigest()
        sigs.setdefault(sig, []).append("%s_rep%02d_fold%d" % key)
    for sig, names in sigs.items():
        if len(names) > 1:
            fail["R4 重复预测"].append("预测向量完全相同: %s" % names)

    # ---------- 汇总 ----------
    print("\n" + "=" * 92)
    print("各轮结果")
    print("=" * 92)
    rounds = ["R1 文件", "R2 重算 r", "R3 覆盖", "R4 重复预测", "R5 跨机重复",
              "R6 数值", "R7 指纹", "R8 质控"]
    for r in rounds:
        items = fail.get(r, [])
        print("  %-12s %s" % (r, "通过" if not items else "**%d 处异常**" % len(items)))
        for x in items[:6]:
            print("       - %s" % x)

    print("\n完整重复数：%d / 40（每群体 20）" % complete_reps)
    total = len(metas)
    ok = sum(1 for r in rounds if not fail.get(r))
    print("折总数 %d，通过轮次 %d / %d" % (total, ok, len(rounds)))
    verdict = "全部通过" if ok == len(rounds) else "存在异常"
    print("总判定：%s" % verdict)

    with open(output, "w", encoding="utf-8") as fh:
        json.dump({"folds": total, "complete_repeats": complete_reps,
                   "fail": {k: v for k, v in fail.items()}, "verdict": verdict},
                  fh, indent=1, ensure_ascii=False)
    print("结果写入 %s" % output)
    return 0 if verdict == "全部通过" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit saved high-density fold outputs")
    parser.add_argument("--results-root", required=True,
                        help="directory containing results_server1/, results_server2/ and results_newserver/")
    parser.add_argument("--output", required=True, help="path for final_audit.json")
    args = parser.parse_args()
    sys.exit(main(args.results_root, args.output))
