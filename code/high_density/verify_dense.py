# -*- coding: utf-8 -*-
"""高密度数据核验：把 PLINK bed / validation npy 与现有低密度输入逐位点比对。

检查
  1. 样本数（241 / 151）与样本顺序
  2. 标记数（bim 4,817,964；validation 4,817,083 = 全局减去 qRSF9.2 的 881）
  3. bed 解码方向：与 development_background.raw 在共享位点上的基因型是否一致
  4. validation npy 与 validation background.npy 在共享位点上是否一致
  5. validation 高密度是否已排除 qRSF9.2
  6. 各 QTL 区间在高密度背景中的位点数
"""
import json
import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OFFSETS = {1: 0, 2: 43270923, 3: 79208173, 4: 115621992, 5: 151124686,
           6: 181083120, 7: 212331907, 8: 242029528, 9: 270472550,
           10: 293485270, 11: 316692557, 12: 345713663}


def unpack_bed(path, n_snp, n_ind):
    with open(path, "rb") as fh:
        magic = fh.read(3)
        assert magic[:2] == b"\x6c\x1b", "不是 PLINK bed"
        raw = np.fromfile(fh, dtype=np.uint8)
    per = (n_ind + 3) // 4
    assert raw.size == n_snp * per, "bed 大小与形状不符: %d vs %d" % (raw.size, n_snp * per)
    b = raw.reshape(n_snp, per)
    out = np.empty((n_snp, n_ind), dtype=np.uint8)
    for s in range(4):
        code = (b >> (2 * s)) & 3
        take = len(range(s, n_ind, 4))
        out[:, s::4] = code[:, :take]
    return out


def main(workspace, output):
    hd = os.path.join(workspace, "unpacked")
    dense = os.path.join(hd, "dense")
    data = os.path.join(hd, "data_prediction_inputs")
    out = output
    os.makedirs(out, exist_ok=True)
    rep = {"checks": []}

    def chk(name, ok, detail=""):
        rep["checks"].append({"check": name, "pass": bool(ok), "detail": detail})
        print("  [%s] %-46s %s" % ("PASS" if ok else "FAIL", name, detail))

    print("=" * 90)
    print("高密度数据核验")
    print("=" * 90)

    bim = pd.read_csv(os.path.join(dense, "PMB2026_full_241.bim"), sep="\t", header=None,
                      names=["chr", "snp", "cm", "pos", "a1", "a2"],
                      dtype={"chr": int, "snp": str, "pos": int, "a1": str, "a2": str})
    fam = pd.read_csv(os.path.join(dense, "PMB2026_full_241.fam"), sep=r"\s+", header=None,
                      names=["fid", "iid", "pat", "mat", "sex", "pheno"], dtype={"iid": str})
    print("\n  primary dense: %d 标记 × %d 样本" % (len(bim), len(fam)))
    chk("primary 样本数=241", len(fam) == 241, str(len(fam)))
    chk("primary 标记数=4,817,964", len(bim) == 4817964, str(len(bim)))

    dev = pd.read_csv(os.path.join(data, "inputs", "development_samples.csv"), dtype={"sample_id": str})
    chk("fam 与 development_samples 顺序一致", fam.iid.tolist() == dev.sample_id.tolist(),
        "首个不一致: %s" % next((i for i, (a, b) in enumerate(zip(fam.iid, dev.sample_id)) if a != b), "无"))

    # ---- bed 解码 ----
    print("\n  解码 bed ...")
    codes = unpack_bed(os.path.join(dense, "PMB2026_full_241.bed"), len(bim), len(fam))
    dose = np.full(codes.shape, np.nan, dtype=np.float32)
    dose[codes == 0] = 2.0     # hom A1
    dose[codes == 2] = 1.0     # het
    dose[codes == 3] = 0.0     # hom A2
    print("  缺失比例 %.4f" % float(np.isnan(dose).mean()))

    # ---- 与 development_background.raw 比对 ----
    raw = pd.read_csv(os.path.join(data, "inputs", "development_background.raw"),
                      sep=r"\s+", dtype={"IID": str})
    low_ids = raw["IID"].tolist()
    low_names = list(raw.columns[6:])
    low = raw.iloc[:, 6:].to_numpy(float)
    low[low < 0] = np.nan
    low_pos = np.array([int(n.rsplit("_", 1)[0]) for n in low_names])
    low_allele = [n.rsplit("_", 1)[1] for n in low_names]
    chk("低密度样本顺序与 fam 一致", low_ids == fam.iid.tolist(), "")

    # 建立 (chr,pos) -> 高密度列
    dense_key = {}
    for i, (c, p) in enumerate(zip(bim.chr.to_numpy(), bim.pos.to_numpy())):
        dense_key[(int(c), int(p))] = i

    membership = os.path.join(hd, "code_point5_membership.csv")
    mem = pd.read_csv(membership) if os.path.exists(membership) else None
    # 低密度位点的染色体：由累计坐标反推
    def chr_of(global_pos):
        c = 1
        for k in range(1, 13):
            if global_pos >= OFFSETS[k]:
                c = k
        return c

    rng = np.random.default_rng(7)
    picks = rng.choice(len(low_pos), size=min(4000, len(low_pos)), replace=False)
    same = flip = mismatch = missing_diff = 0
    examples = []
    for i in picks:
        c = chr_of(int(low_pos[i]))
        pos = int(low_pos[i]) - OFFSETS[c]
        j = dense_key.get((c, pos))
        if j is None:
            continue
        a = low[:, i]
        b = dose[j]
        ma, mb = np.isnan(a), np.isnan(b)
        if not np.array_equal(ma, mb):
            missing_diff += 1
        both = ~ma & ~mb
        if both.any():
            if np.array_equal(a[both], b[both]):
                same += 1
            elif np.array_equal(2.0 - a[both], b[both]):
                flip += 1
            else:
                mismatch += 1
                if len(examples) < 3:
                    examples.append({"chr": c, "pos": pos, "raw_head": a[both][:5].tolist(),
                                     "dense_head": b[both][:5].tolist()})
    tot = same + flip + mismatch
    print("  primary 共享位点比对: 相同 %d / 反向 %d / 其它不符 %d / 缺失模式不同 %d"
          % (same, flip, mismatch, missing_diff))
    chk("primary 高密度与低密度基因型一致", tot > 0 and same == tot, "比对 %d 个" % tot)
    if examples:
        print("    示例:", examples)

    # ---- validation ----
    print("\n  validation dense ...")
    vmeta = np.load(os.path.join(dense, "raw_metadata_validation151.npz"), allow_pickle=True)
    vchr = vmeta["chromosome"].astype(int)
    vpos = vmeta["position_bp"].astype(np.int64)
    print("  validation 标记数 %d（bim %d，差 %d）" % (len(vchr), len(bim), len(bim) - len(vchr)))
    chk("validation 标记数 = 4,817,083", len(vchr) == 4817083, str(len(vchr)))
    inq = (vchr == 9) & (vpos >= 22110508) & (vpos <= 22187677)
    chk("validation 高密度已排除 qRSF9.2", int(inq.sum()) == 0, "区内 %d 个" % int(inq.sum()))

    # 每染色体内部序号：按 metadata 的染色体分组计算偏移
    order = np.argsort(vchr, kind="stable")
    assert np.array_equal(order, np.arange(len(vchr))), "metadata 未按染色体分组"
    counts = {int(c): int((vchr == c).sum()) for c in np.unique(vchr)}
    offs = {}
    run = 0
    for c in sorted(counts):
        offs[c] = run
        run += counts[c]
    vmap = {(int(c), int(p)): (int(c), i - offs[int(c)])
            for i, (c, p) in enumerate(zip(vchr, vpos))}
    vb = np.asarray(np.load(os.path.join(data, "validation_inputs", "background.npy"),
                            mmap_mode="r"), float)
    vbm = pd.read_csv(os.path.join(data, "validation_inputs", "background_markers.csv"))
    vso = pd.read_csv(os.path.join(data, "validation_inputs", "sample_order.csv"), dtype=str)
    chk("validation 样本数=151", len(vso) == 151, str(len(vso)))

    # 逐染色体读 npy，抽查共享位点
    rng2 = np.random.default_rng(11)
    picks2 = rng2.choice(len(vbm), size=min(3000, len(vbm)), replace=False)
    same2 = flip2 = mis2 = miss2 = 0
    cached = {}
    for i in picks2:
        c = int(vbm["chr"].iloc[i]); p = int(vbm["pos"].iloc[i])
        j = vmap.get((c, p))
        if j is None:
            continue
        cc, jj = j
        if cc not in cached:
            a = np.load(os.path.join(dense, "validation", "chr%d" % cc, "A1_GT.npy"), mmap_mode="r")
            cached[cc] = a
        b = np.asarray(cached[cc][jj], dtype=float)
        a = vb[:, i]
        ma, mb = np.isnan(a), np.isnan(b)
        if not np.array_equal(ma, mb):
            miss2 += 1
        both = ~ma & ~mb
        if both.any():
            if np.array_equal(a[both], b[both]):
                same2 += 1
            elif np.array_equal(2.0 - a[both], b[both]):
                flip2 += 1
            else:
                mis2 += 1
    tot2 = same2 + flip2 + mis2
    print("  validation 共享位点比对: 相同 %d / 反向 %d / 其它不符 %d / 缺失模式不同 %d"
          % (same2, flip2, mis2, miss2))
    chk("validation 高密度与低密度基因型一致", tot2 > 0 and same2 == tot2, "比对 %d 个" % tot2)

    # ---- QTL 区间覆盖 ----
    print("\n  QTL 区间在高密度背景中的位点数")
    regs = pd.read_csv(os.path.join(data, "inputs", "development_regions.csv"))
    vregs = pd.read_csv(os.path.join(data, "inputs", "validation_regions.csv"))
    cov = {}
    for tag, chrs, poss, rg in [("development", bim.chr.to_numpy(), bim.pos.to_numpy(), regs),
                                ("validation", vchr, vpos, vregs)]:
        rows = []
        for r in rg.itertuples():
            n = int(((chrs == int(r.chr)) & (poss >= r.start_bp) & (poss <= r.end_bp)).sum())
            rows.append({"qid": r.qid, "chr": int(r.chr), "n_dense": n})
            print("    %-10s %-12s 高密度位点 %6d" % (tag, r.qid, n))
        cov[tag] = rows
        print("    %s 合计 %d" % (tag, sum(x["n_dense"] for x in rows)))

    with open(os.path.join(out, "dense_verification.json"), "w", encoding="utf-8") as fh:
        json.dump({"checks": rep["checks"], "coverage": cov}, fh, indent=1, ensure_ascii=False)
    n_fail = sum(1 for c in rep["checks"] if not c["pass"])
    print("\n检查项 %d，失败 %d" % (len(rep["checks"]), n_fail))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify high-density genotype inputs without model fitting")
    parser.add_argument("--workspace", required=True,
                        help="workspace containing unpacked/dense and unpacked/data_prediction_inputs")
    parser.add_argument("--output", required=True, help="path for dense_verification.json")
    args = parser.parse_args()
    sys.exit(main(args.workspace, args.output))


