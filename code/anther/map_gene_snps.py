"""Map the 51 anther genes to annotations and existing QTL SNPs."""

from pathlib import Path
import gzip
import json
import re
import numpy as np
import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Map frozen regional genes to existing SNP inputs."
    )
    parser.add_argument("output", type=Path)
    OUT = parser.parse_args().output.resolve()
    PACKAGE = next(
        p
        for p in Path(__file__).resolve().parents
        if (p / "documentation/PACKAGE_STATUS.json").is_file()
    )
    if OUT == PACKAGE or OUT.is_relative_to(PACKAGE):
        raise RuntimeError("Output must be outside the archive.")
    src = json.loads((OUT / "inputs/source_paths.json").read_text(encoding="utf-8"))
    genes = pd.read_csv(OUT / "inputs/frozen_51_anther_genes.csv")

    def save(df, p):
        df.to_csv(OUT / p, index=False, encoding="utf-8-sig")

    def dump(x, p):
        (OUT / p).write_text(
            json.dumps(x, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )

    def keys(m):
        return m.chr.astype(str) + ":" + m.pos.astype(str)

    features = {g: {"exon": set(), "CDS": set()} for g in genes.gene_id}
    gffgenes = {}
    with gzip.open(src["gff61"], "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            r = line.rstrip("\n").split("\t")
            if len(r) != 9:
                continue
            ids = {
                re.sub(r"^(Os\d{2})t", r"\1g", x) for x in re.findall(r"Os\d{2}[gt]\d{7}", r[8])
            } & set(features)
            for gene in ids:
                if r[2] == "gene":
                    gffgenes[gene] = (r[0], int(r[3]), int(r[4]), r[6])
                elif r[2] in ["exon", "CDS"]:
                    features[gene][r[2]].add((int(r[3]), int(r[4])))
    assert len(gffgenes) == 51
    assert (
        sum(bool(x["exon"]) for x in features.values()) == 51
    ), "Every selected protein-coding gene must resolve transcript-parent exon features."
    assert all(
        x["CDS"] for x in features.values()
    ), "All selected protein-coding genes must resolve their CDS features."
    for x in genes.itertuples():
        assert gffgenes[x.gene_id] == (str(x.chr), x.start_bp, x.end_bp, x.strand)
    countcoords = pd.read_csv(
        OUT / "inputs/counting_gene_coordinates_release60.csv", dtype={"Chr": str}
    ).set_index("Geneid")
    coord = []
    for g in genes.itertuples():
        x = countcoords.loc[g.gene_id]
        coord.append(
            dict(
                gene_id=g.gene_id,
                counted_chr=x.Chr,
                counted_start=x.Start,
                counted_end=x.End,
                mapping_chr=g.chr,
                mapping_start=g.start_bp,
                mapping_end=g.end_bp,
                exactly_same_gene_body=(
                    str(x.Chr) == str(g.chr)
                    and int(x.Start) == g.start_bp
                    and int(x.End) == g.end_bp
                ),
            )
        )
    save(pd.DataFrame(coord), "validation/51_gene_release60_release61_coordinate_comparison.csv")
    ann = pd.read_csv(OUT / "inputs/functional_term_membership.csv", dtype=str).fillna("")
    bm = pd.read_csv(OUT / "inputs/51_gene_name_mapping.csv", dtype=str).fillna("")
    names = bm.groupby("gene_id").agg(
        gene_symbol=("gene_symbol", lambda z: ";".join(sorted({x for x in z if x}))),
        entrez_id=("entrez_id", lambda z: ";".join(sorted({x for x in z if x}))),
    )
    reg = pd.read_csv(src["development_regions"])
    qnames = {
        "row03": "q12.1-associated region",
        "row07": "qHt9a",
        "row10": "qRSF9.2",
        "row13": "qSF1",
        "row18": "qtl_1.1",
        "new4_04": "qSSR11-1",
        "czx4_01": "qHTH10",
        "takai_tt2_01": "RM1370-anchored region",
    }
    genes["qtl_name"] = genes.region.map(qnames)
    assert genes.qtl_name.notna().all()
    genes = genes.merge(names, on="gene_id", how="left").fillna(
        {"gene_symbol": "", "entrez_id": ""}
    )
    rules = {
        "carbon_energy": {
            "label_zh": "碳水化合物与能量供应",
            "GO_name_regex": r"starch|sucrose|carbohydrate metabolic|glycoly|glucose metabolic|tricarboxylic|ATP biosynthetic|oxidative phosphorylation|electron transport coupled|cellular respiration|pyruvate metabolic",
            "KEGG_ids": [
                "osa00010",
                "osa00020",
                "osa00030",
                "osa00190",
                "osa00500",
                "osa00520",
                "osa00620",
                "osa01200",
            ],
        },
        "protein_homeostasis": {
            "label_zh": "蛋白质稳态与热响应",
            "GO_name_regex": r"protein folding|protein refolding|unfolded protein|chaperone|heat shock|response to heat|proteasom|endoplasmic reticulum stress|ER-associated",
            "KEGG_ids": ["osa04141", "osa03050"],
        },
        "redox": {
            "label_zh": "氧化还原与活性氧调节",
            "GO_name_regex": r"oxidative stress|redox homeostasis|antioxidant|reactive oxygen|hydrogen peroxide catabolic|peroxidase|glutathione|thioredoxin",
            "KEGG_ids": ["osa00480"],
        },
        "anther_pollen": {
            "label_zh": "花药与花粉相关功能",
            "GO_name_regex": r"pollen|anther|tapetum|tapetal|microspore|sporopollenin",
            "KEGG_ids": [],
        },
    }
    dump(
        {
            "status": "ANNOTATION_DICTIONARY_NOT_A_SIGNIFICANCE_TEST",
            "rules": rules,
            "overlapping_membership_allowed": True,
            "interpretation": "generic annotation is not evidence of a heat-fertility role; reported locus-specific experiments require a separate primary reference",
        },
        "inputs/mechanism_annotation_dictionary.json",
    )
    ann["mechanism_groups"] = ""
    for k, v in rules.items():
        sel = (
            (ann.source == "GO")
            & ann.term_name.str.contains(v["GO_name_regex"], case=False, regex=True)
        ) | ann.term_id.isin(v["KEGG_ids"])
        ann.loc[sel, "mechanism_groups"] = ann.loc[sel, "mechanism_groups"].map(
            lambda x: (x + ";" + k).strip(";")
        )
    function51 = ann[ann.gene_id.isin(genes.gene_id)].copy()
    save(function51, "functional/51_gene_GO_KEGG_annotation_long.csv")
    frows = []
    for g in genes.itertuples():
        a = function51[function51.gene_id == g.gene_id]
        terms = {x for s in a.mechanism_groups for x in s.split(";") if x}
        frows.append(
            dict(
                gene_id=g.gene_id,
                gene_symbol=g.gene_symbol,
                qtl_name=g.qtl_name,
                mechanism_groups=";".join(sorted(terms)),
                mechanism_groups_zh="；".join(rules[t]["label_zh"] for t in sorted(terms))
                or "暂无上述四类直接注释",
                GO_ids=";".join(a.loc[a.source == "GO", "term_id"]),
                GO_names="; ".join(a.loc[a.source == "GO", "term_name"]),
                KEGG_ids=";".join(a.loc[a.source == "KEGG", "term_id"]),
                KEGG_names="; ".join(a.loc[a.source == "KEGG", "term_name"]),
                annotation_evidence="existing Ensembl/GO and KEGG mappings; locus-specific experimental evidence not inferred",
            )
        )
    save(pd.DataFrame(frows), "tables/51_genes_function_annotation.csv")
    cohorts = {}
    snprows = []
    mrows = []
    alleles = {}
    for pop in ["development", "validation"]:
        m = pd.read_csv(src[pop + "_markers"])
        m.chr = m.chr.astype(str).str.replace("chr", "", case=False).astype(int)
        m.pos = m.pos.astype(int)
        assert not m.duplicated(["chr", "pos"]).any()
        x = np.load(src[pop + "_matrix"], mmap_mode="r")
        assert x.shape[1] == len(m)
        assert x.shape[0] == (241 if pop == "development" else 151)
        assert len(m) == (67985 if pop == "development" else 67104)
        popregions = pd.read_csv(src[pop + "_regions"])
        included = set(popregions.qid)
        union = set()
        bodyunion = set()
        exonunion = set()
        cdsunion = set()
        for g in genes.itertuples():
            choose = (m.chr == g.chr) & m.pos.between(max(1, g.start_bp - 5000), g.end_bp + 5000)
            idx = (
                np.flatnonzero(choose.to_numpy())
                if g.region in included
                else np.array([], dtype=int)
            )
            pos = m.pos.to_numpy()[idx]
            body = (pos >= g.start_bp) & (pos <= g.end_bp)

            def has_intervals(intervals):
                a = np.zeros(len(pos), dtype=bool)
                for left, right in intervals:
                    a |= (pos >= left) & (pos <= right)
                return a

            ex = has_intervals(features[g.gene_id]["exon"])
            cd = has_intervals(features[g.gene_id]["CDS"])
            data = np.asarray(x[:, idx], dtype=float)
            obs = np.isfinite(data)
            freq = np.nanmean(data, axis=0) / 2 if len(idx) else np.array([])
            maf = np.minimum(freq, 1 - freq)
            missing = 1 - obs.mean(axis=0) if len(idx) else np.array([])
            hetero = (
                (np.isclose(data, 1) & obs).sum(axis=0) / np.maximum(obs.sum(axis=0), 1)
                if len(idx)
                else np.array([])
            )
            union.update(idx.tolist())
            bodyunion.update(idx[body].tolist())
            exonunion.update(idx[ex].tolist())
            cdsunion.update(idx[cd].tolist())
            mrows.append(
                dict(
                    population=pop,
                    gene_id=g.gene_id,
                    qtl_name=g.qtl_name,
                    region=g.region,
                    in_population_panel=g.region in included,
                    n_input_snps=len(idx),
                    n_gene_body_snps=int(body.sum()),
                    n_flank_snps=int((~body).sum()),
                    n_exonic_snps=int(ex.sum()),
                    n_CDS_snps=int(cd.sum()),
                    mean_input_MAF=float(maf.mean()) if len(maf) else None,
                    n_MAF_ge005=int((maf >= 0.05).sum()),
                    mean_input_missingness=float(missing.mean()) if len(missing) else None,
                    mean_input_heterozygote_fraction=float(hetero.mean()) if len(hetero) else None,
                    n_fractional_genotype_cells=int(
                        (obs & ~np.isclose(data, np.round(data), atol=1e-7)).sum()
                    ),
                    genotype_context=(
                        "imputed GT hard calls; missingness is post-imputation, not original observed coverage"
                        if pop == "validation"
                        else "frozen development QTL input matrix before training-fold QC"
                    ),
                )
            )
            for j, ix in enumerate(idx):
                r = m.iloc[ix]
                part = (
                    "gene_body"
                    if body[j]
                    else (
                        "upstream_flank"
                        if (r.pos < g.start_bp) == (g.strand == "+")
                        else "downstream_flank"
                    )
                )
                snprows.append(
                    dict(
                        population=pop,
                        gene_id=g.gene_id,
                        qtl_name=g.qtl_name,
                        region=g.region,
                        marker_column_index_zero_based=int(ix),
                        chr=int(r.chr),
                        pos=int(r.pos),
                        a1=r.a1,
                        a2=r.a2,
                        position_key=f"{r.chr}:{r.pos}",
                        location=part,
                        within_exon=bool(ex[j]),
                        within_CDS=bool(cd[j]),
                        input_MAF=float(maf[j]),
                        post_input_missingness=float(missing[j]),
                        input_heterozygote_fraction=float(hetero[j]),
                    )
                )
        used = np.array(sorted(union), dtype=int)
        qm = m.iloc[used].copy()
        qm.insert(0, "marker_column_index_zero_based", used)
        save(qm, f"tables/{pop}_anther_gene_unique_SNPs.csv")
        alleles[pop] = {(int(r.chr), int(r.pos)): (r.a1, r.a2) for r in qm.itertuples()}
        cohorts[pop] = {
            "samples": int(x.shape[0]),
            "fixed_QTL_input_snps": len(m),
            "genes_in_frozen_population_panel": int(genes.region.isin(included).sum()),
            "unique_anther_gene_pm5kb_snps": len(union),
            "unique_gene_body_snps": len(bodyunion),
            "unique_exon_snps": len(exonunion),
            "unique_CDS_snps": len(cdsunion),
            "per_gene_SNP_counts_not_additive": True,
            "new_variant_calling": False,
        }
    mp = pd.DataFrame(mrows)
    save(mp, "tables/51_genes_SNP_counts_long.csv")
    snp = pd.DataFrame(snprows)
    snp.to_csv(
        OUT / "tables/gene_SNP_correspondence_long.csv.gz",
        index=False,
        encoding="utf-8",
        compression="gzip",
    )
    wide = genes[
        [
            "gene_id",
            "gene_symbol",
            "entrez_id",
            "region",
            "qtl_name",
            "chr",
            "start_bp",
            "end_bp",
            "strand",
        ]
    ].copy()
    for pop in cohorts:
        z = mp[mp.population == pop].drop(columns=["population", "qtl_name", "region"])
        z = z.rename(columns={c: pop + "_" + c for c in z if c != "gene_id"})
        wide = wide.merge(z, on="gene_id", how="left", validate="one_to_one")
    save(wide, "tables/51_genes_SNP_correspondence.csv")
    common = set(alleles["development"]) & set(alleles["validation"])
    disagree = [str(k) for k in common if alleles["development"][k] != alleles["validation"][k]]
    assert not disagree
    dump(
        {
            "state": "GENE_SNP_MAPPING_COMPLETE",
            "cohorts": cohorts,
            "common_positions": len(common),
            "common_position_A1_A2_exactly_match": len(common),
            "mismatched_alleles": disagree,
            "gene_coordinate_release60_to61_identical": all(
                x["exactly_same_gene_body"] for x in coord
            ),
            "data_counts_before_training_fold_QC": True,
            "no_predictive_contribution_estimated": True,
        },
        "validation/GENE_SNP_MAPPING_ACCEPTANCE.json",
    )
    print(json.dumps(cohorts, ensure_ascii=False))
    print(
        pd.DataFrame(frows)[
            ["gene_id", "gene_symbol", "qtl_name", "mechanism_groups", "GO_names", "KEGG_names"]
        ].to_string(index=False, max_colwidth=110)
    )


if __name__ == "__main__":
    main()
