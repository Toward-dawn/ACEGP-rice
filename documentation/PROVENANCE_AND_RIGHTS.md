# Provenance and licensing

This inventory distinguishes file provenance from redistribution licensing. Original software is licensed under MIT (`LICENSE`, Copyright (c) 2026 Anguo Gui), and original data, panel definitions and analysis outputs created by the study authors are licensed under CC BY 4.0. Reused third-party datasets, sequencing records and reference annotations remain subject to their original provider licences and citation requirements.

| Material | Provenance record in package | Primary source citation and licence treatment |
| --- | --- | --- |
| Primary prediction inputs and full PLINK source | Sample/marker files, registrations and source records under `prediction_inputs` and `random_controls/primary` | Phenotypes: Das et al. (2026), *Plant Mol Biol* 116:56 (`https://doi.org/10.1007/s11103-026-01716-8`); genotypes: 3K Rice Genomes Project, Wang et al. (2018), *Nature* 557:43–49 (`https://doi.org/10.1038/s41586-018-0063-9`). Original source terms retained. |
| Validation genotype/phenotype inputs | Run/accession mapping in `validation_phenotype_source.csv`, marker lists and registrations | Hu et al. (2022), *Front Genet* 13:983525 (`https://doi.org/10.3389/fgene.2022.983525`) and associated public resequencing reads. Original source terms retained. |
| Anther counts and sample metadata | BioProject PRJNA1215770; ENA run metadata and 12-sample map under `anther/sources` and `anther/inputs` | Zhou et al. (2025), *Int J Mol Sci* 26:3161 (`https://doi.org/10.3390/ijms26073161`); NCBI SRA BioProject `PRJNA1215770`. Original source terms retained. |
| Genome annotation and functional mappings | IRGSP-1.0 annotation filenames and versions; GFF and functional-membership inputs | IRGSP-1.0 reference genome, Kawahara et al. (2013), *Rice* 6:4 (`https://doi.org/10.1186/1939-8433-6-4`), and public RAP-DB / MSU functional annotations. Original provider terms retained. |
| Curated QTL regions | Eight source links in `region_name_mapping.csv`; IRGSP-1.0 coordinates and unchanged region IDs | Curated from published heat-tolerance QTL literature cited in the accompanying manuscript and Table 2; distinctions between mapping sources and naming sources are preserved in `region_name_mapping.csv`. |
| Study-created predictions, statistics, fixed panels and derived tables | Result files, registrations and figure source tables | Original contributions by the release creators (Anguo Gui and Youhong Song) licensed under CC BY 4.0. |

Raw sequencing reads are referenced through original public archives rather than rehosted. Third-party terms continue to apply to all reused material.

## Scope of the selected data licence

| Material group | Applicable licence treatment |
| --- | --- |
| Study-generated predictions, statistics, registrations and fixed-panel definitions | CC BY 4.0 for the authors' original contributions |
| Reused genotype/phenotype and sequencing inputs, including their derived representations | Original source terms remain applicable; no new licence is asserted for third-party rights |
| Reference-genome and functional annotations or copied literature material | Original provider/author terms retained |
| Tables combining original analyses with externally sourced columns | CC BY 4.0 applies to original contributions only; underlying third-party terms are retained |

No original attribution or copyright notice has been removed.
