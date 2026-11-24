# Figure rendering

The scripts draw Figures 3, 4 and 5 from the matching companion data. The manuscript figures were subsequently adjusted in Illustrator, so the script outputs need not match the final layout exactly.

Install Python with `numpy`, `pandas`, `matplotlib`, `scipy` and `Pillow`. For example, in a suitable Python environment:

```bash
python -m pip install numpy pandas matplotlib scipy pillow
```

With the companion `02_data/` directory at the repository root, run:

```bash
python code/figures/figure3.py --data 02_data/figure_source/Figure3 --output /path/to/figure3
python code/figures/figure4.py --data 02_data/figure_source/Figure4 --output /path/to/figure4
python code/figures/figure5.py --data 02_data/figure_source/Figure5 --output /path/to/figure5
```

The scripts write PDF, SVG and PNG; Figure 3 also writes TIFF. Fonts may differ across systems. These commands draw existing results and do not fit models.

Figure 3 reads `Figure3_all51_heat_response_values.csv` and `Figure3_all8_region_counts.csv`. Figure 4 reads `Figure3_development_source_data.csv` and `paired_repeat_results.csv`. Figure 5 reads `Figure4_random100_source_data.csv`, `Figure4_reference_values.csv`, `TABLE_A_panel_means.csv` and `review_per_repeat.csv`. Historical names are retained for compatibility; see the companion data documentation for experiment definitions.
