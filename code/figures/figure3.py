"""Draw Figure 3 from the supplied source data."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

"""Native vector drawing primitives for Figure 3b."""
from matplotlib.patches import Rectangle, Circle, Ellipse, Polygon, PathPatch, FancyArrowPatch
from matplotlib.path import Path as MP
from matplotlib.transforms import Affine2D
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.labelsize': 9, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'text.color': '#26343D', 'mathtext.fontset': 'dejavusans', 'lines.solid_capstyle': 'round', 'lines.solid_joinstyle': 'round'})
base_INK = '#26343D'
base_MUTED = '#657780'
base_RULE = '#D7E0E3'
base_GREEN = '#538A62'
base_BLUE = '#327CAC'
base_ORANGE = '#BE733D'
base_PALE_G = '#EFF5EE'
base_PALE_B = '#EDF5FA'
base_PALE_O = '#FBF2E9'
base_GW = '#95AEC8'
base_QTL = '#D76B9D'
base_TAP = '#9E84AA'
base_GOLD = '#CEA45E'

def base_canvas(w, h):
    fig = plt.figure(figsize=(w / 25.4, h / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, w), ylim=(h, 0))
    ax.axis('off')
    return (fig, ax)

def base_txt(ax, x, y, s, size=9, weight='normal', ha='left', color=base_INK, style='normal', **kwargs):
    return ax.text(x, y, s, fontsize=size, fontweight=weight, fontstyle=style, ha=ha, va='center', color=color, linespacing=1.18, zorder=10, **kwargs)

def base_line(ax, xs, ys, color=base_RULE, lw=0.65, **kw):
    return ax.plot(xs, ys, color=color, lw=lw, zorder=3, **kw)

def base_arrow(ax, x1, y1, x2, y2, color=base_MUTED, lw=0.7):
    p = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=6.5, color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=4)
    ax.add_patch(p)
    return p

def base_rect(ax, x, y, w, h, face, edge='none', lw=0.6):
    p = Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, lw=lw)
    ax.add_patch(p)
    return p

def base_gene(ax, x, y, s, size=10, ha='center'):
    return base_txt(ax, x, y, s, size, 'bold', ha, style='italic')

def base_draw_anther(ax, x, y, w, h):
    """Conceptual four-locule transverse section; no measured localization."""
    tr = Affine2D().scale(w / 71, -h / 76).translate(x - 14 * w / 71, y + 184 * h / 76) + ax.transData
    vertices = [(50, 170), (42, 184), (21, 181), (19, 159), (14, 145), (19, 119), (31, 113), (40, 108), (46, 113), (50, 117), (57, 110), (72, 108), (79, 122), (85, 140), (81, 163), (74, 174), (64, 184), (55, 179), (50, 170)]
    ax.add_patch(PathPatch(MP(vertices, [MP.MOVETO] + [MP.CURVE4] * 18), facecolor='#F2EADF', edgecolor='#B4A18E', lw=0.8, transform=tr))
    for center, width, height, color in [((50, 143), 9, 39, '#E6D5C0'), ((50, 143), 3.4, 9, '#CBA296')]:
        ax.add_patch(Ellipse(center, width, height, facecolor=color, edgecolor='none', transform=tr))
    for cx, cy in [(34, 158), (66, 158), (34, 130), (66, 130)]:
        for width, height, fc, ec, lw in [(23, 24, '#EEE4D6', '#B4A18E', 0.5), (19.6, 20.6, '#DDD1E3', base_TAP, 0.65), (15.7, 16.7, '#FFFDF7', base_TAP, 0.4)]:
            ax.add_patch(Ellipse((cx, cy), width, height, facecolor=fc, edgecolor=ec, lw=lw, transform=tr))
        for dx, dy, r in [(-3.2, 2.5, 2), (2.7, 2, 2.25), (-0.2, -3.1, 2.05)]:
            ax.add_patch(Circle((cx + dx, cy + dy), r, facecolor='#F7E1B2', edgecolor=base_GOLD, lw=0.45, transform=tr))
            ax.add_patch(Circle((cx + dx + 0.3, cy + dy + 0.25), 0.55, facecolor=base_GOLD, edgecolor='none', transform=tr))

    def point(u, v):
        return (x + (u - 14) * w / 71, y + (184 - v) * h / 76)
    return point

"""Annotation-based process schematics used in Figure 3b."""
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 8.5, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
diagram_GREEN = '#819E7E'
diagram_GREEN_FILL = '#E6EEDF'
diagram_PURPLE = '#9E84AA'
diagram_PURPLE_FILL = '#E9E0EF'
diagram_BLUE = '#6F9EBB'
diagram_BLUE_FILL = '#E4EFF5'
diagram_GOLD = '#CEA45E'
diagram_GOLD_FILL = '#FAE8C7'
diagram_REACTION = '#6D7E85'
diagram_mapping = [{'gene': 'OsPME7', 'gene_id': 'Os01g0880300', 'go': 'GO:0030599', 'caption': 'Pectin demethylesterification', 'concept': 'Methyl ester side groups are hydrolysed; the pectin backbone remains intact.', 'source': 'https://iubmb.qmul.ac.uk/enzyme/EC3/1/1/11.html'}, {'gene': 'OsPILS1', 'gene_id': 'Os09g0491740', 'go': 'GO:0055085', 'caption': 'IAA transport', 'concept': 'A generic carrier illustrates membrane transport, without specifying the organelle, physiological direction or protein structure.', 'source': 'https://rapdb.dna.naro.go.jp/locus/?name=Os09g0491740'}, {'gene': 'CYP19-4', 'gene_id': 'Os06g0708400', 'go': 'GO:0003755', 'caption': 'Prolyl isomerization', 'concept': 'Reversible cis-trans isomerization of a peptide bond preceding a proline residue; no peptide bond cleavage.', 'source': 'https://iubmb.qmul.ac.uk/enzyme/EC5/2/1/8.html'}, {'gene': 'Os11bglu36', 'gene_id': 'Os11g0683500', 'go': 'GO:0004553', 'caption': 'Glycoside hydrolysis', 'concept': 'Generic O-glycoside plus water gives a sugar hydroxyl and an alcohol; R is an unspecified aglycone.', 'source': 'https://amigo.geneontology.org/amigo/term/GO%3A0004553'}]

def diagram_stroke(ax, xy, color, width=0.8, **kwargs):
    xy = np.asarray(xy)
    ax.plot(xy[:, 0], xy[:, 1], color=color, lw=width, solid_capstyle='round', solid_joinstyle='round', zorder=4, **kwargs)

def diagram_bezier(ax, pts, color, width=0.75, fill='none', zorder=5):
    patch = PathPatch(MP(pts, [MP.MOVETO] + [MP.CURVE4] * (len(pts) - 1)), edgecolor=color, facecolor=fill, lw=width, capstyle='round', joinstyle='round', zorder=zorder)
    ax.add_patch(patch)
    return patch

def diagram_flow(ax, x1, y1, x2, y2, color=diagram_REACTION, reversible=False):
    p = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='<->' if reversible else '-|>', mutation_scale=7, linewidth=0.7, color=color, shrinkA=0, shrinkB=0, zorder=7)
    ax.add_patch(p)

def diagram_hexagon(ax, x, y, radius, edge, fill, lw=0.7):
    angles = np.arange(6) * np.pi / 3
    xy = np.column_stack([x + radius * np.cos(angles), y + radius * np.sin(angles)])
    ax.add_patch(Polygon(xy, facecolor=fill, edgecolor=edge, lw=lw, joinstyle='round', zorder=5))

def diagram_pectin(ax, cx, cy):
    """Two short pectin segments; ester hydrolysis does not cut the backbone."""
    for center, ester in [(cx - 12, True), (cx + 12, False)]:
        for x in [center - 3.3, center + 3.3]:
            diagram_hexagon(ax, x, cy, 2.35, diagram_GREEN, diagram_GREEN_FILL)
        diagram_stroke(ax, [(center - 7.4, cy), (center - 5.65, cy)], diagram_GREEN)
        diagram_stroke(ax, [(center - 0.95, cy), (center + 0.95, cy)], diagram_GREEN)
        diagram_stroke(ax, [(center + 5.65, cy), (center + 7.4, cy)], diagram_GREEN)
        diagram_stroke(ax, [(center - 3.3, cy - 2.05), (center - 3.3, cy - 5.15)], diagram_GOLD if ester else diagram_GREEN, 0.65)
        base_txt(ax, center - 3.3, cy - 7.4, 'COOCH₃' if ester else 'COOH', 6.8, ha='center', color='#A47D3D' if ester else diagram_GREEN, fontfamily='DejaVu Sans')
    diagram_flow(ax, cx - 3.8, cy, cx + 3.8, cy)
    base_txt(ax, cx, cy - 4, 'H₂O', 6.6, ha='center', color=base_MUTED, fontfamily='DejaVu Sans')
    base_txt(ax, cx, cy + 5.9, '+ CH₃OH', 6.6, ha='center', color='#A47D3D', fontfamily='DejaVu Sans')

def diagram_auxin_carrier(ax, cx, cy):
    """Generic bilayer and transport carrier, without localization claims."""
    for dx in np.arange(-18, 18.01, 2.35):
        if abs(dx) < 5.0:
            continue
        for side in [-1, 1]:
            yy = cy + side * 2.4
            ax.add_patch(Circle((cx + dx, yy), 0.57, facecolor='#D6E1D0', edgecolor='#91A58A', lw=0.4, zorder=4))
            diagram_stroke(ax, [(cx + dx - 0.25, yy - side * 0.55), (cx + dx - 0.32, cy + side * 0.2)], '#A7B79F', 0.45)
            diagram_stroke(ax, [(cx + dx + 0.25, yy - side * 0.55), (cx + dx + 0.38, cy + side * 0.2)], '#A7B79F', 0.45)
    left = [(cx - 4.4, cy - 4.5), (cx - 1.5, cy - 5.8), (cx - 1.1, cy - 2.8), (cx - 2.0, cy), (cx - 1.0, cy + 2.9), (cx - 1.4, cy + 5.5), (cx - 4.3, cy + 4.7), (cx - 6.2, cy + 2.3), (cx - 6.0, cy - 2.7), (cx - 4.4, cy - 4.5)]
    right = [(2 * cx - x, y) for x, y in left]
    diagram_bezier(ax, left, diagram_PURPLE, 0.7, diagram_PURPLE_FILL)
    diagram_bezier(ax, right, diagram_PURPLE, 0.7, diagram_PURPLE_FILL)
    diagram_flow(ax, cx, cy - 7.8, cx, cy + 8.0, diagram_PURPLE)
    for dx, dy, r in [(-13, -7.2, 0.9), (-8.5, -8.7, 0.8), (7.7, -7.4, 0.85), (-0.05, -9.4, 0.8), (2.1, 9.0, 0.8), (9.8, 7.4, 0.9)]:
        ax.add_patch(Circle((cx + dx, cy + dy), r, facecolor='#9B78A8', edgecolor='none', zorder=8))
    base_txt(ax, cx + 12, cy - 7.8, 'IAA', 6.8, ha='center', color='#886194')

def diagram_proline_conformation(ax, cx, cy, trans=False):
    """Minimal peptide-bond/proline skeleton with the flanking C-alpha atoms."""
    scale = 0.74
    flip = -1 if trans else 1

    def pt(x, y):
        return (cx + x * scale, cy + y * scale)

    def seg(coords, color=diagram_BLUE, width=0.8):
        diagram_stroke(ax, [pt(x, y) for x, y in coords], color, width)
    seg([(-8, -2.7), (-5, -3), (-2, 0)], diagram_BLUE, 0.85)
    seg([(-2.0, 0), (0.95, 0)], diagram_PURPLE, 1.2)
    seg([(-2.2, 0.3), (-3.9, 3.15)], diagram_BLUE, 0.7)
    seg([(-1.5, 0.6), (-3.2, 3.45)], diagram_BLUE, 0.7)
    ox, oy = pt(-4.2, 4.65)
    base_txt(ax, ox, oy, 'O', 6.7, ha='center', color=diagram_BLUE)
    nx, ny = pt(2, 0)
    base_txt(ax, nx, ny, 'N', 6.7, 'bold', ha='center', color=diagram_PURPLE)
    ring = [(2, 0), (5, -3 * flip), (8.6, -1.3 * flip), (7.6, 2.7 * flip), (3.9, 3.55 * flip)]
    ax.add_patch(Polygon([pt(x, y) for x, y in ring], closed=True, facecolor=diagram_BLUE_FILL, edgecolor='none', zorder=1))
    seg([(2.8, -0.8 * flip), (5, -3 * flip), (8.6, -1.3 * flip), (7.6, 2.7 * flip), (3.9, 3.55 * flip), (2.5, 1.0 * flip)], diagram_BLUE, 0.85)
    seg([(5, -3 * flip), (5.3, -5.8 * flip), (7.7, -7.1 * flip)], diagram_BLUE, 0.85)
    for x, y in [(-5, -3), (5, -3 * flip)]:
        ax.add_patch(Circle(pt(x, y), 0.47, facecolor=diagram_BLUE, edgecolor='none', zorder=8))

def diagram_prolyl_isomerization(ax, cx, cy):
    diagram_proline_conformation(ax, cx - 11, cy, False)
    diagram_proline_conformation(ax, cx + 11, cy, True)
    base_txt(ax, cx - 11, cy - 10.3, 'cis', 7.0, ha='center', color=base_MUTED)
    base_txt(ax, cx + 11, cy - 10.3, 'trans', 7.0, ha='center', color=base_MUTED)
    diagram_flow(ax, cx - 3.4, cy, cx + 3.4, cy, reversible=True)

def diagram_glycoside_hydrolysis(ax, cx, cy):
    """Generic glycosyl-O-R + H2O -> glycosyl-OH + HO-R."""
    top = cy - 6.3
    diagram_hexagon(ax, cx - 6.8, top, 3.25, diagram_GOLD, diagram_GOLD_FILL)
    diagram_stroke(ax, [(cx - 3.55, top), (cx - 0.9, top)], diagram_GOLD, 0.8)
    base_txt(ax, cx + 0.6, top, 'O', 7.8, ha='center', color='#A47D3D')
    diagram_stroke(ax, [(cx + 2.0, top), (cx + 5.2, top)], diagram_GOLD, 0.8)
    base_txt(ax, cx + 7.0, top, 'R', 8.0, ha='center', color=diagram_REACTION)
    diagram_flow(ax, cx, cy - 2.6, cx, cy + 2.1)
    base_txt(ax, cx + 6.1, cy - 0.2, 'H₂O', 6.8, ha='center', color=diagram_BLUE, fontfamily='DejaVu Sans')
    bottom = cy + 6.6
    diagram_hexagon(ax, cx - 11.5, bottom, 3.25, diagram_GOLD, diagram_GOLD_FILL)
    diagram_stroke(ax, [(cx - 8.25, bottom), (cx - 6.1, bottom)], diagram_GOLD, 0.8)
    base_txt(ax, cx - 3.7, bottom, 'OH', 7.6, ha='center', color='#A47D3D')
    base_txt(ax, cx + 2.1, bottom, '+', 7.6, ha='center', color=base_MUTED)
    base_txt(ax, cx + 9.6, bottom, 'HO–R', 7.6, ha='center', color=diagram_REACTION)
diagram_DIAGRAMS = [diagram_pectin, diagram_auxin_carrier, diagram_prolyl_isomerization, diagram_glycoside_hydrolysis]

"""Frozen heatmap rendering functions for Figure 3a."""
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
heatmap_INK = '#29343D'
heatmap_MID = '#89939D'
heatmap_BLUE = '#6F9CB9'
heatmap_PINK = '#C16D97'
heatmap_REP = '#835B83'
heatmap_REPS = ['SPW1', 'OsTGA10', 'OsPME7', 'OsPILS1', 'CYP19-4', 'Os11bglu36']
heatmap_REGIONS = {'row13': 'qSF1', 'row18': 'qtl_1.1', 'takai_tt2_01': 'RM1370 - anchored region', 'row07': 'qHt9a', 'row10': 'qRSF9.2', 'czx4_01': 'qHTH10', 'new4_04': 'qSSR11-1', 'row03': 'q12.1-associated\nregion'}
heatmap_WIDTH_MM = 200
heatmap_HEIGHT_MM = 182
heatmap_HEAT_HEIGHT_MM = 62
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'text.color': heatmap_INK})

def draw_heatmap(fig, genes, reg, counts, values):
    ax = fig.add_axes([25 / heatmap_WIDTH_MM, 104 / heatmap_HEIGHT_MM, 155 / heatmap_WIDTH_MM, heatmap_HEAT_HEIGHT_MM / heatmap_HEIGHT_MM])
    cmap = LinearSegmentedColormap.from_list('saved_effects', [heatmap_BLUE, '#FFFFFF', heatmap_PINK])
    im = ax.pcolormesh(np.arange(52) - 0.5, np.arange(4) - 0.5, values.T, shading='flat', cmap=cmap, norm=TwoSlopeNorm(vmin=-10, vcenter=0, vmax=10), rasterized=False)
    labels = [r.gene_id + (f'  {r.gene_symbol}' if r.gene_symbol in heatmap_REPS else '') for r in genes.itertuples()]
    ax.set(xticks=np.arange(51), xticklabels=labels, yticks=[0, 1, 2], yticklabels=['R996 (HT/CT)', 'R4628 (HT/CT)', 'Interaction'])
    ax.tick_params(axis='x', length=0, labelsize=5.8, pad=3, labelrotation=45)
    ax.tick_params(axis='y', length=0, labelsize=6.7, pad=5)
    for label, row in zip(ax.get_xticklabels(), genes.itertuples()):
        label.set_ha('right')
        label.set_va('top')
        label.set_rotation_mode('anchor')
        if row.gene_symbol in heatmap_REPS:
            label.set_fontweight('bold')
            label.set_color(heatmap_REP)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for yy in [0.5, 1.5]:
        ax.axhline(yy, color='white', lw=0.9)
    offset = 0
    heat_height_mm = heatmap_HEAT_HEIGHT_MM

    def above(mm):
        return 1 + mm / heat_height_mm
    xt = ax.get_xaxis_transform()
    for qid, n in zip(reg.qid, counts):
        if n:
            if offset:
                ax.axvline(offset - 0.5, color=heatmap_MID, lw=0.7)
            start, end = (offset - 0.35, offset + n - 0.65)
            middle = offset + (n - 1) / 2
            if n == 1:
                ax.plot([middle, middle], [above(0.8), above(7.8)], color=heatmap_MID, lw=0.65, transform=xt, clip_on=False)
                ax.text(middle, above(8.5), f'{heatmap_REGIONS[qid]} ({n})', ha='center', va='bottom', fontsize=6.3, transform=xt, clip_on=False)
            else:
                ax.plot([start, start, end, end], [above(0.8), above(2.4), above(2.4), above(0.8)], color=heatmap_MID, lw=0.65, transform=xt, clip_on=False)
                ax.text(middle, above(3.2), f'{heatmap_REGIONS[qid]} ({n})', ha='center', va='bottom', fontsize=6.3, transform=xt, clip_on=False)
            offset += int(n)
    ax.set_xlim(-0.5, 50.5)
    ax.set_ylim(2.5, -0.5)
    cax = fig.add_axes([185 / heatmap_WIDTH_MM, 107 / heatmap_HEIGHT_MM, 1.8 / heatmap_WIDTH_MM, 56 / heatmap_HEIGHT_MM])
    cb = fig.colorbar(im, cax=cax, orientation='vertical', ticks=[-10, -5, 0, 5, 10])
    cb.solids.set_rasterized(False)
    cb.solids.set_edgecolor('face')
    cb.ax.tick_params(labelsize=5.8, length=2, pad=2)
    cb.outline.set_visible(False)
    for spine in cb.ax.spines.values():
        spine.set_visible(False)
    cb.set_label('log$_2$ fold change', fontsize=6.3, labelpad=4)
    cb.ax.yaxis.set_label_position('right')
    return ax
parser=argparse.ArgumentParser()
parser.add_argument('--data',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
DATA_DIR=args.data
WIDTH_MM=200
HEIGHT_MM=190
HEAT_LEFT_MM=15
HEAT_RIGHT_MM=180
HEAT_WIDTH_MM=HEAT_RIGHT_MM-HEAT_LEFT_MM
B_LEFT_MM=5
B_TOP_MM=104
B_WIDTH_MM=190
B_HEIGHT_MM=82

def draw_left_updated(ax):
    # Move the literature-function labels down; keep citation details in the caption.
    base_gene(ax,23,24,'SPW1',10.5)
    base_txt(ax,23,30,'Stamen identity',8.8,ha='center')
    base_gene(ax,59,24,'OsTGA10',10.5)
    base_txt(ax,59,31,'Tapetum\ndevelopment',8.8,ha='center')
    point=base_draw_anther(ax,16,36,50,51)
    # Leader endpoints and divider follow the edited Illustrator PDF.
    leader_color='#69757B'
    diagram_stroke(ax,[(61.12,36.01),(54.32,46.87)],leader_color,.50)
    xp,yp=point(72.4,123.3)
    diagram_stroke(ax,[(xp,yp),(xp+4.7,yp+4.7)],leader_color,.50)
    base_txt(ax,59.77,yp+6.2,'Tapetum',8,color=base_INK)
    base_txt(ax,41,91,'Anther cross-section',9,ha='center')
    base_line(ax,[80.943,80.943],[19.24,91.24],base_RULE,.6)


def main():
    genes=pd.read_csv(DATA_DIR/'Figure3_all51_heat_response_values.csv')
    regions=pd.read_csv(DATA_DIR/'Figure3_all8_region_counts.csv')
    cols=['R996_log2FoldChange','R4628_log2FoldChange','interaction_log2FoldChange']
    values=genes[cols].to_numpy(float)
    counts=regions.anther_genes.to_numpy(int)
    assert len(genes)==51 and genes.gene_id.nunique()==51
    assert values.shape==(51,3) and np.isfinite(values).all()
    assert counts.tolist()==[4,6,20,8,1,0,6,6]
    assert abs(values).max()<=10
    # Exact heatmap reuse. Only its figure-relative axes positions change.
    global heatmap_WIDTH_MM, heatmap_HEIGHT_MM
    heatmap_WIDTH_MM=WIDTH_MM
    heatmap_HEIGHT_MM=HEIGHT_MM
    fig=plt.figure(figsize=(WIDTH_MM/25.4,HEIGHT_MM/25.4))
    heat_ax=draw_heatmap(fig,genes,regions,counts,values)
    heat_ax.set_position([HEAT_LEFT_MM/WIDTH_MM,(HEIGHT_MM-78)/HEIGHT_MM,
                          HEAT_WIDTH_MM/WIDTH_MM,62/HEIGHT_MM])
    # Narrower multiline row labels make room for a 10 mm leftward extension.
    heat_ax.set_yticklabels(['R996\n(HT/CT)','R4628\n(HT/CT)','Interaction'])
    heat_ax.tick_params(axis='y',length=0,labelsize=6.7,pad=3.5)
    for label in heat_ax.get_yticklabels():
        label.set_multialignment('center')
        label.set_va('center')
    fig.axes[1].set_position([185/WIDTH_MM,(HEIGHT_MM-75)/HEIGHT_MM,
                             1.8/WIDTH_MM,56/HEIGHT_MM])
    assert np.array_equal(np.asarray(heat_ax.collections[0].get_array()).reshape(3,51),values.T)
    # Retain the independent b canvas at its existing 190 x 82 mm physical size.
    bx=fig.add_axes([B_LEFT_MM/WIDTH_MM,
                    (HEIGHT_MM-B_TOP_MM-B_HEIGHT_MM)/HEIGHT_MM,
                    B_WIDTH_MM/WIDTH_MM,B_HEIGHT_MM/HEIGHT_MM])
    bx.set(xlim=(0,190),ylim=(94,12))
    bx.axis('off')
    draw_left_updated(bx)
    for t in list(bx.texts):
        if t.get_text()=='b':
            t.remove()
    # Compact each gene/diagram/caption group, with a wider inter-row gap.
    # Expand the vertical anchor spacing while retaining every drawing's size,
    # proportions and text size. The upper gene-name row remains anchored.
    vertical_factor=1.040297936202122
    centers=[[110.5, 38.56417110682971], [164, 38.56417110682971], [110.5, 79.34385020595289], [164, 78.30355226975078]]
    gene_y=[24.0, 24.0, 65.61191744808488, 65.61191744808488]
    caption_y=[51.56789530935623, 51.56789530935623, 91.10748511420354, 91.09094865587019]
    function_texts=[]
    gene_texts=[]
    for i,(item,draw,(cx,cy)) in enumerate(zip(diagram_mapping,diagram_DIAGRAMS,centers)):
        gene_texts.append(base_gene(bx,cx,gene_y[i],item['gene'],10.5))
        start=len(bx.texts)
        draw(bx,cx,cy)
        if i==2:
            # Keep cis/trans labels close to their molecular conformations.
            for t in list(bx.texts)[start:]:
                if t.get_text() in ['cis','trans']:
                    xx,yy=t.get_position()
                    t.set_position((xx,yy+2.0))
        function_texts.append(base_txt(bx,cx,caption_y[i],item['caption'],8.4,ha='center'))
    # The two panel letters share their left edge and type size.
    for letter,top in [('a',2.73),('b',106)]:
        fig.text(5/WIDTH_MM,1-top/HEIGHT_MM,letter,fontsize=13,
                 fontweight='bold',ha='left',va='top',color='#26343D')
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    outside=[]
    for t in [*fig.texts,*[t for ax in fig.axes for t in ax.texts],
              *heat_ax.get_xticklabels(),*heat_ax.get_yticklabels()]:
        bb=t.get_window_extent(renderer)
        if bb.x0<-.5 or bb.y0<-.5 or bb.x1>fig.bbox.width+.5 or bb.y1>fig.bbox.height+.5:
            outside.append(t.get_text())
    assert not outside,outside
    row_gaps=[]
    for j in [0,1]:
        upper=function_texts[j].get_window_extent(renderer)
        lower=gene_texts[j+2].get_window_extent(renderer)
        gap=(upper.y0-lower.y1)*25.4/fig.dpi
        assert gap>8.0,gap
        row_gaps.append(round(gap,3))
    stem=args.output/'Figure3'
    fig.savefig(stem.with_suffix('.pdf'),metadata={'Title':'Figure 3. Regional anther heat responses and functional context'})
    fig.savefig(stem.with_suffix('.svg'))
    fig.savefig(stem.with_suffix('.png'),dpi=350)
    fig.savefig(stem.with_suffix('.tif'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)


if __name__ == "__main__":
    main()
