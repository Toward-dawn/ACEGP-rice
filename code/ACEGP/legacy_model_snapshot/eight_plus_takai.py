"""Add Takai to the frozen eight-region representation without refitting its PCs.

The existing five-plus-three policy is retained inside this opt-in outer layer.
Both existing new-region weights and the Takai weight are selected inside CV.
Setting the outer Takai weight to zero delegates to the original eight-region
functions, retaining the original dimensions, candidate order and RNG mapping.
"""
from __future__ import annotations
import numpy as np
from incremental_support import weights as eight_weights, weighted_rbf

POLICY = 'preserve_eight_plus_takai_v1'
KEY = 'incremental_takai_weight'
PREFIX = 'incremental_takai_'


class EightPlusTakaiDesign(tuple):
    def __new__(cls, values, reference):
        result = tuple.__new__(cls, values)
        result.reference = reference
        return result

    def __reduce__(self):
        return type(self), (tuple(self), self.reference)


def active(config):
    return config.get('incremental_takai_policy') == POLICY


def weights(config, block=False):
    field = 'incremental_takai_block_weight_grid' if block else 'incremental_takai_weight_grid'
    values = [float(v) for v in config[field]]
    assert values and values[0] == 0.0 and len(values) == len(set(values))
    assert all(0 <= v <= 1 for v in values)
    if block:
        assert set(values) <= {0.0, 1.0}
    return values


def reference_config(config):
    return {key: value for key, value in config.items() if not key.startswith(PREFIX)}


def install(base, block):
    if getattr(base, '_eight_takai_installed', False):
        return
    base._eight_takai_installed = True
    old_prepare, old_grid = base.prepare_fold_data, base.model_grid
    old_fit, old_inner = base.fit_predict_prepared, base.inner_oof
    old_design, old_block_grid, old_block_fit = block.design_matrices, block.hyper_grid, block.fit_predict

    def prepare(condition, gt, gx, at, ax, config):
        if not active(config) or condition == 'Genome_only':
            return old_prepare(condition, gt, gx, at, ax, reference_config(config))
        cut = int(config['incremental_takai_old_raw_columns'])
        assert cut == 77969 and at.shape[1] > cut and ax.shape[1] == at.shape[1]
        old = old_prepare(condition, gt, gx, at[:, :cut], ax[:, :cut], reference_config(config))
        ws = weights(config)
        if ws == [0.0]:
            return old
        nt, nx, qc = base.filter_markers_from_training(
            at[:, cut:], ax[:, cut:], float(config['functional_min_maf']),
            float(config['functional_max_marker_missing_rate']), 'Takai_region')
        pt, px = base.pca_scores(nt, nx, int(config['incremental_takai_pcs']))
        zt, zx = base.standardize(pt, px)
        kt, kx = base.linear_kernel(nt, nx)
        old_variants = old['incremental_variants']
        variants = {0.0: old}
        for weight in ws[1:]:
            expanded = {}
            for inner_weight, ref in old_variants.items():
                value = dict(ref)
                value['train_z'] = np.column_stack((ref['train_z'], np.sqrt(weight) * zt))
                value['test_z'] = np.column_stack((ref['test_z'], np.sqrt(weight) * zx))
                value['addon_k_train'] = (1 - weight) * ref['addon_k_train'] + weight * kt
                value['addon_k_test'] = (1 - weight) * ref['addon_k_test'] + weight * kx
                value['rbf_kernels'] = {
                    float(scale): weighted_rbf(ref['train_z'], ref['test_z'], zt, zx,
                                              weight, float(scale), base.EPS)
                    for scale in config['rkhs_bandwidth_scale_grid']}
                expanded[inner_weight] = value
            value = dict(old)
            value['incremental_variants'] = expanded
            variants[weight] = value
        result = dict(old)
        result['eight_plus_takai_variants'] = variants
        result['takai_marker_qc'] = qc
        return result

    def grid(model, condition, config):
        original = old_grid(model, condition, reference_config(config))
        if not active(config) or condition == 'Genome_only' or weights(config) == [0.0]:
            return original
        return [{**hyper, KEY: weight} for weight in weights(config) for hyper in original]

    def fit(model, condition, prepared, y, hyper, seed):
        if KEY not in hyper:
            return old_fit(model, condition, prepared, y, hyper, seed)
        ref_hyper = {key: value for key, value in hyper.items() if key != KEY}
        return old_fit(model, condition, prepared['eight_plus_takai_variants'][float(hyper[KEY])],
                       y, ref_hyper, seed)

    def inner(model, condition, genome, addon, y, folds, config, seed, fold_cache=None):
        if not active(config) or condition == 'Genome_only':
            return old_inner(model, condition, genome, addon, y, folds,
                             reference_config(config), seed, fold_cache=fold_cache)
        if weights(config) == [0.0]:
            cut = int(config['incremental_takai_old_raw_columns'])
            return old_inner(model, condition, genome, addon[:, :cut], y, folds,
                             reference_config(config), seed, fold_cache=fold_cache)
        original = old_grid(model, condition, reference_config(config))
        # The eight-panel wrapper used the underlying original hyper index,
        # irrespective of its existing five-plus-three region weight.
        n_weights = len(eight_weights(reference_config(config)))
        assert len(original) % n_weights == 0
        core_count = len(original) // n_weights
        candidates = base.model_grid(model, condition, config)
        predictions = [np.full(len(y), np.nan) for _ in candidates]
        if fold_cache is None:
            fold_cache = base.prepare_inner_fold_cache(condition, genome, addon, folds, config)
        for fold_number, (train, valid, prepared) in enumerate(fold_cache):
            for number, hyper in enumerate(candidates):
                predictions[number][valid] = base.fit_predict_prepared(
                    model, condition, prepared, y[train], hyper,
                    seed + (number % core_count) * 100 + fold_number)
        best_score, best_hyper, best_prediction = -np.inf, None, None
        for hyper, prediction in zip(candidates, predictions):
            score = base.pearson(y, prediction)
            if np.isfinite(score) and score > best_score:
                best_score, best_hyper, best_prediction = float(score), hyper, prediction
        if best_hyper is None:
            raise RuntimeError('Eight-plus-Takai inner selection failed')
        return best_prediction, best_hyper, best_score

    def design(gt, gx, at, ax, groups, config, seed):
        if not active(config):
            return old_design(gt, gx, at, ax, groups, config, seed)
        cut = int(config['incremental_takai_old_raw_columns'])
        names = list(config['incremental_takai_old_qids'])
        assert [str(name) for name, _ in groups[:8]] == names and len(names) == 8
        ref = old_design(gt, gx, at[:, :cut], ax[:, :cut], groups[:8], reference_config(config), seed)
        if weights(config, block=True) == [0.0]:
            return ref
        assert len(groups) == 9 and str(groups[-1][0]) == config['incremental_takai_qid']
        bypass = reference_config(config)
        bypass['qtl_incremental_policy'] = ''
        full = old_design(gt, gx, at, ax, groups, bypass, seed)
        assert full[-1] == names + [config['incremental_takai_qid']]
        assert full[2].shape[1] == 18 and ref[2].shape[1] == 16
        for a, b in [(full[0], ref[0]), (full[1], ref[1]),
                     (full[2][:, :16], ref[2]), (full[3][:, :16], ref[3])]:
            assert np.array_equal(a, b), 'Eight-panel block feature drift'
        return EightPlusTakaiDesign(full, ref)

    def block_grid(config):
        original = old_block_grid(reference_config(config))
        if not active(config) or weights(config, block=True) == [0.0]:
            return original
        return [{**hyper, KEY: weight} for weight in weights(config, block=True) for hyper in original]

    def block_fit(y, designs, hyper):
        if KEY not in hyper:
            return old_block_fit(y, designs, hyper)
        assert isinstance(designs, EightPlusTakaiDesign)
        ref_hyper = {key: value for key, value in hyper.items() if key != KEY}
        if float(hyper[KEY]) == 0.0:
            return old_block_fit(y, designs.reference, ref_hyper)
        assert float(hyper[KEY]) == 1.0
        old_new_weight = float(ref_hyper.pop('incremental_new_weight'))
        assert old_new_weight in {0.0, 1.0}
        if old_new_weight == 0.0:
            # Preserve the original ability to omit the three earlier additions,
            # while retaining the newly selected Takai block.
            indices = list(range(10)) + [16, 17]
            selected = (designs[0], designs[1], designs[2][:, indices], designs[3][:, indices],
                        designs[-1][:5] + designs[-1][-1:])
        else:
            selected = tuple(designs)
        return old_block_fit(y, selected, ref_hyper)

    base.prepare_fold_data, base.model_grid = prepare, grid
    base.fit_predict_prepared, base.inner_oof = fit, inner
    block.design_matrices, block.hyper_grid, block.fit_predict = design, block_grid, block_fit
