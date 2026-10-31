"""Preserve the old panel representation and tune a separate new-region weight.

Opt-in hooks, installed only in an isolated ACEGP model copy. All QC, imputation,
PCA, kernels and weight selection remain within the relevant training fold.
The zero candidate has the original feature dimensions and original RNG seeds.
"""
from __future__ import annotations
import itertools
import numpy as np

POLICY='preserve_old_v1'
WEIGHT_KEY='incremental_new_weight'

class IncrementalDesign(tuple):
    def __new__(cls, values, old_count, old_group_count):
        result=tuple.__new__(cls,values)
        result.old_count=int(old_count)
        result.old_group_count=int(old_group_count)
        return result
    def __reduce__(self):
        return (type(self),(tuple(self),self.old_count,self.old_group_count))

def active(config):return config.get('qtl_incremental_policy')==POLICY
def weights(config,block=False):
    key='incremental_block_new_weight_grid' if block else 'incremental_new_weight_grid'
    values=[float(x) for x in config.get(key,[0.0,1.0] if block else [0.0,.25,.5,1.0])]
    assert values and values[0]==0.0 and len(set(values))==len(values)
    assert all(0.0<=x<=1.0 for x in values)
    if block:assert set(values).issubset({0.0,1.0})
    return values

def weighted_rbf(old_train_z,old_test_z,new_train_z,new_test_z,weight,scale,eps):
    """Weight AFTER standardization; standardizing again would cancel the weight."""
    dimensions=max(1,old_train_z.shape[1])
    train_dot=(old_train_z@old_train_z.T+weight*(new_train_z@new_train_z.T))/dimensions
    test_dot=(old_test_z@old_train_z.T+weight*(new_test_z@new_train_z.T))/dimensions
    train_diag=np.diag(train_dot)
    test_diag=(np.sum(old_test_z**2,axis=1)+weight*np.sum(new_test_z**2,axis=1))/dimensions
    train_d2=np.clip(train_diag[:,None]+train_diag[None,:]-2*train_dot,0,None)
    test_d2=np.clip(test_diag[:,None]+train_diag[None,:]-2*test_dot,0,None)
    upper=train_d2[np.triu_indices_from(train_d2,k=1)]
    bandwidth=float(np.median(upper[upper>0])) if np.any(upper>0) else 1.0
    bandwidth=max(bandwidth*float(scale),eps)
    return np.exp(-train_d2/bandwidth),np.exp(-test_d2/bandwidth)

def install(base,block):
    if getattr(base,'_incremental_installed',False):return
    base._incremental_installed=True
    original_prepare=base.prepare_fold_data
    original_grid=base.model_grid
    original_fit=base.fit_predict_prepared
    original_inner=base.inner_oof
    original_design=block.design_matrices
    original_block_grid=block.hyper_grid
    original_block_fit=block.fit_predict

    def prepare(condition,gt,gx,at,ax,config):
        if not active(config) or condition=='Genome_only':
            return original_prepare(condition,gt,gx,at,ax,config)
        cut=int(config['incremental_old_raw_columns'])
        assert at.shape[1]>cut and ax.shape[1]==at.shape[1]
        old=original_prepare(condition,gt,gx,at[:,:cut],ax[:,:cut],config)
        ws=weights(config)
        if ws==[0.0]:return old
        nt,nx,audit=base.filter_markers_from_training(at[:,cut:],ax[:,cut:],
            float(config['functional_min_maf']),float(config['functional_max_marker_missing_rate']),'new_region')
        new_pc_train,new_pc_test=base.pca_scores(nt,nx,int(config['incremental_new_pcs']))
        nz,nxz=base.standardize(new_pc_train,new_pc_test)
        new_k,new_kx=base.linear_kernel(nt,nx)
        variants={0.0:old}
        for w in ws[1:]:
            variant=dict(old)
            variant['train_z']=np.column_stack((old['train_z'],np.sqrt(w)*nz))
            variant['test_z']=np.column_stack((old['test_z'],np.sqrt(w)*nxz))
            variant['addon_k_train']=(1-w)*old['addon_k_train']+w*new_k
            variant['addon_k_test']=(1-w)*old['addon_k_test']+w*new_kx
            variant['rbf_kernels']={float(s):weighted_rbf(old['train_z'],old['test_z'],nz,nxz,w,float(s),base.EPS)
                for s in config['rkhs_bandwidth_scale_grid']}
            variants[w]=variant
        result=dict(old)
        result['incremental_variants']=variants
        result['incremental_new_marker_qc']=audit
        return result

    def grid(model,condition,config):
        initial=original_grid(model,condition,config)
        if not active(config) or condition=='Genome_only' or weights(config)==[0.0]:return initial
        # Original candidates first: zero preserves original hyper-index RNG seeds.
        return [{**hyper,WEIGHT_KEY:w} for w in weights(config) for hyper in initial]

    def fit(model,condition,prepared,y,hyper,seed):
        if WEIGHT_KEY not in hyper:return original_fit(model,condition,prepared,y,hyper,seed)
        w=float(hyper[WEIGHT_KEY])
        assert 'incremental_variants' in prepared
        old_hyper={k:v for k,v in hyper.items() if k!=WEIGHT_KEY}
        return original_fit(model,condition,prepared['incremental_variants'][w],y,old_hyper,seed)

    def inner(model,condition,genome,addon,y,folds,config,seed,fold_cache=None):
        if not active(config) or condition=='Genome_only' or weights(config)==[0.0]:
            return original_inner(model,condition,genome,addon,y,folds,config,seed,fold_cache=fold_cache)
        initial=original_grid(model,condition,config)
        candidates=base.model_grid(model,condition,config)
        preds=[np.full(len(y),np.nan) for _ in candidates]
        if fold_cache is None:fold_cache=base.prepare_inner_fold_cache(condition,genome,addon,folds,config)
        for fold_no,(train,valid,prepared) in enumerate(fold_cache):
            for number,hyper in enumerate(candidates):
                # Same estimator RNG for an original hyper, irrespective of new weight.
                hyper_index=number%len(initial)
                preds[number][valid]=base.fit_predict_prepared(model,condition,prepared,y[train],hyper,
                    seed+hyper_index*100+fold_no)
        best_score=-np.inf;best_hyper=None;best_pred=None
        for hyper,pred in zip(candidates,preds):
            score=base.pearson(y,pred)
            if np.isfinite(score) and score>best_score:
                best_score,best_hyper,best_pred=float(score),hyper,pred
        if best_hyper is None:raise RuntimeError('Incremental inner screening failed')
        return best_pred,best_hyper,best_score

    def design(gt,gx,at,ax,groups,config,seed):
        if not active(config):return original_design(gt,gx,at,ax,groups,config,seed)
        cut=int(config['incremental_old_raw_columns'])
        expected=list(config['incremental_old_qids'])
        assert [str(name) for name,_ in groups[:len(expected)]]==expected
        added=list(config.get('incremental_new_qids',[str(name) for name,_ in groups[len(expected):]]))
        assert added and len(set(expected+added))==len(expected)+len(added)
        assert [str(name) for name,_ in groups[len(expected):]]==added
        if weights(config,block=True)==[0.0]:
            return original_design(gt,gx,at[:,:cut],ax[:,:cut],groups[:len(expected)],config,seed)
        result=original_design(gt,gx,at,ax,groups,config,seed)
        assert result[-1]==expected+added
        old_count=int(config['qtl_pcs_per_block'])*len(expected)
        assert result[2].shape[1]==old_count+int(config['qtl_pcs_per_block'])*len(added)
        return IncrementalDesign(result,old_count,len(expected))

    def block_grid(config):
        initial=original_block_grid(config)
        if not active(config) or weights(config,block=True)==[0.0]:return initial
        return [{**hyper,WEIGHT_KEY:w} for w in weights(config,block=True) for hyper in initial]

    def block_fit(y,designs,hyper):
        if WEIGHT_KEY not in hyper:return original_block_fit(y,designs,hyper)
        assert isinstance(designs,IncrementalDesign)
        w=float(hyper[WEIGHT_KEY]);clean={k:v for k,v in hyper.items() if k!=WEIGHT_KEY}
        if w==0:
            values=(designs[0],designs[1],designs[2][:,:designs.old_count],
                designs[3][:,:designs.old_count],designs[-1][:designs.old_group_count])
            return original_block_fit(y,values,clean)
        assert w==1.0
        return original_block_fit(y,tuple(designs),clean)

    base.prepare_fold_data=prepare
    base.model_grid=grid
    base.fit_predict_prepared=fit
    base.inner_oof=inner
    block.design_matrices=design
    block.hyper_grid=block_grid
    block.fit_predict=block_fit
