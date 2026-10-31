"""Exact content-addressed reuse; no alteration of the published model equations.

Caches are keyed by actual train/test arrays, phenotype, folds, random seed,
and every statistical configuration value. Only file paths and reporting scope
are excluded. Test phenotypes NEVER enter inner selection or feature caches.
"""
from __future__ import annotations
import functools
import hashlib
import json
import os
import pickle
from pathlib import Path
import numpy as np

CACHE_VERSION='incremental-preserve-eight-takai-1'
LEGACY_VERSION='boundary-runtime-1'
EXCLUDED={'genotype','phenotype','functional_panels','qtl_loci_path','qtl_marker_coordinate_path',
          'expected_accessions','outer_repeats','analysis_repeat_ids','analysis_fold_ids',
          'analysis_sample_ids','boundary_cache_path','boundary_run_fingerprint'}


def hash_value(h,x):
    if isinstance(x,np.ndarray):
        h.update(str((x.dtype.str,x.shape)).encode())
        a=np.ascontiguousarray(x)
        h.update(memoryview(a).cast('B'))
    elif isinstance(x,dict):
        for k in sorted(x):
            hash_value(h,k);hash_value(h,x[k])
    elif isinstance(x,(tuple,list)):
        h.update(str(len(x)).encode())
        for a in x: hash_value(h,a)
    else: h.update(repr(x).encode())


def clean_config(c, condition=None):
    values={k:v for k,v in c.items() if k not in EXCLUDED}
    if condition=='Genome_only':
        values={k:v for k,v in values.items() if not k.startswith('incremental_') and k!='qtl_incremental_policy'}
    return values


def cached(root,kind,args,compute):
    if not root: return compute()
    h=hashlib.sha256();hash_value(h,(CACHE_VERSION,kind,args));key=h.hexdigest()
    p=Path(root)/kind/key[:2]/(key+'.pkl');p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        with p.open('rb') as f: return pickle.load(f)
    # Read-only reuse of identical unchanged transformations/Genome-only fits.
    legacy_root=os.environ.get('ACEGP_LEGACY_READONLY_CACHE')
    eligible=kind in {'exact_pca','linear_kernel','block_pcs'}
    if kind=='prepared': eligible=isinstance(args[0],str) and args[0]=='Genome_only'
    if kind in {'inner_oof','outer_fit'}: eligible=args[1]=='Genome_only'
    legacy_path=None
    if legacy_root and eligible:
        old_hash=hashlib.sha256();hash_value(old_hash,(LEGACY_VERSION,kind,args));old_key=old_hash.hexdigest()
        legacy_path=Path(legacy_root)/kind/old_key[:2]/(old_key+'.pkl')
    if legacy_path is not None and legacy_path.exists():
        with legacy_path.open('rb') as f: value=pickle.load(f)
    else: value=compute()
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f: pickle.dump(value,f,protocol=5)
    os.replace(tmp,p)
    return value


def expected_oof_count(config,base,n,repeat):
    folds=base.folds_for(n,int(config['seed'])+repeat,int(config['outer_folds']))
    selected=config.get('analysis_fold_ids') or list(range(1,len(folds)+1))
    return sum(len(folds[i-1]) for i in selected)


def install(base,block):
    from incremental_support import install as incremental_install
    incremental_install(base,block)
    from eight_plus_takai import install as takai_install
    takai_install(base,block)
    if getattr(base,'_boundary_runtime_installed',False): return
    base._boundary_runtime_installed=True
    active={'root':None}
    original_pca=base.pca_scores
    def pca(train,test,n_components):
        return cached(active['root'],'exact_pca',(train,test,n_components),
                      lambda:original_pca(train,test,n_components))
    base.pca_scores=pca
    original_kernel=base.linear_kernel
    def kernel(train,test):
        return cached(active['root'],'linear_kernel',(train,test),lambda:original_kernel(train,test))
    base.linear_kernel=kernel
    original_randomized=block.randomized_pcs
    def randomized(train,test,n_components,random_state):
        return cached(active['root'],'block_pcs',(train,test,n_components,random_state),
                      lambda:original_randomized(train,test,n_components,random_state))
    block.randomized_pcs=randomized
    original_load=base.load_inputs
    def load_inputs(config):
        data=list(original_load(config))
        requested=config.get('analysis_sample_ids')
        if requested is not None:
            assert len(requested)==len(set(requested))
            lookup={str(x):i for i,x in enumerate(data[0])}
            ix=np.array([lookup[str(x)] for x in requested],dtype=int)
            data[0]=data[0][ix];data[1]=data[1][ix]
            data[2]={k:v[ix] for k,v in data[2].items()};data[4]=data[4][ix]
            data[6]={**data[6],'n_accessions':len(ix),'analysis_subset_ids':list(requested),
                     'analysis_subset_sha256':hashlib.sha256(json.dumps(requested).encode()).hexdigest(),
                     'target_mean':float(np.mean(data[4])),'target_sd':float(np.std(data[4],ddof=1)),
                     'target_min':float(np.min(data[4])),'target_max':float(np.max(data[4])),
                     'target_q1':float(np.quantile(data[4],0.25)),'target_median':float(np.median(data[4])),
                     'target_q3':float(np.quantile(data[4],0.75))}
        return tuple(data)
    base.load_inputs=load_inputs
    original_prepare=base.prepare_fold_data
    def prepare(condition,gt,gx,at,ax,config):
        active['root']=config.get('boundary_cache_path')
        return cached(config.get('boundary_cache_path'),'prepared',
                      (condition,gt,gx,at,ax,clean_config(config,condition)),
                      lambda:original_prepare(condition,gt,gx,at,ax,config))
    base.prepare_fold_data=prepare
    original_inner=base.inner_oof
    def inner(model,condition,genome,addon,y,folds,config,seed,fold_cache=None):
        return cached(config.get('boundary_cache_path'),'inner_oof',
                      (model,condition,genome,addon,y,folds,clean_config(config,condition),seed),
                      lambda:original_inner(model,condition,genome,addon,y,folds,config,seed,fold_cache=fold_cache))
    base.inner_oof=inner
    original_fit=base.fit_predict
    def fit(model,condition,gt,gx,at,ax,y,hyper,seed,config):
        return cached(config.get('boundary_cache_path'),'outer_fit',
                      (model,condition,gt,gx,at,ax,y,hyper,seed,clean_config(config,condition)),
                      lambda:original_fit(model,condition,gt,gx,at,ax,y,hyper,seed,config))
    base.fit_predict=fit
    original_design=block.design_matrices
    def design(gt,gx,at,ax,groups,config,seed):
        active['root']=config.get('boundary_cache_path')
        return cached(config.get('boundary_cache_path'),'block_design',
                      (gt,gx,at,ax,groups,clean_config(config),seed),
                      lambda:original_design(gt,gx,at,ax,groups,config,seed))
    block.design_matrices=design
