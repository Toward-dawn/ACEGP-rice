"""Proposed audit-only wrappers. Not integrated into A or registered model code."""
from __future__ import annotations
import hashlib,importlib.util,math
from pathlib import Path
import numpy as np

SRC=Path(__file__).resolve().parent/'statistical_reporting.py'
spec=importlib.util.spec_from_file_location('unmodified_statistical_reporting',SRC)
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
PRIMARY_FAMILY=(
 'development_QTL_ACEGP_vs_GW_ACEGP_Fisher_z',
 'validation_QTL_ACEGP_vs_GW_ACEGP_Fisher_z',
)

def registered_holm_family(comparisons,registered_ids,*,family_id,alpha=.05):
    """Keep every prespecified member; incomplete families have no final Holm P."""
    names=tuple(registered_ids)
    if not names or len(names)!=len(set(names)) or not family_id:
        raise ValueError('Unique nonempty prespecified hypothesis IDs and family identity required')
    if set(comparisons)-set(names):
        raise ValueError('An unregistered comparison cannot be inserted after observing P values')
    if not 0<alpha<1:raise ValueError('Alpha must be in(0,1)')
    rows=[]
    for name in names:
        supplied=comparisons.get(name)
        if supplied is None:
            rows.append({'comparison_id':name,'status':'NOT_COMPLETED','raw_p_value':None,'Holm_p_value':None})
            continue
        state=supplied.get('status','NOT_COMPLETED')
        raw=supplied.get('p_value')
        if raw is not None and (not np.isfinite(raw) or not 0<=raw<=1):
            raise ValueError('Invalid raw P value')
        if state=='COMPLETE' and raw is None:raise ValueError('Completed valid comparison requires raw P')
        rows.append({'comparison_id':name,'status':state,'raw_p_value':float(raw) if raw is not None else None,'Holm_p_value':None})
    complete=all(r['status']=='COMPLETE' and r['raw_p_value'] is not None for r in rows)
    if complete:
        adj=s.holm([r['raw_p_value'] for r in rows])
        for r,value in zip(rows,adj):r.update(Holm_p_value=float(value),reject_after_Holm=bool(value<=alpha))
    return {'family_id':family_id,'registered_family_size':len(names),'completed_valid_comparisons':sum(r['status']=='COMPLETE' and r['raw_p_value'] is not None for r in rows),'family_complete':complete,'final_Holm_inference_reportable':complete,'alpha':alpha,'comparisons':rows,'missing_or_unusable_members_are_not_removed':True,'raw_P_values_are_interim_if_family_incomplete':not complete}

def guarded_fisher_comparison(qtl_r,genome_r,train_sizes,test_sizes):
    """Preserve perfect/undefined fold state rather than assigning finite clipped z."""
    q,g=np.asarray(qtl_r,float),np.asarray(genome_r,float)
    tr,te=np.asarray(train_sizes,float),np.asarray(test_sizes,float)
    if q.ndim!=1 or q.shape!=g.shape or tr.shape!=q.shape or te.shape!=q.shape or len(q)<2:
        raise ValueError('Aligned fold correlations and actual split sizes required')
    if np.any(np.abs(q[np.isfinite(q)])>1) or np.any(np.abs(g[np.isfinite(g)])>1):raise ValueError('Correlation outside[-1,1]')
    perfect=np.flatnonzero((np.abs(q)==1)|(np.abs(g)==1))
    if len(perfect):
        return {'status':'UNDEFINED_FISHER_TRANSFORM_PERFECT_CORRELATION','perfect_correlation_fold_indices':perfect.tolist(),'folds_expected':len(q),'p_value':None,'CI':None,'estimand':'paired_outer_fold_Fisher_z_difference','CI_is_raw_Pearson_r_difference':False,'folds_dropped':0,'perfect_r_replaced_by_epsilon':False,'underlying_raw_metrics_should_remain_reported':True}
    result=s.fisher_fold_comparison(q,g,tr,te)
    if result['status']=='ZERO_VARIANCE':
        result.update(p_value=None,CI=None,inference_convention='Corrected t is undefined at zero difference variance; keep effect and state, do not manufacture t/P.')
    return result

def selection_with_tie_audit(y,qtl_prediction,genome_prediction,material_ids,fraction,*,bootstrap_clones=False):
    ids=np.asarray(material_ids,str)
    if not bootstrap_clones and len(np.unique(ids))!=len(ids):raise ValueError('Original material bank IDs must be unique; clone handling is explicit')
    result=s.selection_metrics(y,qtl_prediction,genome_prediction,ids,fraction)
    n,k=result['n'],result['k'];slots=np.arange(n)
    ties={}
    for label,values in [('actual_low_loss_reference',y),('QTL_prediction',qtl_prediction),('GW_prediction',genome_prediction)]:
        values=np.asarray(values,float);order=np.lexsort((slots,ids,values));threshold=float(values[order[k-1]])
        all_equal=values==threshold
        selected_equal=int(all_equal[order[:k]].sum());total=int(all_equal.sum())
        ties[label]={'threshold':threshold,'tie_group_total_count':total,'tie_group_selected_count':selected_equal,'tie_group_unselected_count':total-selected_equal,'tie_spans_selection_boundary':total>selected_equal,'ties_resolved_by':'stable_material_ID_then_current_draw_slot'}
    result.update(boundary_ties=ties,bootstrap_clones_retained=bootstrap_clones)
    return result

def selection_repeat_summary(y,qtl_predictions,genome_predictions,material_ids,*,expected_repeats=20,fractions=(.10,.20)):
    q,g=np.asarray(qtl_predictions,float),np.asarray(genome_predictions,float)
    if q.ndim==1:q=q[:,None]
    if g.ndim==1:g=g[:,None]
    if q.shape!=g.shape or q.ndim!=2 or q.shape[1]!=expected_repeats:
        raise ValueError('Require every registered repeat before formal selection summary')
    rows=[{'repeat':j+1,'proportions':{str(f):selection_with_tie_audit(y,q[:,j],g[:,j],material_ids,f) for f in fractions}} for j in range(expected_repeats)]
    summary={}
    keys=['QTL_selected_actual_mean_loss','GW_selected_actual_mean_loss','GW_minus_QTL_selection_loss','QTL_hit_rate','GW_hit_rate','QTL_minus_GW_hit_rate','QTL_gain_relative_to_population_mean','GW_gain_relative_to_population_mean']
    for f in fractions:
        current={}
        for key in keys:
            values=np.asarray([r['proportions'][str(f)][key] for r in rows],float)
            current[key]={'mean':float(values.mean()),'SD':float(values.std(ddof=1)) if len(values)>1 else None,'min':float(values.min()),'max':float(values.max()),'positive_repeats':int(np.sum(values>0)) if key in ['GW_minus_QTL_selection_loss','QTL_minus_GW_hit_rate'] else None}
        summary[str(f)]=current
    return {'repeat_count':expected_repeats,'repeat_records_with_ties':rows,'summary':summary,'phenotype_units':'percentage_points','repeat_SD_is_not_dependence_corrected_standard_error':True,'ordinary_t_on20selection_scores_performed':False,'next_generation_genetic_gain_measured':False}
