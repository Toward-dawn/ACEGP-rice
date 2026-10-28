"""Planned same-panel feature-policy control, never a replacement main method.

Use the unchanged uniform holdout orchestration, with the literal historical
5+3+Takai preparation, nested weights and block RNG policy installed in an
isolated module instance. This includes historical SVD/block-PCA conventions;
the resulting background is reported separately, not asserted equal to the
uniform policy. Historical outer seeds/results are not reproduced or replaced.
"""
from pathlib import Path
import ast, hashlib, importlib.util, sys

ROOT=Path(__file__).resolve().parent.parent
SNAPSHOT=ROOT/'legacy_model_snapshot'
POLICY='historical_5plus3plusTakai_under_registered_uniform_outer_workflow_v1'

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 module=importlib.util.module_from_spec(spec);sys.modules[name]=module
 spec.loader.exec_module(module)
 return module

def legacy_reference(tag='reference'):
 """Literal historical preparation/grid implementation for fidelity checks."""
 if str(SNAPSHOT) not in sys.path:sys.path.insert(0,str(SNAPSHOT))
 b=load('tiered_'+tag+'_base',SNAPSHOT/'run_scheme2_tuned.py')
 block=load('tiered_'+tag+'_block',SNAPSHOT/'run_qtl_block_ard.py');block.base=b
 import runtime_support
 runtime_support.install(b,block)
 return b,block

def build():
 if str(SNAPSHOT) not in sys.path:sys.path.insert(0,str(SNAPSHOT))
 workflow=load('tiered_control_holdout_workflow',ROOT/'src/acegp_uniform.py')
 # Restore exactly the original preparation definitions in the SAME base
 # namespace, so every call (including calls from inner_oof) sees its hooks.
 source=SNAPSHOT/'run_scheme2_tuned.py'
 tree=ast.parse(source.read_text(encoding='utf-8-sig'))
 wanted={'pca_scores','build_feature_sets','prepare_fold_data'}
 nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
 assert {n.name for n in nodes}==wanted
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),workflow.base.__dict__)
 block=load('tiered_adapter_block',SNAPSHOT/'run_qtl_block_ard.py');block.base=workflow.base
 import runtime_support
 runtime_support.install(workflow.base,block)
 workflow.prepare=workflow.base.prepare_fold_data
 workflow.block_design=block.design_matrices
 workflow.block_hyper_grid=block.hyper_grid
 workflow.block_predict=block.fit_predict
 workflow.VERSION=POLICY
 workflow.CODE_HASH=hashlib.sha256(workflow.CODE_HASH.encode()+Path(__file__).read_bytes()).hexdigest()
 return workflow,block
