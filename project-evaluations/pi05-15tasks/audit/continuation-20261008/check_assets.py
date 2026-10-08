"""Read-only runtime checks; does not restore weights, train, or evaluate."""
import ast
import contextlib
import dataclasses
import importlib.util
import io
import json
from pathlib import Path
import numpy as np
import jax
from flax import nnx
from b1k.training.config import get_config
from b1k.transforms import TaskIndexToTaskId
from b1k.models.pi_behavior_config import TASK_NUM_STAGES, TASK_STAGE_OFFSETS
from b1k.shared.eval_b1k_wrapper import B1KPolicyWrapper
from b1k.shared import normalize

root=Path(__file__).resolve().parent.parent
cfg=get_config('pi_behavior_b1k_fast')
report={'scope':'Compatibility with available local code, not proof of training-run provenance','model_config':dataclasses.asdict(cfg.model),'checkpoints':[]}
for cp in sorted((root/'snapshot').glob('checkpoint_*')):
 assets=cp/'assets/IliaLarchenko/behavior_224_rgb'
 stats=normalize.load(assets)
 for name,s in stats.items():
  for field in ['mean','std','q01','q99']:
   assert np.isfinite(getattr(s,field)).all(),(cp,name,field)
 for field in ['per_timestamp_mean','per_timestamp_std','per_timestamp_q01','per_timestamp_q99']:
  x=getattr(stats['actions'],field); assert x.shape==(30,32) and np.isfinite(x).all()
 L=stats['actions'].action_correlation_cholesky
 assert L.shape==(960,960) and np.isfinite(L).all()
 reg=np.linalg.cholesky(cfg.model.correlation_beta*(L@L.T)+(1-cfg.model.correlation_beta)*np.eye(960))
 assert np.isfinite(reg).all()
 p=assets/'fast_tokenizer'
 spec=importlib.util.spec_from_file_location('audited_fast',p/'processing_action_tokenizer.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
 tok=m.UniversalActionProcessor.from_pretrained(str(p),local_files_only=True)
 processor_config=json.loads((p/'processor_config.json').read_text())
 assert processor_config['action_dim']==22 and processor_config['time_horizon']==30 and tok.vocab_size==1024
 loaded_dims={'action_dim':tok.action_dim,'time_horizon':tok.time_horizon}
 x=np.random.default_rng(0).uniform(-0.5,0.5,(1,30,22)).astype(np.float32)
 encoded=tok(x)
 assert all(0<=t<1024 for t in encoded[0])
 captured=io.StringIO()
 with contextlib.redirect_stdout(captured): decoded=tok.decode(encoded)
 assert not captured.getvalue(),captured.getvalue()
 assert decoded.shape==x.shape and np.isfinite(decoded).all()
 err=float(np.max(np.abs(decoded-x))); assert err<0.15,err
 report['checkpoints'].append({'checkpoint':cp.name,'norm_runtime_load':'pass','regularized_cholesky':'pass','tokenizer_runtime_roundtrip':'pass','synthetic_roundtrip_max_abs_error':err,'token_count':len(encoded[0]),'loaded_dimensions':loaded_dims,'dimensions_cached_after_encoding':[tok.called_time_horizon,tok.called_action_dim]})
class Dummy:
 def reset(self):pass
wrapper=B1KPolicyWrapper(Dummy())
report['local_task_stage_checks']=[]
for i in [5,7,11,13,14,15,16,17,18,21,24,25,29,30,45]:
 wrapper._handle_task_change(i)
 assert wrapper.current_stage==0 and not wrapper.prediction_history
 for stage in [0,TASK_NUM_STAGES[i]-1]:
  wrapper.current_stage=stage
  a=wrapper.prepare_batch_for_pi_behavior({})
  b=TaskIndexToTaskId()({'task_index':i,'subtask_state':stage})
  assert a['tokenized_prompt'].tolist()==b['tokenized_prompt'].tolist()==[i,stage]
 wrapper.reset(); assert wrapper.current_stage==0
 report['local_task_stage_checks'].append({'task':i,'row':i,'stages':TASK_NUM_STAGES[i],'stage_offset':TASK_STAGE_OFFSETS[i]})
print('Checking abstract parameter shapes',flush=True)
model=nnx.eval_shape(cfg.model.create,jax.random.key(0))
_,state=nnx.split(model)
flat=state.flat_state()
shapes={tuple(str(x) for x in k):list(v.value.shape) for k,v in flat.items() if hasattr(v.value, "shape")}
report['parameter_shape_comparisons']=[]
for cp in sorted((root/'snapshot').glob('checkpoint_*')):
 meta=json.loads((cp/'params/_METADATA').read_text())['tree_metadata']
 expected={tuple(ast.literal_eval(k))[1:-1]:v['value_metadata']['write_shape'] for k,v in meta.items()}
 missing=sorted(set(expected)-set(shapes));extra=sorted(set(shapes)-set(expected));mismatch=[k for k in expected.keys()&shapes.keys() if expected[k]!=shapes[k]]
 report['parameter_shape_comparisons'].append({'checkpoint':cp.name,'checkpoint_leaves':len(expected),'local_leaves':len(shapes),'missing':missing,'extra':extra,'shape_mismatches':mismatch})
(root/'continuation-20261008/runtime-checks.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
print(json.dumps(report['parameter_shape_comparisons'],indent=2))
assert all(not r['missing'] and not r['extra'] and not r['shape_mismatches'] for r in report['parameter_shape_comparisons'])
