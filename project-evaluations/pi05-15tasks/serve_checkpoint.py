"""One shared checkpoint, existing 2026 proprio adapter, strict inference diagnostics."""
import argparse
import importlib.util
import json
import logging
import os
from pathlib import Path
import time

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.5')

import numpy as np
from b1k.training.config import get_config
from b1k.policies.policy_config import create_trained_policy
from b1k.shared.eval_b1k_wrapper import B1KPolicyWrapper
from b1k.models.pi_behavior_config import TASK_NUM_STAGES, TASK_STAGE_OFFSETS
from omnigibson.learning.utils.network_utils import WebsocketPolicyServer

TASKS = (5,7,11,13,14,15,16,17,18,21,24,25,29,30,45)
source = Path(__file__).resolve().parents[1] / 'ckpt3-10k-tasks-31-27/serve_checkpoint.py'
spec = importlib.util.spec_from_file_location('existing_2026_adapter', source)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)

class AuditedAdapter(legacy.Eval2026Adapter):
    def __init__(self, wrapper, emit):
        super().__init__(wrapper, None)
        self.emit = emit
        self.steps = 0

    def reset(self):
        self.emit('rollout_boundary', task_id=self.task_id, finite_action_count=self.steps)
        super().reset()
        self.steps = 0
        self.logged = False

    def act(self, obs):
        ids = np.asarray(obs['task_id']).reshape(-1)
        if ids.size != 1 or int(ids[0]) not in TASKS:
            raise ValueError(f'Unexpected evaluator task ID: {ids}')
        self.task_id = int(ids[0])
        state = np.asarray(obs['robot_r1::proprio'])
        if not np.isfinite(state).all():
            raise ValueError('Non-finite evaluator proprioception')
        action = super().act(obs)
        values = action.detach().cpu().numpy()
        if values.shape not in ((23,), (1,23)) or not np.isfinite(values).all():
            raise ValueError(f'Invalid action shape or non-finite action: {values.shape}')
        self.steps += 1
        if self.steps == 1 or self.steps % 100 == 0:
            self.emit('finite_actions', task_id=self.task_id, count=self.steps,
                      action_min=float(values.min()), action_max=float(values.max()))
        return action


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--audit-log',required=True)
    ap.add_argument('--port',type=int,default=8000)
    args=ap.parse_args()
    audit=Path(args.audit_log).open('x',buffering=1)
    def emit(event,**data):
        audit.write(json.dumps({'time':time.time(),'event':event,**data})+'\n')
    cfg=get_config('pi_behavior_b1k_fast')
    emit('loading',checkpoint=str(Path(args.checkpoint).resolve()),config=cfg.name,
         historical_training_config='UNRECOVERABLE',asset_id=cfg.data.repo_id)
    policy=create_trained_policy(cfg,args.checkpoint,sample_kwargs={'num_steps':20})
    prepare=policy._prepare_inputs
    def checked_prepare(obs):
        prepared=prepare(obs)
        pair=np.asarray(prepared['tokenized_prompt'])
        if pair.shape!=(1,2):
            raise ValueError(f'Invalid model task/stage shape {pair.shape}')
        task,stage=map(int,pair[0])
        if task not in TASKS or not 0<=stage<TASK_NUM_STAGES[task]:
            raise ValueError(f'Invalid task/stage: {task}/{stage}')
        if not np.isfinite(np.asarray(prepared['state'])).all():
            raise ValueError('Non-finite normalized state')
        emit('model_input',task_id=task,embedding_row=task,stage=stage,
             task_stage_row=TASK_STAGE_OFFSETS[task]+stage)
        return prepared
    policy._prepare_inputs=checked_prepare
    infer=policy.infer
    def checked_infer(*a,**kw):
        result=infer(*a,**kw)
        if not np.isfinite(result['actions']).all():
            raise ValueError('Non-finite model output: actions')
        # PiBehavior.sample_actions explicitly masks unavailable stages with -inf.
        task = int(np.asarray(a[0]['tokenized_prompt'])[0])
        logits = np.asarray(result['subtask_logits'])
        count = TASK_NUM_STAGES[task]
        if logits.shape != (15,) or not np.isfinite(logits[:count]).all():
            raise ValueError('Invalid or non-finite valid-stage logits')
        if not np.isneginf(logits[count:]).all():
            raise ValueError('Unexpected unavailable-stage mask')
        return result
    policy.infer=checked_infer
    wrapped=B1KPolicyWrapper(policy,task_id=None)
    adapter=AuditedAdapter(wrapped,emit)
    emit('loaded',checkpoint=str(Path(args.checkpoint).resolve()),tasks=list(TASKS))
    WebsocketPolicyServer(adapter,host='127.0.0.1',port=args.port,
                          metadata={**policy.metadata,'checkpoint':'checkpoint_40000','tasks':list(TASKS)}).serve_forever()

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,force=True)
    main()
