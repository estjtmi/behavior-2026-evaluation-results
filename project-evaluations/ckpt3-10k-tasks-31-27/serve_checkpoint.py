"""Serve the original policy with explicit 2026 checkpoint assets and observation layout."""
import argparse
import dataclasses
import logging
import os

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.5')
os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')

import numpy as np
from b1k.training import config
from b1k.policies.policy_config import create_trained_policy
from b1k.shared.eval_b1k_wrapper import B1KPolicyWrapper
from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES
from omnigibson.learning.utils.network_utils import WebsocketPolicyServer

# Exact order in the official 2026 eval/r1pro.yaml. Only public proprioception
# features are copied into the legacy layout; the model reads joint positions,
# gripper widths and base velocity from that layout.
FEATURES = [('base_qvel',3), ('arm_left_qpos',7), ('arm_left_qvel',7),
            ('eef_left_pos',3), ('eef_left_quat',4), ('gripper_left_qpos',2),
            ('gripper_left_qvel',2), ('arm_right_qpos',7), ('arm_right_qvel',7),
            ('eef_right_pos',3), ('eef_right_quat',4), ('gripper_right_qpos',2),
            ('gripper_right_qvel',2), ('trunk_qpos',4), ('trunk_qvel',4)]

class Eval2026Adapter:
    def __init__(self, policy, task_id):
        self.policy = policy
        self.task_id = task_id
        self.logged = False

    def reset(self):
        self.policy.reset()

    def act(self, obs):
        obs = dict(obs)
        state = np.asarray(obs['robot_r1::proprio'])
        batched = state.ndim == 2 and state.shape[0] == 1
        if batched:
            obs = {k: np.asarray(v)[0] if isinstance(v, np.ndarray) and v.ndim > 0 and v.shape[0] == 1 else v for k, v in obs.items()}
            state = np.asarray(obs['robot_r1::proprio'])
        if state.shape != (sum(n for _,n in FEATURES),):
            raise ValueError(f'Expected single-env 2026 proprio, got {state.shape}')
        expanded = np.zeros(256, dtype=state.dtype)
        offset = 0
        for name, width in FEATURES:
            target = PROPRIOCEPTION_INDICES['R1Pro'][name]
            if target.stop - target.start != width:
                raise ValueError(f'Proprio width mismatch for {name}')
            expanded[target] = state[offset:offset+width]
            offset += width
        obs['robot_r1::proprio'] = expanded
        if not self.logged:
            logging.info('2026 observation shapes: %s', {k:np.shape(v) for k,v in obs.items()})
            logging.info('Evaluator task id=%s; checkpoint task id=%s', obs.get('task_id'), self.task_id)
            self.logged = True
        obs['task_id'] = np.array([self.task_id], dtype=np.int64)
        action = self.policy.act(obs)
        return action.reshape(1, -1) if batched else action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--task-id', type=int, choices=[31,27,32], required=True)
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    original = config.get_config('pi_behavior_b1k_fast')
    cfg = dataclasses.replace(original, data=dataclasses.replace(original.data,
        assets=config.AssetsConfig(asset_id='behavior-1k/2026-challenge-demos')))
    policy = create_trained_policy(cfg, args.checkpoint, sample_kwargs={'num_steps':20})
    wrapped = B1KPolicyWrapper(policy, text_prompt='PI_BEHAVIOR model (task-conditioned)', task_id=args.task_id)
    server = WebsocketPolicyServer(Eval2026Adapter(wrapped,args.task_id),host='127.0.0.1',port=args.port,metadata=policy.metadata)
    server.serve_forever()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,force=True)
    main()
