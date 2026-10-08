"""Evaluate ten public task-32 instances sequentially in one simulator process."""
import runpy
from omnigibson.eval.evaluator import BatchedEvaluator, resolve_instance_ids
original_run = BatchedEvaluator.run

def run_ten(self, instances_to_run, **kwargs):
    if self.cfg.task.name != 'wash_a_baseball_cap' or self.num_envs != 1:
        raise ValueError('This runner is restricted to task 32 with one environment')
    results = {}
    ids = resolve_instance_ids(self.cfg.task.name, list(range(10)), mode='public_test')
    for index, instance_id in enumerate(ids):
        print(f'Task 32: starting public index {index}, instance {instance_id}', flush=True)
        results.update(original_run(self, [instance_id], **kwargs))
        print(f'Task 32: completed public index {index}', flush=True)
    return results

BatchedEvaluator.run = run_ten
if __name__ == '__main__':
    runpy.run_module('omnigibson.eval.eval', run_name='__main__')
