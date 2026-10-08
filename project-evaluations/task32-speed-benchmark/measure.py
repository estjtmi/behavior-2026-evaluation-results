"""Short timing experiment; capped rollouts are NOT submission scores."""
import json,pathlib,runpy,statistics,time
from omnigibson.eval.evaluator import BatchedEvaluator
samples=[]
rollouts=[]
default_episode_steps=None
original_run=BatchedEvaluator.run
def measured_run(self,*args,**kwargs):
    global default_episode_steps
    default_episode_steps=int(self.human_stats["length"]*1.5)
    t=time.perf_counter(); before=len(samples)
    value=original_run(self,*args,**kwargs)
    rollouts.append({"wall_seconds":time.perf_counter()-t,"steps":len(samples)-before})
    return value
BatchedEvaluator.run=measured_run
original=BatchedEvaluator._step_fn
started=time.perf_counter()
def measured(self,*args,**kwargs):
    t=time.perf_counter()
    value=original(self,*args,**kwargs)
    samples.append(time.perf_counter()-t)
    return value
BatchedEvaluator._step_fn=measured
try:
    runpy.run_module('omnigibson.eval.eval',run_name='__main__')
finally:
    warm=samples[40:]
    result={'benchmark_only':True,'task':'wash_a_baseball_cap','wrapper':'DefaultWrapper','default_episode_steps':default_episode_steps,'rollouts':rollouts,'steps_measured':len(samples),'wall_seconds':time.perf_counter()-started,'mean_step_seconds_excluding_first_40':statistics.mean(warm) if warm else None,'note':'Two capped 300-step rollouts reuse one simulator. Timing excludes video writing outside _step_fn; wall_seconds includes startup, video and shutdown. These are not full evaluation scores.'}
    pathlib.Path('/workspace/evaluation-results/task32-speed-benchmark/timing.json').write_text(json.dumps(result,indent=2)+'\n')
