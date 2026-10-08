# Task 32: wash_a_baseball_cap

Queued after tasks 31 and 27, their uploads, and the short speed benchmark.

Uses the same checkpoint and simulator revision as the previous two tasks. The observation profile is different: official `DefaultWrapper`, three 224x224 RGB cameras plus proprioception, without depth. Public-test indices 0–9, one rollout each, seed 0, default task timeout (1.5x mean demonstration length). No shortened step cap for this full evaluation. Videos remain enabled.

`run_instances.py` calls the original evaluator once per instance and reuses its simulator process. The original evaluator loads/resets each instance and resets policy state. All task logic, success criteria and metrics remain unchanged. Capped benchmark outputs are separate and must not be counted as these full evaluation results.

Results are published alongside the earlier tasks under `evaluations/ckpt3-10k-tasks-31-27/results/wash_a_baseball_cap`; videos are GitHub release assets indexed by `ROLLOUT_VIDEOS.md` in that directory's parent evaluation folder.
