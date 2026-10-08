# Performance investigation

Observed 2026-10-06 01:00 UTC; current evaluation remains unchanged.

- Hardware reports RTX 4090, 24 GB. Snapshot GPU utilization was 18%; this snapshot alone does not isolate the bottleneck.
- Task 27 completed episodes execute 23,712 steps, generally reach the timeout, and take 92–100 minutes including roughly 5–6 minutes of startup. Rollout throughput is 4.18–4.59 steps/s.
- Task 31 timeout is 12,354 steps; rollout throughput observed 3.79–4.81 steps/s. One successful episode ended at 6,294 steps.
- Runner selected RGBDFullResWrapper: 720x720 head and two 480x480 wrists, with RGB and depth. Original policy consumes RGB resized to 224x224 and discards depth. The standard low-resolution wrapper renders three 224x224 RGB cameras: 6.5x fewer camera pixels, without depth. Pixel ratio is not a measured speedup.
- Every simulation step sends the complete observation over localhost websocket, including unused depth. The policy normally predicts once per 20 actions; observations are still transferred on intervening steps.
- A new simulator process loads the scene for each instance. Across 20 episodes this accounts for roughly 1.8–2 hours of startup.
- About three hours elapsed between initial setup and the first working rollout, including installation and failed starts. These are separate from rollout runtime.
- Both tasks run one environment at a time. Earlier run episode counts, max steps, wrappers, versions and batching are unknown, so comparison with five tasks in 12 hours cannot be normalized yet.
- Live py-spy sampling was unavailable because this container lacks SYS_PTRACE. No measured rendering/inference/video breakdown is claimed.

Before another full evaluation: time 200–500 steps with separate simulator, websocket, inference and video timers; compare the prior observation wrapper if available. Avoid changing wrapper or step caps mid-run, as either can change evaluation outcomes. Reuse the simulator between sequential instances where supported.
