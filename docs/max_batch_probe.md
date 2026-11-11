# CURRENT: MAX_OK=192 (2026-11-11T12:16+08:00 exclusive probe; 160/176/192 OK)

# A800 max-batch probe FINAL
date: 2026-11-11T05:05:40+08:00
GPU: NVIDIA A800 80GB PCIe
cgroup_mem_limit_GiB=120.0
config: fsdp_devices=1 num_flow_samples=5 num_workers=2 save_interval=100000 ema_decay=None
note: nvidia-smi steady often ~15GB; **peak** during JIT/autotune/ckpt is the binding constraint

| batch | result | last_loss | peak_vram_MiB | peak_cgroup_GiB | notes |
|------:|:------:|----------:|--------------:|----------------:|-------|
| 32 | OK | 1.8687 | ~15371 | n/a | baseline |
| 48 | SKIP | - | - | - | concurrent kill; not retested |
| 64 | OK | 2.1736 | 41317 | 71.03 | |
| 96 | OK | 2.1687 | 53971 | 71.26 | XLA remat 52.65→49.86 GiB |
| 128 | OK | 1.9997 | 60557 | 71.52 | XLA remat 62.43→58.77 GiB; min_avail_GiB=889.83 |
| 192 | DEFER | - | - | - | earlier exit=137 under contention/host kill; not re-run this pass |

**MAX_OK=128** (Ilia yaml is 192; 128 is stable here without host kill in exclusive runs)
192 still plausible on 80GB GPU (128 peak ~60GB) but needs exclusive run + watch cgroup during ckpt; not required for ablation start.

# continue probe HI 2026-11-11T11:42:08+08:00 tiers=160,176,192 steps=3 num_workers(dataloader hardcode)=2 save_interval_tmp=100000
cgroup_limit_GiB=120.0
NOTE: ablation ckpt sim_rl_awr_sample_b128/500 preserved untouched

## batch_size=160 2026-11-11T11:42:08+08:00
/dev/md127       50G   29G   22G  58% /root/autodl-tmp
               total        used        free      shared  buff/cache   available
Mem:           1.0Ti        76Gi        76Gi       1.3Gi       854Gi       924Gi
result=OK exit=0 steps_logged=1 peak_vram_MiB=70277 peak_cgroup_GiB=83.26 min_avail_GiB=886.36 last_loss=0.0000

note_160: Progress 4/4 completed; Step0 loss=2.2207; XLA remat could not go below 68.30GiB (from 72.22GiB); nvidia peak 70277MiB
ladder_interrupt_after_160: cleanup did not finish / process gone; manually cleaned sim_rl/3; continuing 176,192
               total        used        free      shared  buff/cache   available
Mem:           1.0Ti        77Gi        76Gi       1.3Gi       853Gi       923Gi
/dev/md127       50G   29G   22G  58% /root/autodl-tmp

# continue probe HI2 2026-11-11T11:51:35+08:00 tiers=176,192 prior_MAX_OK=160

## batch_size=176 2026-11-11T11:51:35+08:00
/dev/md127       50G   29G   22G  58% /root/autodl-tmp
               total        used        free      shared  buff/cache   available
Mem:           1.0Ti        77Gi        75Gi       1.3Gi       853Gi       923Gi
result=OK exit=0 steps_logged=1 progress_lines=0
0 peak_vram_MiB=70865 peak_cgroup_GiB=83.48 min_avail_GiB=885.92 last_loss=1.9772
11:58:02.982 [I] Progress on: 1.00it/4.00it rate:289.4s/it remaining:14:28 elapsed:04:49 postfix:- (76967:tqdm_logging.py:154)
11:58:28.558 [I] Progress on: 2.00it/4.00it rate:134.2s/it remaining:04:28 elapsed:05:14 postfix:- (76967:tqdm_logging.py:154)
11:58:45.194 [I] Progress on: 3.00it/4.00it rate:80.5s/it remaining:01:20 elapsed:05:31 postfix:- (76967:tqdm_logging.py:154)
11:59:10.268 [I] Progress on: 4.00it/4.00it rate:58.6s/it remaining:00:00 elapsed:05:56 postfix:- (76967:tqdm_logging.py:154)
11:59:10.268 [I] Progress on: 4.00it/4.00it rate:58.6s/it remaining:00:00 elapsed:05:56 postfix:- (76967:tqdm_logging.py:154)
2026-11-11 11:56:57.711796: W external/xla/xla/hlo/transforms/simplifiers/hlo_rematerialization.cc:3021] Can't reduce memory use below 46.38GiB (49796213181 bytes) by rematerialization; only reduced to 73.37GiB (78785243286 bytes), down from 77.12GiB (82805220752 bytes) originally
yaml_restored save_interval=1000
note_176: Progress 4/4 OK; Step0 loss=1.9772; remat 77.12→73.37GiB; nvidia peak 70865MiB; peak_cgroup 83.48
ladder_bug: set -e + pipefail + final [ TAG != OK ] false-exit aborted cleanup after OK; fixed in hi3
MAX_OK_so_far=176

# continue probe HI3 2026-11-11T12:05:39+08:00 tiers=192 prior_MAX_OK=176

## batch_size=192 2026-11-11T12:05:39+08:00
/dev/md127       50G   29G   22G  58% /root/autodl-tmp
               total        used        free      shared  buff/cache   available
Mem:           1.0Ti        76Gi        72Gi       1.3Gi       857Gi       923Gi
result=OK exit=0 steps_logged=1 peak_vram_MiB=70079 peak_cgroup_GiB=83.23 min_avail_GiB=885.42 last_loss=2.2918
12:12:43.421 [I] Progress on: 1.00it/4.00it rate:315.1s/it remaining:15:45 elapsed:05:15 postfix:- (83452:tqdm_logging.py:154)
12:13:12.151 [I] Progress on: 2.00it/4.00it rate:146.6s/it remaining:04:53 elapsed:05:43 postfix:- (83452:tqdm_logging.py:154)
12:13:30.488 [I] Progress on: 3.00it/4.00it rate:88.1s/it remaining:01:28 elapsed:06:02 postfix:- (83452:tqdm_logging.py:154)
12:13:54.878 [I] Progress on: 4.00it/4.00it rate:62.9s/it remaining:00:00 elapsed:06:26 postfix:- (83452:tqdm_logging.py:154)
12:13:54.878 [I] Progress on: 4.00it/4.00it rate:62.9s/it remaining:00:00 elapsed:06:26 postfix:- (83452:tqdm_logging.py:154)
2026-11-11 12:11:30.911367: W external/xla/xla/hlo/transforms/simplifiers/hlo_rematerialization.cc:3021] Can't reduce memory use below 46.35GiB (49768726018 bytes) by rematerialization; only reduced to 77.32GiB (83019386080 bytes), down from 82.01GiB (88060822160 bytes) originally
cleanup_begin 2026-11-11T12:14:11+08:00
outputs/checkpoints/pi_modified_bc_rl/sim_rl/3
/dev/md127       50G   29G   22G  58% /root/autodl-tmp
cleanup_done 2026-11-11T12:14:11+08:00
               total        used        free      shared  buff/cache   available
Mem:           1.0Ti        76Gi        72Gi       1.3Gi       857Gi       923Gi
cgroup_after_GiB=38.13
ablation_ckpt500_OK
MAX_OK_so_far=192
MAX_OK=192
DONE_PROBE_HI3 2026-11-11T12:14:11+08:00
yaml_restored save_interval=1000

# A800 max-batch probe FINAL UPDATE 2026-11-11T12:16:34+08:00
GPU: NVIDIA A800 80GB PCIe | cgroup_limit=120GiB | exclusive flock | steps≈4 | num_workers(dataloader)=2
ablation ckpt untouched: sim_rl_awr_sample_b128/500

| batch | result | Step0 loss | peak_vram_MiB | peak_cgroup_GiB | exit_137 |
|------:|:------:|-----------:|--------------:|----------------:|:--------:|
| 32 | OK | 1.8687 | ~15371 | n/a | no |
| 64 | OK | 2.1736 | 41317 | 71.03 | no |
| 96 | OK | 2.1687 | 53971 | 71.26 | no |
| 128 | OK | 1.9997 | 60557 | 71.52 | no |
| 160 | OK | 2.2207 | 70277 | 83.26 | no |
| 176 | OK | 1.9772 | 70865 | 83.48 | no |
| 192 | OK | 2.2918 | 70079 | 83.23 | no (prior defer was contention) |

**MAX_OK=192**
note: nvidia peak saturates ~70–71GB for 160–192 due to XLA remat; higher batch still completes Steps on 80GB.
DONE_FINAL 2026-11-11T12:16:34+08:00
