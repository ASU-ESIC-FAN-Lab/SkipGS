# Integrating SkipGS into GaussianSpa

> **Status: validated.** Benchmarked on 13 scenes (Mip-NeRF 360 / Deep Blending /
> Tanks&Temples), resumed from 15k checkpoints, skip + snr-accumulation:
> T_post −8~10% at ≤0.1 PSNR (the skip window is short here — see the note).

Target: GaussianSpa's `train_imp_score.py` / `train_opacity.py` (the two method
variants share the same loop; patch both the same way).

## 1. Pick the right `start_iter` — not `densify_until_iter`

This is the one trainer where the fine-tuning phase does **not** begin when
densification ends. GaussianSpa keeps alternating its "optimizing-sparsifying"
step until `optimizing_spa_stop_iter` (25200 stock, training to 40k); only the
tail after that is plain fine-tuning. Skipping must not touch the alternation:

```python
from skipgs import SkipController

skip = SkipController(
    start_iter=int(opt.optimizing_spa_stop_iter) + 1,   # spa checks use strict >
    threshold=0.0,
    min_bwd_ratio="auto",
)
```

Nothing before `start_iter` is ever skipped, so the spa steps, pruning at the
two prune ratios, and densification all run untouched.

## 2. Decide before the backward

```python
        # --- SkipGS: one call decides and records ---
        skip_backward = skip(viewpoint_cam.uid, loss.item(), iteration)
        if not skip_backward:
            loss.backward()
```

and gate the optimizer step with `if not skip_backward:`.

## 3. (Optional) accumulation

```python
skip = SkipController(start_iter=int(opt.optimizing_spa_stop_iter) + 1,
                      threshold=0.0, min_bwd_ratio="auto",
                      step_mode="adaptive_norm")

            params = [p for g in gaussians.optimizer.param_groups for p in g["params"]]
            if not skip_backward and skip.after_backward(params, iteration):
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
```

## Notes

- **Short window, smaller wins.** SkipGS only acts on 25.2k→40k here (~37% of
  the run), so the relative saving is bounded accordingly — expect single-digit
  percent of total time, not the −40% of trainers whose whole 15k→30k half is
  skippable.
- If you resume from a checkpoint *after* `start_iter + warmup` (e.g. a 30k
  ckpt), the warmup calibration window has already passed and the floor stays
  at its 0.5 default. Resume from before `start_iter` (or carry
  `state_dict()`) for a properly calibrated floor.
- Baseline runs: construct with `enabled=False` and keep the integration in place.
