# Integrating SkipGS into Speedy-Splat

> **Status: validated.** Benchmarked on 13 scenes (Mip-NeRF 360 / Deep Blending /
> Tanks&Temples), resumed from 15k checkpoints, skip + snr-accumulation:
> T_post −30~34% at ≤0.06 PSNR vs the paper baselines.

Target: `j-alex-hanson/speedy-splat` `train.py`. Speedy-Splat's speedups
(precise tile intersection, soft/hard pruning during densification) are all
forward/densify-side, so they compose cleanly with SkipGS's backward-side
saving — the two attack different halves of the iteration.

## 1. Construct the controller (once, before the loop)

```python
from skipgs import SkipController

skip = SkipController(
    start_iter=opt.densify_until_iter,   # 15000 in the stock config
    threshold=0.0,
    min_bwd_ratio="auto",
)
```

## 2. Decide before the backward

Identical to the vanilla-3DGS guide — after the loss, before `loss.backward()`:

```python
        # --- SkipGS: one call decides and records ---
        skip_backward = skip(viewpoint_cam.uid, loss.item(), iteration)
        if not skip_backward:
            loss.backward()
```

and gate the optimizer step with `if not skip_backward:`.

## 3. (Optional) accumulation

```python
skip = SkipController(start_iter=opt.densify_until_iter, threshold=0.0,
                      min_bwd_ratio="auto", step_mode="adaptive_norm")

            params = [p for g in gaussians.optimizer.param_groups for p in g["params"]]
            if not skip_backward and skip.after_backward(params, iteration):
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
```

## Notes

- Speedy-Splat's hard/soft pruning schedule ends with densification, before
  `start_iter` — no interaction with skipping.
- Because its Gaussian count is small, iterations are cheap-ish; the relative
  saving (−30%) sits between FastGS (−20%, very cheap iterations) and vanilla /
  taming (−40~50%, expensive iterations). The rule of thumb across every
  trainer we tested: SkipGS's relative saving grows with per-iteration cost.
- Baseline runs: construct with `enabled=False` and keep the integration in place.
