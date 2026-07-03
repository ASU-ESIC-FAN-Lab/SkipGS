# Integrating SkipGS into FastGS

> **Status: validated.** This is the integration actually used to benchmark the
> package on FastGS — 13 scenes (Mip-NeRF 360 / Deep Blending / Tanks&Temples),
> resumed from 15k checkpoints, in both skip-only and skip+accumulation modes.

Target: FastGS's `train.py` (the `arnaud-lb/FastGS`-style trainer with
`render_fastgs`, `optimizer_step(iteration)` and a separate `shoptimizer`).
FastGS is interesting because it is *already* a fast trainer with its own sparse
optimizer schedule — SkipGS layers on top of it without touching that schedule.

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

FastGS computes its loss from `l1_loss` + `fast_ssim`; right after that:

```python
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # --- SkipGS: one call decides and records ---
        skip_backward = skip(viewpoint_cam.uid, loss.item(), iteration)

        if not skip_backward:
            loss.backward()
```

## 3. Leave FastGS's optimizer schedule alone (skip-only mode)

This is the part that differs from the vanilla-3DGS guide. FastGS's native
`gaussians.optimizer_step(iteration)` steps different parameter groups on its own
schedule and implicitly accumulates gradients between their turns. **Keep calling it
every iteration, skipped or not**:

```python
            if opt.optimizer_type == "default":
                gaussians.optimizer_step(iteration)          # unchanged, every iteration
            elif opt.optimizer_type == "sparse_adam":
                if not skip_backward:                        # sparse step needs this iter's grads
                    gaussians.optimizer.step(radii > 0, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)
```

A skipped iteration simply contributes zero gradient to whatever the native schedule
does next — no gating needed, no interaction to reason about. (This mirrors how the
validated runs behave; per-dataset results: ~34% of post-densification iterations
skipped, T_post −18~21%, PSNR within ±0.05 of FastGS on M360/DB.)

## 4. (Optional) package-side gradient accumulation

To also space out optimizer steps (`step_mode="adaptive_norm"`), the accumulated
window must be applied as a *full* step, so here you do replace the native schedule
after `start_iter` — the same substitution FastGS research forks make for cycle-based
stepping:

```python
skip = SkipController(start_iter=opt.densify_until_iter, threshold=0.0,
                      min_bwd_ratio="auto", step_mode="adaptive_norm")

            # replaces the optimizer block above, post-start_iter only:
            params = [gaussians._xyz, gaussians._features_dc, gaussians._opacity,
                      gaussians._scaling, gaussians._rotation, gaussians._features_rest]
            if not skip_backward and skip.after_backward(params, iteration):
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                gaussians.shoptimizer.step()                 # FastGS's separate SH optimizer
                gaussians.shoptimizer.zero_grad(set_to_none=True)
```

With sparse-Adam, feed each iteration's mask after the backward and step the union:

```python
            if not skip_backward:
                loss.backward()
                skip.observe_visibility(radii > 0)
            ...
            if not skip_backward and skip.after_backward(params, iteration):
                gaussians.optimizer.step(skip.window_visibility(), radii.shape[0])
                gaussians.optimizer.zero_grad(set_to_none=True)
```

## 5. Stats

```python
    with open(os.path.join(scene.model_path, "skip_stats.txt"), "w") as f:
        for k, v in skip.summary().items():
            f.write(f"{k}={v}\n")
```

## Notes

- FastGS's multiview pruning at 18k/21k/24k/27k (`final_prune_fastgs`) needs no
  gating — it renders its own score passes under `no_grad` and never reads the
  training iteration's `.grad`, so skipping doesn't interact with it.
- Baseline runs: construct with `enabled=False` and keep the integration in place.
- FastGS saves less backward time than vanilla 3DGS in relative terms (its
  iterations are already cheap), so expect T_post around −20% rather than −40%.
