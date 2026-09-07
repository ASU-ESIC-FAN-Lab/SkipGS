# Integrating SkipGS into LightGaussian

> **Status: validated.** Benchmarked on 13 scenes (Mip-NeRF 360 / Deep Blending /
> Tanks&Temples), prune+finetune resumed from completed vanilla 30k runs, skip +
> snr-accumulation: T_post −25~29% (and for LightGaussian T_total = T_post, since
> the whole thing is post-training).

Target: `VITA-Group/LightGaussian` `prune_finetune.py`. LightGaussian is unlike
the other trainers: it starts from a *finished* 3DGS run, prunes ~66% of the
Gaussians at the first iteration, then fine-tunes for 5k iterations. The whole
phase is fine-tuning, so SkipGS is active from iteration one.

## 1. Construct the controller (once, before the loop)

`first_iter` comes from the loaded checkpoint (30000 → the loop starts at 30001):

```python
from skipgs import SkipController

skip = SkipController(
    start_iter=first_iter,     # the first finetune iteration, right at the prune
    threshold=0.0,
    min_bwd_ratio="auto",
)
```

The `warmup` (default 500) does double duty here: it seeds the per-view EMAs on
the *post-prune* loss landscape (losses spike at the prune and recover), so the
skip policy calibrates on what fine-tuning actually looks like.

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
skip = SkipController(start_iter=first_iter, threshold=0.0,
                      min_bwd_ratio="auto", step_mode="adaptive_norm")

            params = [gaussians._xyz, gaussians._features_dc, gaussians._features_rest,
                      gaussians._scaling, gaussians._rotation, gaussians._opacity]
            if not skip_backward and skip.after_backward(params, iteration):
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
```

The prune at the first iteration replaces the parameter tensors *between* a
backward and the next `after_backward` — any grads accumulated up to that point
are simply dropped with the old tensors. The controller doesn't need a reset;
its window just refills from the new tensors' gradients.

## Notes

- **Short recovery phases are quality-sensitive.** The 5k-iter finetune is a
  post-prune *recovery*, where gradients carry real signal. In our runs skip +
  accumulation traded ~0.15 PSNR (M360) for the −29% time; if quality is the
  priority here, prefer plain skip-only (`step_mode="off"`).
- LightGaussian's importance-score pass (`prune_list`) renders under `no_grad`
  and doesn't read the training iteration's grads — no gating needed.
- Baseline runs: construct with `enabled=False` and keep the integration in place.
