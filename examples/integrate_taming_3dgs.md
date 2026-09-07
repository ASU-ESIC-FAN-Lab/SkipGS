# Integrating SkipGS into Taming 3DGS

> **Status: validated.** Benchmarked on 13 scenes (Mip-NeRF 360 / Deep Blending /
> Tanks&Temples), big-mode budgets, resumed from 15k checkpoints, skip +
> snr-accumulation. The biggest win of the trainers we tested: taming's
> iterations are expensive, so T_post dropped 42–49% at ≤0.1 PSNR.

Target: `humansensinglab/taming-3dgs` `train.py`. Taming densifies on a
score-based budget until `densify_until_iter`; SkipGS activates exactly there
and never touches the budgeted densification.

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

Same as vanilla — right after the loss is assembled:

```python
        # --- SkipGS: one call decides and records ---
        skip_backward = skip(viewpoint_cam.uid, loss.item(), iteration)
        if not skip_backward:
            loss.backward()
```

and gate the existing optimizer block with `if not skip_backward:` (taming steps
`gaussians.optimizer` and, with `--optimizer_type default`, a separate
`gaussians.shoptimizer` — gate both together).

## 3. (Optional) accumulation, both optimizer types

With `step_mode="adaptive_norm"` an accumulated window is applied as one full
step across both optimizers:

```python
skip = SkipController(start_iter=opt.densify_until_iter, threshold=0.0,
                      min_bwd_ratio="auto", step_mode="adaptive_norm")

            params = [gaussians._xyz, gaussians._features_dc, gaussians._opacity,
                      gaussians._scaling, gaussians._rotation, gaussians._features_rest]
            if not skip_backward and skip.after_backward(params, iteration):
                if opt.optimizer_type == "default":
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
                    gaussians.shoptimizer.step()
                    gaussians.shoptimizer.zero_grad(set_to_none=True)
                elif opt.optimizer_type == "sparse_adam":
                    visible = skip.window_visibility()
                    if visible is None:
                        visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)
```

For `sparse_adam`, also feed each backward's mask so the union covers the window:

```python
        if not skip_backward:
            loss.backward()
            skip.observe_visibility(radii > 0)
```

## Notes

- Taming's own research fork enforced a floor of `max(0.6, …)` instead of the
  package's `max(0.5, …)`; in practice the natural skip rate puts the auto floor
  around 0.7 on these scenes, so the difference never binds.
- The score-based densification and its budget accounting all happen before
  `start_iter` — no interaction with skipping.
- Baseline runs: construct with `enabled=False` and keep the integration in place.
