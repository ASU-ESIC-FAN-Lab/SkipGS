# Integrating SkipGS into the original gaussian-splatting

Target: `graphdeco-inria/gaussian-splatting` `train.py` (the vanilla trainer). The change
is ~4 lines. SkipGS only skips **after** densification ends, so it never interferes with
the densification path (which needs `viewspace_point_tensor.grad`).

## 1. Construct the controller (once, before the loop)

Find where the training loop starts (`for iteration in range(first_iter, opt.iterations + 1):`)
and add, just before it:

```python
from skipgs import SkipController
skip = SkipController(
    start_iter=opt.densify_until_iter,   # skipping activates here (e.g. 15000)
    threshold=0.0,
    min_bwd_ratio="auto",
)
```

## 2. Gate the backward + optimizer step

The vanilla loop looks like this:

```python
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # ... progress bar, logging ...
            # Densification
            if iteration < opt.densify_until_iter:
                ...
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
```

Change the `loss.backward()` line and the optimizer step so both are gated by SkipGS:

```python
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # --- SkipGS: one call decides and records ---
        do_skip = skip(viewpoint_cam.uid, loss.item(), iteration)
        if not do_skip:
            loss.backward()

        iter_end.record()

        with torch.no_grad():
            # ... progress bar, logging ...
            # Densification  (unchanged; only runs while iteration < densify_until_iter,
            #                 i.e. before SkipGS ever skips, so grads are always present)
            if iteration < opt.densify_until_iter:
                ...
            # Optimizer step
            if iteration < opt.iterations and not do_skip:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
```

If you need the decision and the bookkeeping separated (e.g. you force a backward on
logging iterations and want to record the true outcome), use the two-call form:
`skip.should_skip(...)` where the decision is made, then exactly one
`skip.record(..., skipped=<what actually happened>)` for that iteration.

## 3. (Optional) adaptive optimizer-step scheduling

To also cut how often the optimizer steps, add `step_mode="adaptive_norm"` to the
constructor and gate the optimizer step on `skip.after_backward(...)`. Grads then
accumulate across iterations until the controller says "step now":

```python
skip = SkipController(start_iter=opt.densify_until_iter, threshold=0.0,
                      min_bwd_ratio="auto", step_mode="adaptive_norm")

params = [p for group in gaussians.optimizer.param_groups for p in group["params"]]

        # ... same as section 2, except the optimizer step becomes:
            if iteration < opt.iterations and not do_skip:
                if skip.after_backward(params, iteration):
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
```

`after_backward` fires once the accumulated gradient carries more signal than noise
(no knob to tune; while gradients are informative it steps almost every backward,
near convergence the windows stretch out). Prefer a fixed target instead? Add
`step_trigger="ref_k"` to step at ~16 normal steps' worth.
`after_backward` always returns True in the default `step_mode="off"`, so you can
leave this line in unconditionally. Leave `step_mode="off"` (default) for the
validated minimal-SkipGS behavior.

With a sparse/selective optimizer (steps only the visible Gaussians), an accumulated
step must cover everything that got gradient during the window — the controller keeps
that union for you:

```python
            if not do_skip:
                skip.observe_visibility(visibility_filter)      # after backward
                if iteration < opt.iterations and skip.after_backward(params, iteration):
                    gaussians.optimizer.step(skip.window_visibility(), radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)
```

## 4. (Optional) stats and checkpointing

```python
    # at the end of training:
    with open(os.path.join(scene.model_path, "skip_stats.txt"), "w") as f:
        for k, v in skip.summary().items():
            f.write(f"{k}={v}\n")

    # alongside gaussians.capture() in the checkpoint branch:
    torch.save(skip.state_dict(), os.path.join(scene.model_path, "skipgs.pth"))
    # on resume (after constructing the controller with the same arguments):
    skip.load_state_dict(torch.load(os.path.join(scene.model_path, "skipgs.pth")))
```

## Notes

- **A/B testing:** construct with `enabled=False` for an exact baseline run with the
  integration code left in place.
- **Sparse-Adam / selective optimizers:** the vanilla trainer uses standard Adam, so a
  plain "no `.step()` this iteration" is correct. If you use a visibility-based sparse
  optimizer that accumulates across iterations, gate its accumulation the same way (skip =
  don't accumulate this view). The extra gradient-accumulation variants (`cycle_step`,
  `grad_scale`, ...) in the research code exist for exactly that case and are intentionally
  **not** part of minimal SkipGS.
- **Checkpoints:** SkipGS holds only tiny Python state (per-view EMAs + counters). Use
  `state_dict()`/`load_state_dict()` (section 4) for exact continuity; resuming without it
  is also fine — the controller re-seeds its EMAs within one pass over the views.
