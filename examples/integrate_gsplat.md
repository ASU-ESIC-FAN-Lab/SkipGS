# Integrating SkipGS into gsplat

> **Status: integration notes, not a tested patch.** Unlike the gaussian-splatting
> guide (validated on real training runs), this document was written against
> `nerfstudio-project/gsplat`'s `examples/simple_trainer.py` without running it —
> treat it as a map of what to touch, and check every detail against your version.

The same recipe applies to any gsplat-based trainer: **decide before
`loss.backward()`, gate the backward and *every* optimizer's step, start after the
strategy stops refining.**

## 1. Construct the controller

`start_iter` must be at or after the point where the densification strategy stops
needing gradients for refinement:

- `DefaultStrategy`: use `strategy.refine_stop_iter` (default 15000).
- `MCMCStrategy`: use `strategy.refine_stop_iter` (default 25000) — relocation/adding
  also runs off optimizer state up to that iteration.

```python
from skipgs import SkipController

skip = SkipController(
    start_iter=cfg.strategy.refine_stop_iter,
    threshold=0.0,
    min_bwd_ratio="auto",
)
```

## 2. Gate backward, strategy post-backward, and all optimizer steps

The core of `simple_trainer.py`'s loop looks like:

```python
        loss.backward()
        ...
        self.cfg.strategy.step_post_backward(..., step=step, info=info)   # needs grads
        for optimizer in self.optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for optimizer in self.pose_optimizers + self.app_optimizers + ...:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for scheduler in schedulers:
            scheduler.step()
```

Gate it like this:

```python
        # image_ids: the view indices of this minibatch (see the batching note below)
        view_id = int(image_ids.item()) if image_ids.numel() == 1 else tuple(image_ids.tolist())
        do_skip = skip(view_id, loss.item(), step)

        if not do_skip:
            loss.backward()
            self.cfg.strategy.step_post_backward(..., step=step, info=info)
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in ...:   # pose / appearance / bilateral-grid optimizers too
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        for scheduler in schedulers:
            scheduler.step()        # keep LR schedules on wall-clock iterations
```

Notes on the pieces:

- **`step_post_backward` inside the gate.** After `refine_stop_iter`, `DefaultStrategy`'s
  post-backward is a no-op except for gradient-stat accumulation, but it reads
  `info["means2d"].grad` — on a skipped iteration there are no grads, so it must not run.
  Since SkipGS only skips after `refine_stop_iter`, gating it changes nothing about
  densification.
- **Gate every optimizer.** gsplat holds one optimizer per Gaussian attribute plus
  optional pose/appearance/bilateral-grid optimizers. A skipped iteration must step
  *none* of them (with `set_to_none=True` grads stay `None`, so an un-gated `step()`
  would be a mostly-harmless no-op for Adam — but gating is exact and also skips the
  optimizer-state walk).
- **Schedulers stay un-gated** so the LR decay stays tied to the iteration count.
- **`packed=True`, `sparse_grad=True`, selective/sparse Adam:** all fine — a skipped
  iteration simply performs no backward and no step, so there is no partial state to
  reconcile. If you also enable `step_mode="adaptive_norm"`, feed each iteration's
  visibility mask to `skip.observe_visibility(...)` and pass `skip.window_visibility()`
  (the union over the accumulation window) to the selective optimizer when a step fires.

## 3. Batching (`cfg.batch_size > 1`)

SkipGS's per-view EMA assumes its decision unit is *revisited*. With `batch_size=1`
(gsplat's default) the unit is a view and everything above applies as-is. With random
batches of size B > 1 there are two workable options:

1. **Global EMA (recommended, simplest):** pass a constant id — `skip(0, loss.item(),
   step)`. The policy degenerates to "skip when the batch loss is at/below the global
   running average", which still tracks convergence, just without per-view resolution.
2. **Per-view EMA with separable losses:** if you compute per-view losses before the
   reduction, use `skip.decide_batch(image_ids.tolist(), per_view_losses, step)` and
   backward only the non-skipped subset. More invasive; only worth it if B is small and
   views differ a lot.

Do **not** use `tuple(image_ids)` of a random sampler as the id — random B-subsets
almost never repeat, so the EMA never seeds and nothing is ever skipped.

## 4. Checkpointing

```python
# save (e.g. next to the ckpt dict):
data["skipgs"] = skip.state_dict()
# resume:
skip.load_state_dict(ckpt["skipgs"])
```

## Sanity checks after integration

- With `enabled=False` the run must be bit-identical to your baseline.
- `skip.summary()["skip_ratio"]` should be ~0 before `start_iter + warmup` and typically
  0.2–0.5 afterwards with the auto floor.
- PSNR at the end should match the baseline within noise; if it degrades, raise
  `min_bwd_ratio` (e.g. `0.7`).
