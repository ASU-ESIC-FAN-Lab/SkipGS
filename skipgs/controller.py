from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, Iterable, List, Optional, Tuple, Union

__all__ = ["SkipController"]

_EPS = 1e-8
_STATE_VERSION = 1


def _grad_norm(params) -> float:
    """||concatenated .grad|| over params; works on torch tensors and numpy."""
    sq = 0.0
    for p in params:
        g = getattr(p, "grad", None)
        if g is None:
            continue
        if hasattr(g, "detach"):
            g = g.detach()
        sq += float((g * g).sum())
    return sq ** 0.5


@dataclass
class SkipController:
    """Per-view backward-skip policy for the 3DGS fine-tuning phase.

    Args:
        start_iter: When skipping becomes active. Use your densify_until_iter
            (e.g. 15000); nothing is ever skipped before it.
        threshold: Skip when ``loss <= ema * (1 + threshold)``. 0.0 (default)
            = skip at/below the running average; raise it to skip more.
        ema_decay: Per-view loss EMA decay (default 0.95).
        warmup: Iterations after start_iter with no skipping, to seed the EMAs
            and calibrate the floor (default 500).
        min_bwd_ratio: At least this fraction of iterations must run a real
            backward. "auto" (default) calibrates it from the skip rate seen
            during warmup: ``max(0.5, 1 - 0.5 * natural_skip_rate)``. Pass a
            float to set it yourself, 0.0 to turn it off.
        enabled: Set False to make the controller inert (never skips, records
            nothing) -- handy for baseline runs. Can be toggled at runtime.

        step_trigger: = parameter-free trigger, step when
            ||sum g||^2 >= 2n * EMA(single-step ||g||^2), i.e. accumulated
            signal energy has caught up with the noise floor. "ref_k" = step
            at a fixed target of ~16 single-step gradients' worth.

    Everything else runs at the configuration validated in our experiments
    (floor enforced after 100 samples, adaptive windows capped at 64
    backwards, reference EMA decay 0.9). Those live as class attributes below
    if an experiment truly needs to move them.
    """

    start_iter: int
    threshold: float = 0.0
    ema_decay: float = 0.95
    warmup: int = 500
    min_bwd_ratio: Union[float, str] = "auto"
    enabled: bool = True
    step_mode: str = "off"
    step_trigger: str = "snr"

    # fixed at the values used in every validated run (not constructor args)
    floor_min_samples = 100    # floor kicks in after this many recorded iters
    step_ref_k = 16.0          # "ref_k" trigger target, in single-step grads
    step_max_accum = 64        # hard cap on backwards per accumulation window
    step_ref_decay = 0.9       # EMA decay of the single-step grad-norm sample

    # internal state, not constructor args
    view_loss_ema: Dict[Hashable, float] = field(default_factory=dict, init=False)
    stats: Dict[str, int] = field(default_factory=lambda: {"bwd": 0, "skip": 0}, init=False)

    def __post_init__(self) -> None:
        self._auto = isinstance(self.min_bwd_ratio, str) and self.min_bwd_ratio == "auto"
        # 0.5 is the pre-calibration floor, same as the reference impl.
        self._min_bwd_ratio: float = 0.5 if self._auto else float(self.min_bwd_ratio)
        self._calibrated: bool = not self._auto
        self._warmup_would_skip: int = 0
        self._warmup_would_total: int = 0
        # catches should_skip calls whose record never came
        self._pending: Optional[Tuple[Hashable, int]] = None
        self._warned_unpaired: bool = False
        # adaptive optimizer-step state
        if self.step_mode not in ("off", "adaptive_norm"):
            raise ValueError(f"step_mode must be 'off' or 'adaptive_norm', got {self.step_mode!r}")
        if self.step_trigger not in ("snr", "ref_k"):
            raise ValueError(f"step_trigger must be 'snr' or 'ref_k', got {self.step_trigger!r}")
        self._accum_count: int = 0             # backwards since the last step
        self._last_accum_norm: float = 0.0     # ||sum g_i|| reported most recently
        self._single_scale_ema: Optional[float] = None  # EMA of single-iteration grad norm
        self._single_sq_ema: Optional[float] = None     # EMA of single-iteration ||g||^2 (snr noise floor)
        self._reference: float = 0.0           # step_ref_k * single_scale_ema (ref_k trigger)
        self._step_count: int = 0              # optimizer steps taken post start_iter
        self._vis_union: Any = None            # OR of visibility masks this window
        self._vis_stale: bool = False          # window done; clear union on next observe

    # ------------------------------------------------------------------ #
    # One-call API (decide + record)
    # ------------------------------------------------------------------ #
    def decide(self, view_id: Hashable, loss: float, iteration: int) -> bool:
        """Decide and record in one call. True = skip the backward.

        Same decisions as should_skip() followed by record() -- state only
        cares about their order, not where in the iteration they run. If you
        sometimes override the decision, use the two-call form instead; this
        one assumes you honor what it returns.
        """
        skipped = self.should_skip(view_id, loss, iteration)
        self.record(view_id, loss, skipped=skipped, iteration=iteration)
        return skipped

    __call__ = decide

    def decide_batch(
        self, view_ids: Iterable[Hashable], losses: Iterable[float], iteration: int
    ) -> List[bool]:
        """decide() over (view_id, loss) pairs; one decision each.

        For trainers that render several views per iteration with separable
        per-view losses. If the batch has one joint loss, call decide() once
        instead (a stable batch id, or a constant id for a global EMA).
        """
        return [self.decide(v, l, iteration) for v, l in zip(view_ids, losses)]

    def after_backward(self, params, iteration: int) -> bool:
        """Call once right after loss.backward(); True = run optimizer.step()
        + zero_grad() now.

        With step_mode="off" (default) this is always True, so the same loop
        works whether adaptive stepping is on or not -- in "off" mode it
        doesn't even look at params. In "adaptive_norm" mode it measures the
        accumulated grad norm off ``params`` (anything iterable with .grad)
        and feeds the trigger. Like decide(), it assumes you honor the answer;
        drive observe_grad() / should_step() / reset_accum() yourself if you
        need a custom norm.
        """
        if self.step_mode == "off":
            self._vis_stale = True  # each window is one backward
            return True
        self.observe_grad(_grad_norm(params))
        if not self.should_step(iteration):
            return False
        self.reset_accum()
        return True

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #
    def active(self, iteration: int) -> bool:
        """True once skipping is eligible (iteration >= start_iter)."""
        return self.enabled and iteration >= self.start_iter

    def should_skip(self, view_id: Hashable, loss: float, iteration: int) -> bool:
        """Skip backward for this view this iteration? Pure query; follow up
        with record() (or just use decide())."""
        if not self.active(iteration):
            return False

        # a should_skip whose record never came = EMAs silently stop updating
        if self._pending is not None and not self._warned_unpaired:
            warnings.warn(
                "skipgs: should_skip() called again before record() -- the previous "
                f"decision {self._pending} was never recorded, so EMAs and budget-floor "
                "statistics are not being updated. Pair every should_skip() with a "
                "record(), or use the one-call decide() API.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned_unpaired = True
        self._pending = (view_id, iteration)

        iters_since = iteration - self.start_iter
        if iters_since < self.warmup:
            return False
        if view_id not in self.view_loss_ema:
            return False

        surprise = loss / (self.view_loss_ema[view_id] + _EPS)
        skip = surprise <= (1.0 + self.threshold)

        # budget floor: don't let the cumulative bwd ratio drop below the floor
        if skip and self._min_bwd_ratio > 0:
            total = self.stats["bwd"] + self.stats["skip"]
            if total > self.floor_min_samples and (
                self.stats["bwd"] / total
            ) < self._min_bwd_ratio:
                skip = False

        return skip

    # ------------------------------------------------------------------ #
    # Update
    # ------------------------------------------------------------------ #
    def record(self, view_id: Hashable, loss: float, skipped: bool, iteration: int) -> None:
        """Record what happened this iteration; updates all controller state."""
        if not self.active(iteration):
            return
        self._pending = None
        iters_since = iteration - self.start_iter

        # during warmup, count how often we *would* have skipped;
        # that rate sets the auto floor once warmup ends
        if self._auto and not self._calibrated:
            if iters_since < self.warmup and view_id in self.view_loss_ema:
                surprise = loss / (self.view_loss_ema[view_id] + _EPS)
                if surprise <= (1.0 + self.threshold):
                    self._warmup_would_skip += 1
                self._warmup_would_total += 1
            elif iters_since >= self.warmup and self._warmup_would_total > 0:
                rate = self._warmup_would_skip / self._warmup_would_total
                self._min_bwd_ratio = max(0.5, 1.0 - 0.5 * rate)
                self._calibrated = True

        # EMA updates every visit, skipped or not
        if view_id in self.view_loss_ema:
            self.view_loss_ema[view_id] = (
                self.ema_decay * self.view_loss_ema[view_id] + (1.0 - self.ema_decay) * loss
            )
        else:
            self.view_loss_ema[view_id] = loss

        if skipped:
            self.stats["skip"] += 1
        else:
            self.stats["bwd"] += 1

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #
    def state_dict(self) -> Dict[str, Any]:
        """All mutable state, picklable / torch.save-able.

        Constructor args are config, not state -- rebuild with the same args,
        then load_state_dict(). (JSON turns int view ids into strings; use
        pickle or torch.save if your ids are ints.)
        """
        return {
            "state_version": _STATE_VERSION,
            "view_loss_ema": dict(self.view_loss_ema),
            "stats": dict(self.stats),
            "min_bwd_ratio_effective": self._min_bwd_ratio,
            "calibrated": self._calibrated,
            "warmup_would_skip": self._warmup_would_skip,
            "warmup_would_total": self._warmup_would_total,
            "accum_count": self._accum_count,
            "last_accum_norm": self._last_accum_norm,
            "single_scale_ema": self._single_scale_ema,
            "single_sq_ema": self._single_sq_ema,
            "reference": self._reference,
            "step_count": self._step_count,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore what state_dict() saved (config is not restored)."""
        version = state.get("state_version")
        if version != _STATE_VERSION:
            raise ValueError(f"skipgs: unsupported state_version {version!r} (expected {_STATE_VERSION})")
        self.view_loss_ema = dict(state["view_loss_ema"])
        self.stats = dict(state["stats"])
        self._min_bwd_ratio = float(state["min_bwd_ratio_effective"])
        self._calibrated = bool(state["calibrated"])
        self._warmup_would_skip = int(state["warmup_would_skip"])
        self._warmup_would_total = int(state["warmup_would_total"])
        self._accum_count = int(state["accum_count"])
        self._last_accum_norm = float(state["last_accum_norm"])
        sse = state["single_scale_ema"]
        self._single_scale_ema = None if sse is None else float(sse)
        ssq = state.get("single_sq_ema")
        self._single_sq_ema = None if ssq is None else float(ssq)
        self._reference = float(state["reference"])
        self._step_count = int(state["step_count"])
        self._pending = None

    # ------------------------------------------------------------------ #
    # Adaptive optimizer-step scheduling (optional; step_mode="adaptive_norm")
    # ------------------------------------------------------------------ #
    def observe_grad(self, accum_grad_norm: float) -> None:
        """Report ||accumulated .grad|| after each backward, before the
        (possible) step. The first report of each window is a clean
        single-iteration sample and updates the trigger reference."""
        if self.step_mode == "off":
            return
        if self._accum_count == 0:
            sq = accum_grad_norm * accum_grad_norm
            if self._single_scale_ema is None:
                self._single_scale_ema = accum_grad_norm
                self._single_sq_ema = sq
            else:
                d = self.step_ref_decay
                self._single_scale_ema = d * self._single_scale_ema + (1.0 - d) * accum_grad_norm
                self._single_sq_ema = d * self._single_sq_ema + (1.0 - d) * sq
            self._reference = self.step_ref_k * self._single_scale_ema
        self._accum_count += 1
        self._last_accum_norm = accum_grad_norm

    def should_step(self, iteration: int) -> bool:
        """Run optimizer.step() this iteration?

        "off": always True. "adaptive_norm": always True before start_iter and
        during warmup (calibration), then once the trigger fires -- "snr":
        accumulated energy clears twice the noise floor; "ref_k": accumulated
        norm reaches the fixed reference -- or step_max_accum is hit.
        """
        if self.step_mode == "off":
            return True
        if not self.active(iteration) or iteration - self.start_iter < self.warmup:
            return True  # calibration: step every backward, as the trainer would
        if self._accum_count <= 0:
            return False  # nothing accumulated yet this window
        if self.step_trigger == "snr":
            fired = (
                self._single_sq_ema is not None
                and self._last_accum_norm ** 2 >= 2.0 * self._accum_count * self._single_sq_ema
            )
        else:  # "ref_k"
            fired = self._reference > 0 and self._last_accum_norm >= self._reference
        return fired or self._accum_count >= self.step_max_accum

    def reset_accum(self) -> None:
        """Call right after an optimizer step + zero_grad."""
        if self._accum_count > 0:
            self._step_count += 1
        self._accum_count = 0
        self._last_accum_norm = 0.0
        # keep the visibility union readable until the next observe_visibility
        self._vis_stale = True

    # ------------------------------------------------------------------ #
    # Visibility union (for sparse / selective optimizers)
    # ------------------------------------------------------------------ #
    def observe_visibility(self, mask) -> None:
        """OR this iteration's visibility mask into the accumulation window.

        For sparse/selective optimizers (FastGS, taming-3dgs, gsplat's
        selective Adam) that only step the visible Gaussians: an accumulated
        step must cover every Gaussian that received gradient during the
        window, i.e. the union of the masks. Call this once per non-skipped
        iteration, right after backward and *before* after_backward(); when a
        step fires, hand window_visibility() to the optimizer::

            loss.backward()
            skip.observe_visibility(visibility_filter)
            if skip.after_backward(params, it):
                optimizer.step(skip.window_visibility(), N)
                optimizer.zero_grad(set_to_none=True)

        Works with anything supporting ``|`` (torch bool tensors, numpy
        arrays, Python sets). In step_mode="off" every window is a single
        backward, so window_visibility() is just the mask you passed -- the
        same code runs in both modes. The union is transient window state and
        is not checkpointed (same as the accumulated grads themselves).
        """
        if self._vis_stale:
            self._vis_union = None
            self._vis_stale = False
        self._vis_union = mask if self._vis_union is None else (self._vis_union | mask)

    def window_visibility(self):
        """Union of the masks seen this window (valid right after
        after_backward() returns True). None if never observed."""
        return self._vis_union

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def min_bwd_ratio_effective(self) -> float:
        """The floor actually being enforced (post-calibration if auto)."""
        return self._min_bwd_ratio

    @property
    def step_reference(self) -> float:
        """Current step trigger: step_ref_k * EMA(single-iteration norm)."""
        return self._reference

    def summary(self) -> Dict[str, float]:
        """Skip statistics for logging / skip_stats.txt. Same six keys in every mode."""
        total = self.stats["bwd"] + self.stats["skip"]
        bwd_ratio = self.stats["bwd"] / total if total else 0.0
        return {
            "total_bwd": self.stats["bwd"],
            "total_skip": self.stats["skip"],
            "total_iter_post": total,
            "bwd_ratio": bwd_ratio,
            "skip_ratio": 1.0 - bwd_ratio,
            "min_bwd_ratio_final": self._min_bwd_ratio,
        }
