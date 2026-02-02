from __future__ import annotations

import enum
from typing import Optional


class PromptState(enum.Enum):
    """FSM states for single-hand prompting."""

    IDLE = 0
    PRE_CUE = 1
    DWELL = 2


class PromptAPI:
    """Abstract interface for prompt visuals and sounds."""

    def set_plate_background_color(self, plate_idx: int, color_name: str) -> None:
        raise NotImplementedError

    def set_plate_background_text(self, plate_idx: int, text: Optional[str]) -> None:
        raise NotImplementedError

    def clear_plate_background(self, plate_idx: int) -> None:
        raise NotImplementedError

    def play_sound(self, key: str, left_vol: float = 1.0, right_vol: float = 1.0) -> None:
        raise NotImplementedError

    def set_controlled_plate(self, plate_idx: int) -> None:
        raise NotImplementedError

    # Convenience defaults for single-hand prompts ---------------------------------
    def show_precue_single_hand(self, source_idx: int, target_idx: int) -> None:
        # Source (switch-from) plate gets the new "switch_from" color
        self.set_plate_background_color(source_idx, "switch_from")
        # Target (switch-to) plate keeps the existing alert color
        self.set_plate_background_color(target_idx, "single_alert")
        self.set_plate_background_text(target_idx, None)


    def show_commit_single_hand(self, source_idx: int, target_idx: int) -> None:
        self.clear_plate_background(source_idx)
        self.set_plate_background_color(target_idx, "single_normal")
        self.set_plate_background_text(target_idx, None)


class NullPromptAPI(PromptAPI):
    """No-op prompt API for headless runs."""

    def set_plate_background_color(self, plate_idx: int, color_name: str) -> None:
        return None

    def set_plate_background_text(self, plate_idx: int, text: Optional[str]) -> None:
        return None

    def clear_plate_background(self, plate_idx: int) -> None:
        return None

    def play_sound(self, key: str, left_vol: float = 1.0, right_vol: float = 1.0) -> None:
        return None

    def set_controlled_plate(self, plate_idx: int) -> None:
        return None


class SingleHandPromptController:
    """Serialized single-hand prompt controller with fixed pre-cue/dwell timings."""

    def __init__(
        self,
        core,
        api: PromptAPI,
        pre_cue_duration: float = 0.3,
        dwell_duration: float = 0.0,
        metrics_manager=None,
        cognitive_agent=None,
    ):
        self.core = core
        self.api = api
        self.pre_cue_duration = pre_cue_duration
        self.dwell_duration = dwell_duration
        self.metrics = metrics_manager
        self.cognitive_agent = cognitive_agent

        self.state: PromptState = PromptState.IDLE
        self.precue_start_time: Optional[float] = None
        self.commit_time: Optional[float] = None
        self.pending_target: Optional[int] = None
        self.dwell_until: Optional[float] = None

    # --------------------------------------------------------------------- API
    def request_switch(self, target_idx: int, t_now: float) -> bool:
        """Start a pre-cue to switch to target_idx (internal index)."""
        source_idx = self.core.controlled_plate

        if target_idx == source_idx:
            return False
        if self.state != PromptState.IDLE:
            return False

        self.pending_target = target_idx
        self.precue_start_time = t_now
        self.commit_time = t_now + self.pre_cue_duration
        self.state = PromptState.PRE_CUE

        if self.metrics is not None:
            precue_ms = int(round(t_now * 1000))
            self.metrics.on_switch_precue(
                core=self.core,
                precue_start_ms=precue_ms,
                source_idx=source_idx,
                target_idx=target_idx,
            )

        self.api.play_sound("single_switch")
        self._apply_precue_visuals(source_idx, target_idx)
        return True

    def update(self, t_now: float) -> None:
        if self.state == PromptState.PRE_CUE:
            if t_now >= (self.commit_time or 0.0):
                self._commit_switch(t_now)
        elif self.state == PromptState.DWELL:
            if t_now >= (self.dwell_until or 0.0):
                self._transition_to_idle()

    def abort_current_precue(self) -> None:
        if self.state == PromptState.PRE_CUE and self.metrics is not None:
            self.metrics.mark_last_switch_aborted()
        self._transition_to_idle()

    # ------------------------------------------------------------------ internals
    def _apply_precue_visuals(self, source_idx: int, target_idx: int) -> None:
        self.api.show_precue_single_hand(source_idx, target_idx)

    def _apply_commit_visuals(self, source_idx: int, target_idx: int) -> None:
        self.api.show_commit_single_hand(source_idx, target_idx)

    def _commit_switch(self, t_now: float) -> None:
        source_internal = self.core.controlled_plate
        target_internal = self.pending_target
        if target_internal is None:
            self._transition_to_idle()
            return

        self.core.commit_switch(target_internal, t_now)
        self.api.set_controlled_plate(target_internal)

        source_logical = self.core.internal_to_logical[source_internal]
        target_logical = self.core.internal_to_logical[target_internal]

        if self.cognitive_agent is not None:
            self.cognitive_agent.on_switch_commit(source_logical, target_logical, t_now)

        if self.metrics is not None:
            commit_ms = int(round(t_now * 1000))
            self.metrics.on_switch_commit_for_last(core=self.core, commit_ms=commit_ms)

        self._apply_commit_visuals(source_internal, target_internal)

        if self.dwell_duration > 0.0:
            self.state = PromptState.DWELL
            self.dwell_until = t_now + self.dwell_duration
        else:
            self._transition_to_idle()

    def _transition_to_idle(self) -> None:
        self.state = PromptState.IDLE
        self.pending_target = None
        self.precue_start_time = None
        self.commit_time = None
        self.dwell_until = None
