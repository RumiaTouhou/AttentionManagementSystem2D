from __future__ import annotations

import csv
import math
import os
import uuid
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import Dict, List, Optional


class MetricLevel(IntEnum):
    NONE = 0
    CORE = 1
    XAI = 2


@dataclass
class MetricsConfig:
    level: MetricLevel = MetricLevel.CORE
    log_to_csv: bool = False
    episodes_csv_path: str = os.path.join(os.path.dirname(__file__), "PlayLogs", "episodes.csv")
    switches_csv_path: str = os.path.join(os.path.dirname(__file__), "PlayLogs", "switches.csv")

def _ensure_csv_header(path: str, fieldnames: List[str]) -> None:
    """
    Ensure an existing CSV file has exactly the given header.
    If it has an older header (e.g., missing new columns), rewrite it
    preserving all rows and filling missing fields with "".
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return

    tmp_path = path + ".tmp"
    try:
        with open(path, "r", newline="", encoding="utf-8") as fr:
            reader = csv.reader(fr)
            header = next(reader, None)

        if header == fieldnames:
            return

        with open(path, "r", newline="", encoding="utf-8") as fr, open(
            tmp_path, "w", newline="", encoding="utf-8"
        ) as fw:
            dr = csv.DictReader(fr)
            dw = csv.DictWriter(fw, fieldnames=fieldnames)
            dw.writeheader()
            for row in dr:
                out = {k: row.get(k, "") for k in fieldnames}
                dw.writerow(out)

        os.replace(tmp_path, path)

    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return


@dataclass
class SwitchMetrics:
    participant_id: Optional[int]
    episode_id: int
    episode_uid: str
    switch_id: int

    precue_start_ms: int
    commit_ms: int
    dwell_time_ms: int
    dt_since_last_switch_ms: int

    source_plate_idx: int
    target_plate_idx: int

    reaction_time_ms: Optional[int] = None

    current_drag_norm_precue: float = 0.0

    # State @ precue
    source_dist_precue: float = 0.0
    source_speed_precue: float = 0.0
    source_outvel_precue: float = 0.0
    source_tilt_mag_precue: float = 0.0
    source_tilt_out_precue: float = 0.0

    target_dist_precue: float = 0.0
    target_speed_precue: float = 0.0
    target_outvel_precue: float = 0.0
    target_tilt_mag_precue: float = 0.0
    target_tilt_out_precue: float = 0.0

    # State @ commit
    source_dist_commit: float = 0.0
    source_speed_commit: float = 0.0
    source_outvel_commit: float = 0.0
    source_tilt_mag_commit: float = 0.0
    source_tilt_out_commit: float = 0.0

    target_dist_commit: float = 0.0
    target_speed_commit: float = 0.0
    target_outvel_commit: float = 0.0
    target_tilt_mag_commit: float = 0.0
    target_tilt_out_commit: float = 0.0

    # XAI
    priority_lag: Optional[int] = None
    target_R_commit: Optional[float] = None
    target_R_plus1s: Optional[float] = None
    source_R_commit: Optional[float] = None
    source_R_plus1s: Optional[float] = None
    HMR: Optional[float] = None
    OC: Optional[float] = None
    hmr_available: bool = False
    oc_available: bool = False

    stabilization_time_ms: Optional[int] = None
    stab_available: bool = False

    is_flap: bool = False
    layout_mapping_id: Optional[str] = None
    aborted: bool = False


@dataclass
class EpisodeMetrics:
    participant_id: Optional[int]
    condition: str
    N_plates: int
    episode_id: int
    episode_uid: str

    block_order: Optional[str] = None
    trial_index: Optional[int] = None
    layout_mapping_id: Optional[str] = None

    duration_sec: float = 0.0
    avg_plate_health: float = 0.0
    num_switches: int = 0
    failure_plate_idx: Optional[int] = None
    avg_tilt_mag: float = 0.0
    avg_current_drag_norm: float = 0.0
    max_neglect_time_sec: float = 0.0
    danger_time_frac: float = 0.0

    reset_profile_used: Optional[str] = None
    init_mean_distance_norm: float = 0.0
    init_max_distance_norm: float = 0.0
    init_mean_speed_norm: float = 0.0
    init_max_speed_norm: float = 0.0
    init_mean_tilt_mag_norm: float = 0.0
    init_max_tilt_mag_norm: float = 0.0
    init_drag: float = 0.0
    init_drag_decrease_amount: float = 0.0
    init_anti_stall_speed_threshold: float = 0.0

    _health_integral: float = 0.0
    _tilt_mag_integral: float = 0.0
    _drag_norm_integral: float = 0.0
    _neglect_timers: List[float] = field(default_factory=list)
    _danger_time_total: float = 0.0

    switches: List[SwitchMetrics] = field(default_factory=list)


@dataclass
class ReactionWatcher:
    switch_id: int
    anchor_ms: int
    active: bool = True
    frames_above_thresh: int = 0


@dataclass
class HmrOcWatcher:
    switch_id: int
    sample_time_ms: int
    active: bool = True


@dataclass
class StabilizationWatcher:
    switch_id: int
    anchor_ms: int
    deadline_ms: int
    active: bool = True
    good_frames: int = 0


class MetricsManager:
    """In-memory metrics engine; agnostic to UI."""

    R_DANGER: float = 0.8

    def __init__(self, config: MetricsConfig):
        self.config = config
        self.current_episode: Optional[EpisodeMetrics] = None
        self._next_switch_id: int = 1

        self._reaction_watchers: Dict[int, ReactionWatcher] = {}
        self._hmr_watchers: Dict[int, HmrOcWatcher] = {}
        self._stab_watchers: Dict[int, StabilizationWatcher] = {}

    # ---------------------------------------------------------------- Episode lifecycle
    def start_episode(
        self,
        participant_id,
        condition: str,
        N_plates: int,
        episode_id: int,
        block_order: Optional[str],
        trial_index: Optional[int],
        layout_mapping_id: Optional[str],
        core,
    ) -> None:
        if self.config.level == MetricLevel.NONE:
            self.current_episode = None
            return

        ep = EpisodeMetrics(
            participant_id=participant_id,
            condition=condition,
            N_plates=N_plates,
            episode_id=episode_id,
            episode_uid=uuid.uuid4().hex,
            block_order=block_order,
            trial_index=trial_index,
            layout_mapping_id=layout_mapping_id,
        )
        ep._neglect_timers = [0.0] * N_plates

        try:
            init_stats = getattr(core, "_episode_init_stats", None)
            if isinstance(init_stats, dict):
                ep.reset_profile_used = init_stats.get("reset_profile_used", None)
                ep.init_mean_distance_norm = float(init_stats.get("init_mean_distance_norm", 0.0))
                ep.init_max_distance_norm = float(init_stats.get("init_max_distance_norm", 0.0))
                ep.init_mean_speed_norm = float(init_stats.get("init_mean_speed_norm", 0.0))
                ep.init_max_speed_norm = float(init_stats.get("init_max_speed_norm", 0.0))
                ep.init_mean_tilt_mag_norm = float(init_stats.get("init_mean_tilt_mag_norm", 0.0))
                ep.init_max_tilt_mag_norm = float(init_stats.get("init_max_tilt_mag_norm", 0.0))
                ep.init_drag = float(init_stats.get("init_drag", 0.0))
                ep.init_drag_decrease_amount = float(init_stats.get("init_drag_decrease_amount", 0.0))
                ep.init_anti_stall_speed_threshold = float(init_stats.get("init_anti_stall_speed_threshold", 0.0))
        except Exception:
            pass

        self.current_episode = ep

        self._next_switch_id = 1
        self._reaction_watchers.clear()
        self._hmr_watchers.clear()
        self._stab_watchers.clear()

    def end_episode(self, core, failure_plate_idx: Optional[int]) -> Optional[EpisodeMetrics]:
        if not self.current_episode or self.config.level == MetricLevel.NONE:
            return None

        ep = self.current_episode
        ep.duration_sec = core.game_time
        logical_fail = None
        if failure_plate_idx is not None:
            logical_fail = core.internal_to_logical[failure_plate_idx]
        ep.failure_plate_idx = logical_fail

        if ep.duration_sec > 0:
            ep.avg_plate_health = ep._health_integral / ep.duration_sec
            ep.avg_tilt_mag = ep._tilt_mag_integral / ep.duration_sec
            ep.avg_current_drag_norm = ep._drag_norm_integral / ep.duration_sec
            ep.danger_time_frac = ep._danger_time_total / (ep.N_plates * ep.duration_sec)

        self._reaction_watchers.clear()
        self._hmr_watchers.clear()
        self._stab_watchers.clear()
        self.current_episode = None
        return ep

    # ---------------------------------------------------------------- Micro-step update
    def on_micro_step(self, core, dt: float, joystick_x: float = 0.0, joystick_y: float = 0.0) -> None:
        if not self.current_episode or self.config.level == MetricLevel.NONE:
            return

        ep = self.current_episode

        H_t = self._compute_plate_health(core)
        ep._health_integral += H_t * dt

        mean_tilt = self._compute_mean_tilt_mag_norm(core)
        drag_norm = self._compute_drag_norm(core)
        ep._tilt_mag_integral += mean_tilt * dt
        ep._drag_norm_integral += drag_norm * dt

        cp_internal = core.controlled_plate
        for internal_idx in range(ep.N_plates):
            if internal_idx == cp_internal:
                ep._neglect_timers[internal_idx] = 0.0
            else:
                ep._neglect_timers[internal_idx] += dt
                ep.max_neglect_time_sec = max(ep.max_neglect_time_sec, ep._neglect_timers[internal_idx])

        for internal_idx in range(ep.N_plates):
            status = core.get_status_for_plate(internal_idx)
            R_i = 0.5 * status["distance_norm"] + 0.5 * status["speed_norm"]
            if R_i >= self.R_DANGER:
                ep._danger_time_total += dt

        if self.config.level == MetricLevel.NONE:
            return

        t_ms = int(round(core.game_time * 1000))
        if self.config.level >= MetricLevel.CORE:
            self._update_reaction_watchers(core, t_ms, joystick_x, joystick_y)
        if self.config.level >= MetricLevel.XAI:
            self._update_hmr_watchers(core, t_ms)
            self._update_stab_watchers(core, t_ms)

    # ---------------------------------------------------------------- Switch handling
    def on_switch_precue(
        self, core, precue_start_ms: int, source_idx: int, target_idx: int
    ) -> Optional[SwitchMetrics]:
        if not self.current_episode or self.config.level == MetricLevel.NONE:
            return None

        ep = self.current_episode
        switch_id = self._next_switch_id
        self._next_switch_id += 1

        internal_src = source_idx
        internal_tgt = target_idx
        logical_src = core.internal_to_logical[internal_src]
        logical_tgt = core.internal_to_logical[internal_tgt]

        dwell_ms = self._compute_dwell_time_ms(core, internal_src, precue_start_ms)
        last_commit_ms = int(round(core.last_switch_commit_time * 1000))
        dt_since_last_ms = precue_start_ms - last_commit_ms

        src = core.get_status_for_plate(internal_src)
        tgt = core.get_status_for_plate(internal_tgt)
        drag_norm = self._compute_drag_norm(core)

        sm = SwitchMetrics(
            participant_id=ep.participant_id,
            episode_id=ep.episode_id,
            episode_uid=ep.episode_uid,
            switch_id=switch_id,
            precue_start_ms=precue_start_ms,
            commit_ms=precue_start_ms,
            dwell_time_ms=dwell_ms,
            dt_since_last_switch_ms=dt_since_last_ms,
            source_plate_idx=logical_src,
            target_plate_idx=logical_tgt,
            current_drag_norm_precue=drag_norm,
            source_dist_precue=src["distance_norm"],
            source_speed_precue=src["speed_norm"],
            source_outvel_precue=src["outward_vel_norm"],
            source_tilt_mag_precue=src["tilt_mag_norm"],
            source_tilt_out_precue=src["tilt_outward_norm"],
            target_dist_precue=tgt["distance_norm"],
            target_speed_precue=tgt["speed_norm"],
            target_outvel_precue=tgt["outward_vel_norm"],
            target_tilt_mag_precue=tgt["tilt_mag_norm"],
            target_tilt_out_precue=tgt["tilt_outward_norm"],
            layout_mapping_id=ep.layout_mapping_id,
        )

        if self.config.level >= MetricLevel.XAI:
            sm.priority_lag = self._compute_priority_lag(core, logical_tgt)

        if ep.switches:
            prev = ep.switches[-1]
            if (
                prev.commit_ms is not None
                and precue_start_ms - prev.commit_ms <= 1000
                and logical_tgt == prev.source_plate_idx
            ):
                prev.is_flap = True

        ep.switches.append(sm)
        return sm

    def on_switch_commit_for_last(self, core, commit_ms: int) -> None:
        if not self.current_episode or self.config.level == MetricLevel.NONE:
            return
        ep = self.current_episode
        if not ep.switches:
            return

        switch = ep.switches[-1]
        switch.commit_ms = commit_ms

        internal_src = core.get_internal_plate_index(switch.source_plate_idx)
        internal_tgt = core.get_internal_plate_index(switch.target_plate_idx)

        src = core.get_status_for_plate(internal_src)
        tgt = core.get_status_for_plate(internal_tgt)

        switch.source_dist_commit = src["distance_norm"]
        switch.source_speed_commit = src["speed_norm"]
        switch.source_outvel_commit = src["outward_vel_norm"]
        switch.source_tilt_mag_commit = src["tilt_mag_norm"]
        switch.source_tilt_out_commit = src["tilt_outward_norm"]

        switch.target_dist_commit = tgt["distance_norm"]
        switch.target_speed_commit = tgt["speed_norm"]
        switch.target_outvel_commit = tgt["outward_vel_norm"]
        switch.target_tilt_mag_commit = tgt["tilt_mag_norm"]
        switch.target_tilt_out_commit = tgt["tilt_outward_norm"]

        if self.config.level >= MetricLevel.XAI:
            switch.target_R_commit = self._risk_index(switch.target_dist_commit, switch.target_speed_commit)
            switch.source_R_commit = self._risk_index(switch.source_dist_commit, switch.source_speed_commit)

            self._hmr_watchers[switch.switch_id] = HmrOcWatcher(
                switch_id=switch.switch_id, sample_time_ms=commit_ms + 1000
            )
            self._stab_watchers[switch.switch_id] = StabilizationWatcher(
                switch_id=switch.switch_id, anchor_ms=commit_ms, deadline_ms=commit_ms + 2000
            )

        if self.config.level >= MetricLevel.CORE:
            anchor_ms = commit_ms
            self._reaction_watchers[switch.switch_id] = ReactionWatcher(switch_id=switch.switch_id, anchor_ms=anchor_ms)

        if not switch.aborted:
            ep.num_switches += 1

    def on_baseline_switch(self, core, event_ms: int, source_idx: int, target_idx: int) -> None:
        self.on_switch_precue(core, event_ms, source_idx, target_idx)
        self.on_switch_commit_for_last(core, event_ms)

    def mark_last_switch_aborted(self) -> None:
        if not self.current_episode or not self.current_episode.switches:
            return
        self.current_episode.switches[-1].aborted = True

    # ---------------------------------------------------------------- Watchers
    def _update_reaction_watchers(self, core, t_ms: int, joystick_x: float, joystick_y: float) -> None:
        if not self.current_episode:
            return
        ep = self.current_episode
        mag = math.sqrt(joystick_x * joystick_x + joystick_y * joystick_y)

        for switch in ep.switches:
            watcher = self._reaction_watchers.get(switch.switch_id)
            if watcher is None or not watcher.active:
                continue
            if t_ms < watcher.anchor_ms:
                continue
            # Only measure reaction while the switched-to (target) plate is currently controlled.
            # If the user switches away before reacting, reaction_time_ms is undefined for this switch.
            # Deactivate the watcher to prevent late/false attribution.
            target_internal = core.get_internal_plate_index(switch.target_plate_idx)
            if core.controlled_plate != target_internal:
                watcher.active = False
                self._reaction_watchers.pop(switch.switch_id, None)
                continue

            if mag < 0.10:
                watcher.frames_above_thresh = 0
                continue

            target_internal = core.get_internal_plate_index(switch.target_plate_idx)
            ball = core.balls[target_internal]
            corr_x, corr_y = -ball.x, -ball.y
            corr_norm = math.sqrt(corr_x * corr_x + corr_y * corr_y)
            if corr_norm < 1e-6:
                watcher.frames_above_thresh = 0
                continue

            jx_n, jy_n = joystick_x / mag, joystick_y / mag
            cx_n, cy_n = corr_x / corr_norm, corr_y / corr_norm
            dot = jx_n * cx_n + jy_n * cy_n
            if dot < 0.0:
                watcher.frames_above_thresh = 0
                continue

            watcher.frames_above_thresh += 1
            if watcher.frames_above_thresh >= 3:
                watcher.active = False
                switch.reaction_time_ms = t_ms - watcher.anchor_ms
                self._reaction_watchers.pop(switch.switch_id, None)


    def _update_hmr_watchers(self, core, t_ms: int) -> None:
        if not self.current_episode:
            return
        ep = self.current_episode
        done_ids = []
        for switch_id, watcher in self._hmr_watchers.items():
            if not watcher.active or t_ms < watcher.sample_time_ms:
                continue

            switch = next((s for s in ep.switches if s.switch_id == switch_id), None)
            if switch is None:
                done_ids.append(switch_id)
                continue

            tgt_internal = core.get_internal_plate_index(switch.target_plate_idx)
            src_internal = core.get_internal_plate_index(switch.source_plate_idx)
            tgt = core.get_status_for_plate(tgt_internal)
            src = core.get_status_for_plate(src_internal)

            switch.target_R_plus1s = self._risk_index(tgt["distance_norm"], tgt["speed_norm"])
            switch.source_R_plus1s = self._risk_index(src["distance_norm"], src["speed_norm"])
            if switch.target_R_commit is not None and switch.target_R_plus1s is not None:
                switch.HMR = switch.target_R_plus1s - switch.target_R_commit
                switch.hmr_available = True
            if switch.source_R_commit is not None and switch.source_R_plus1s is not None:
                switch.OC = switch.source_R_plus1s - switch.source_R_commit
                switch.oc_available = True

            watcher.active = False
            done_ids.append(switch_id)

        for sid in done_ids:
            self._hmr_watchers.pop(sid, None)

    def _update_stab_watchers(self, core, t_ms: int) -> None:
        if not self.current_episode:
            return
        ep = self.current_episode
        done_ids = []
        for switch_id, watcher in self._stab_watchers.items():
            if not watcher.active:
                done_ids.append(switch_id)
                continue

            switch = next((s for s in ep.switches if s.switch_id == switch_id), None)
            if switch is None:
                done_ids.append(switch_id)
                continue

            if t_ms < watcher.anchor_ms:
                continue

            # Only measure stabilization while the switched-to (target) plate is currently controlled.
            # If the user switches away before stabilization, treat stabilization as not achieved and
            # deactivate to prevent late/false attribution.
            tgt_internal = core.get_internal_plate_index(switch.target_plate_idx)
            if core.controlled_plate != tgt_internal:
                watcher.active = False
                done_ids.append(switch_id)
                continue

            if t_ms > watcher.deadline_ms:
                watcher.active = False
                done_ids.append(switch_id)
                continue

            tgt = core.get_status_for_plate(tgt_internal)
            if tgt["distance_norm"] < 0.25 and tgt["speed_norm"] < 0.30:
                watcher.good_frames += 1
                if watcher.good_frames >= 12:
                    watcher.active = False
                    switch.stab_available = True
                    switch.stabilization_time_ms = t_ms - watcher.anchor_ms
                    done_ids.append(switch_id)
            else:
                watcher.good_frames = 0

        for sid in done_ids:
            self._stab_watchers.pop(sid, None)

    # ---------------------------------------------------------------- Helpers
    def _compute_plate_health(self, core) -> float:
        total = 0.0
        for internal_idx in range(core.num_plates):
            status = core.get_status_for_plate(internal_idx)
            d = status["distance_norm"]
            total += 1.0 - d * d
        return total / float(core.num_plates)

    def _compute_mean_tilt_mag_norm(self, core) -> float:
        total = 0.0
        for internal_idx in range(core.num_plates):
            status = core.get_status_for_plate(internal_idx)
            total += status["tilt_mag_norm"]
        return total / float(core.num_plates)

    def _compute_drag_norm(self, core) -> float:
        status = core.get_status()
        return status.get("current_drag_norm", 0.0)

    def _compute_dwell_time_ms(self, core, source_internal_idx: int, precue_start_ms: int) -> int:
        if source_internal_idx < len(core.last_take_control_time):
            start_s = core.last_take_control_time[source_internal_idx]
            dwell = max(0.0, (precue_start_ms / 1000.0) - start_s)
            return int(round(dwell * 1000))
        return 0

    def _compute_priority_lag(self, core, target_logical_idx: int) -> Optional[int]:
        risks = []
        for logical_idx in range(core.num_plates):
            internal_idx = core.get_internal_plate_index(logical_idx)
            status = core.get_status_for_plate(internal_idx)
            risks.append((logical_idx, self._risk_index(status["distance_norm"], status["speed_norm"])))
        risks.sort(key=lambda x: x[1], reverse=True)
        for rank, (logical, _) in enumerate(risks):
            if logical == target_logical_idx:
                return rank
        return None

    @staticmethod
    def _risk_index(distance_norm: float, speed_norm: float) -> float:
        return 0.5 * distance_norm + 0.5 * speed_norm


class EpisodeLogger:
    """CSV logger with optional o-key confirmation gate."""

    def __init__(self, config: MetricsConfig, require_o_confirmation: bool):
        self.config = config
        self.require_o = require_o_confirmation
        self._pending_episode: Optional[EpisodeMetrics] = None

        self._episodes_file = None
        self._switches_file = None
        self._episodes_writer = None
        self._switches_writer = None

        if self.config.log_to_csv:
            # Ensure output directory exists
            ep_dir = os.path.dirname(self.config.episodes_csv_path) or "."
            sw_dir = os.path.dirname(self.config.switches_csv_path) or "."
            os.makedirs(ep_dir, exist_ok=True)
            os.makedirs(sw_dir, exist_ok=True)
            # Migrate existing CSV headers if needed (e.g., add episode_uid column)
            _ensure_csv_header(self.config.episodes_csv_path, self._episode_fieldnames())
            _ensure_csv_header(self.config.switches_csv_path, self._switch_fieldnames())

            self._episodes_file = open(self.config.episodes_csv_path, "a", newline="")
            self._switches_file = open(self.config.switches_csv_path, "a", newline="")
            self._episodes_writer = csv.DictWriter(self._episodes_file, fieldnames=self._episode_fieldnames())
            self._switches_writer = csv.DictWriter(self._switches_file, fieldnames=self._switch_fieldnames())

            if os.path.getsize(self.config.episodes_csv_path) == 0:
                self._episodes_writer.writeheader()
            if os.path.getsize(self.config.switches_csv_path) == 0:
                self._switches_writer.writeheader()

    def on_episode_finished(self, ep: EpisodeMetrics) -> None:
        if not self.config.log_to_csv:
            return
        if self.require_o:
            self._pending_episode = ep
        else:
            self._write_episode_and_switches(ep)

    def confirm_current_episode(self) -> bool:
        if not self.config.log_to_csv:
            return False
        if self._pending_episode is None:
            return False
        try:
            self._write_episode_and_switches(self._pending_episode)
            self._pending_episode = None
            return True
        except Exception:
            return False

    def discard_current_episode(self) -> None:
        self._pending_episode = None

    # ---------------------------------------------------------------- I/O helpers
    def _write_episode_and_switches(self, ep: EpisodeMetrics) -> None:
        if not self._episodes_writer or not self._switches_writer:
            return
        ep_row = self._episode_to_row(ep)
        self._episodes_writer.writerow(ep_row)
        self._episodes_file.flush()

        for sm in ep.switches:
            if sm.aborted:
                continue
            sw_row = self._switch_to_row(sm)
            self._switches_writer.writerow(sw_row)
        self._switches_file.flush()

    @staticmethod
    def _episode_fieldnames() -> List[str]:
        return [
            "participant_id",
            "condition",
            "N_plates",
            "episode_id",
            "episode_uid",
            "block_order",
            "trial_index",
            "layout_mapping_id",
            "duration_sec",
            "avg_plate_health",
            "num_switches",
            "failure_plate_idx",
            "avg_tilt_mag",
            "avg_current_drag_norm",
            "max_neglect_time_sec",
            "danger_time_frac",
            "reset_profile_used",
            "init_mean_distance_norm",
            "init_max_distance_norm",
            "init_mean_speed_norm",
            "init_max_speed_norm",
            "init_mean_tilt_mag_norm",
            "init_max_tilt_mag_norm",
            "init_drag",
            "init_drag_decrease_amount",
            "init_anti_stall_speed_threshold",
        ]

    @staticmethod
    def _switch_fieldnames() -> List[str]:
        return [
            "participant_id",
            "episode_id",
            "episode_uid",
            "switch_id",
            "precue_start_ms",
            "commit_ms",
            "dwell_time_ms",
            "dt_since_last_switch_ms",
            "source_plate_idx",
            "target_plate_idx",
            "reaction_time_ms",
            "current_drag_norm_precue",
            "source_dist_precue",
            "source_speed_precue",
            "source_outvel_precue",
            "source_tilt_mag_precue",
            "source_tilt_out_precue",
            "target_dist_precue",
            "target_speed_precue",
            "target_outvel_precue",
            "target_tilt_mag_precue",
            "target_tilt_out_precue",
            "source_dist_commit",
            "source_speed_commit",
            "source_outvel_commit",
            "source_tilt_mag_commit",
            "source_tilt_out_commit",
            "target_dist_commit",
            "target_speed_commit",
            "target_outvel_commit",
            "target_tilt_mag_commit",
            "target_tilt_out_commit",
            "priority_lag",
            "target_R_commit",
            "target_R_plus1s",
            "source_R_commit",
            "source_R_plus1s",
            "HMR",
            "OC",
            "hmr_available",
            "oc_available",
            "stabilization_time_ms",
            "stab_available",
            "is_flap",
            "layout_mapping_id",
            "aborted",
        ]

    def _episode_to_row(self, ep: EpisodeMetrics) -> Dict:
        row = {k: getattr(ep, k) for k in self._episode_fieldnames()}
        return row

    def _switch_to_row(self, sm: SwitchMetrics) -> Dict:
        row = {k: getattr(sm, k) for k in self._switch_fieldnames()}
        return row
