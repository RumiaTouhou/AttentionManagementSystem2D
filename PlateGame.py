from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np

import pygame

from CognitiveAgent import CognitiveAgent, CognitiveParams
from ControlMapping import NullPromptAPI, PromptAPI, PromptState, SingleHandPromptController
from Metrics import EpisodeLogger, MetricLevel, MetricsConfig, MetricsManager, _ensure_csv_header

from AMSCore import PPOAMS

from datetime import datetime

# Extend AMS for robust performance when N>2
# Game re-implemented in 2D to avoid unwanted human factors
# Essential safety layers added for robust performance when N>2 

FPS = 60
DT_MICRO = 1.0 / FPS

INITIAL_DRAG = 1.2
DRAG_DECREASE_AMOUNT = 0.0353
DRAG_DECREASE_INTERVAL = 5.0
MIN_DRAG = -0.2

BASE_GRAVITY = 0.32

ANTI_STALL_SPEED_THRESHOLD = 2.0
ANTI_STALL_BASE_FORCE = 0.01

PLATE_RADIUS = 1500
PLATE_DISPLAY_RADIUS = 1000
BALL_RADIUS = 30
MAX_TILT = 45
TILT_BEGINNING = 3

GRID_HORIZONTAL_MARGIN_PERCENT = 0.04
GRID_LEFT_MARGIN_PERCENT = 0.06
GRID_VERTICAL_MARGIN_PERCENT = 0.03

PLATE_DISPLAY_SCALE_FACTOR = 1.1

PLATE_NUMBER_X_OFFSET = -1 / 3
PLATE_NUMBER_Y_OFFSET = 1 / 4
PLATE_LABEL_Y_SPACING = 1.2

SINGLE_HAND_MODE = True

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

BACKGROUND_COLORS = {
    "default_background": (0, 0, 0),
    "single_alert": (120, 120, 0),
    "right_normal": (0, 27, 46),
    "preserve_plate": (52, 52, 52),
    "right_alert": (70, 70, 106),
    "left_normal": (54, 0, 0),
    "single_normal": (48, 48, 0),
    "switch_from": (143, 45, 87),
    "left_alert": (91, 59, 59),
    "mode_switch": (8, 74, 77),
}

BACKGROUND_COLOR_NAMES = list(BACKGROUND_COLORS.keys())

TILT_RATE = 0.48
JOYSTICK_DEADZONE = 0.05
JOYSTICK_SENSITIVITY = 3.4

STATUS_FONT_SIZE = 32
MESSAGE_FONT_SIZE = 64
PLATE_NUMBER_FONT_SIZE = 54
BACKGROUND_TEXT_SCALE = 3.5
BACKGROUND_TEXT_FONT_SIZE = int(72 * BACKGROUND_TEXT_SCALE)
PLATE_INTERNAL_TRANSPARENCY = 50

R_USABLE = PLATE_RADIUS - BALL_RADIUS
V_MAX_NORMS = 0.04597
U_MAX = 10.0
VOLUME_TUNING = 0.7

# Emergency triage threshold (seconds): when switching is allowed and any plate's
# time-to-boundary falls below this value, force attention to the most urgent plate.
EMERGENCY_TTB_SEC = 2.0

class Mode(Enum):
    TRAINING = "Training"
    HUMAN_BASELINE = "HumanBaseline"
    AMS_PLAY = "AMSPlay"
    PLAYGROUND_NO_AMS = "PlaygroundNoAMS"
    PLAYGROUND_AMS = "PlaygroundAMS"


@dataclass
class GameConfig:
    mode: Mode = Mode.PLAYGROUND_NO_AMS
    num_plates: int = 4
    participant_id: Optional[int] = None
    display_belief: bool = False
    debug_prompt: bool = False
    headless: bool = False
    seed: Optional[int] = None
    N_max: int = 12
    ams_checkpoint: str = "auto"

    # Logging metadata only (does not affect gameplay or training).
    block_order: Optional[str] = None   # e.g., "BaselineFirst" / "AMSFirst"
    trial_index: Optional[int] = None   # e.g., within-block N index

    # Output directory override for user-study CSV logs.
    log_dir: Optional[str] = None
    # Opt-in per-decision trace log (decision_trace.csv).
    decision_trace: bool = False

class AMSInterface:
    """Minimal interface for AMS policies."""

    def select_action(self, obs, action_mask) -> int:
        raise NotImplementedError




class Ball:
    def __init__(self, x: float, y: float):
        self.reset(x, y)

    def reset(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.ax = 0.0
        self.ay = 0.0

    def calculate_anti_stall(self, speed: float, current_drag: float, anti_stall_speed_threshold: float) -> float:
        if 0 < speed < anti_stall_speed_threshold:
            drag_factor = max(1.0, (INITIAL_DRAG - current_drag) / 0.6)
            speed_factor = 1.0 - (speed / anti_stall_speed_threshold)
            return ANTI_STALL_BASE_FORCE * drag_factor * speed_factor
        return 0.0

    def update(self, plate: "Plate", current_drag: float, anti_stall_speed_threshold: float = ANTI_STALL_SPEED_THRESHOLD) -> bool:
        angle_rad = math.radians(plate.tilt_magnitude)
        direction_rad = math.radians(plate.tilt_direction)

        gravity_acc = BASE_GRAVITY * math.sin(angle_rad)
        self.ax = gravity_acc * math.cos(direction_rad)
        self.ay = gravity_acc * math.sin(direction_rad)

        speed = math.sqrt(self.vx * self.vx + self.vy * self.vy)
        if 0 < speed < anti_stall_speed_threshold:
            anti_stall = self.calculate_anti_stall(speed, current_drag, anti_stall_speed_threshold)
            self.ax += (self.vx / speed) * anti_stall
            self.ay += (self.vy / speed) * anti_stall

        self.vx += self.ax
        self.vy += self.ay

        drag_mult = math.exp(-current_drag * DT_MICRO)
        self.vx *= drag_mult
        self.vy *= drag_mult

        self.x += self.vx
        self.y += self.vy

        dist = math.sqrt(self.x * self.x + self.y * self.y)
        if dist > PLATE_RADIUS - BALL_RADIUS:
            return False
        return True

    def get_speed(self) -> float:
        return math.sqrt(self.vx * self.vx + self.vy * self.vy)

    def get_distance_from_center(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)


class Plate:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.tilt_magnitude = 0.0
        self.tilt_direction = 0.0
        self.x_tilt = 0.0
        self.y_tilt = 0.0

    def apply_random_tilt(self) -> None:
        angle = random.randint(0, 359)
        self.tilt_magnitude = TILT_BEGINNING
        self.tilt_direction = angle
        rad = math.radians(angle)
        self.x_tilt = self.tilt_magnitude * math.cos(rad)
        self.y_tilt = self.tilt_magnitude * math.sin(rad)

    def update_with_joystick(self, stick_x: float, stick_y: float) -> None:
        def apply_deadzone(value: float, deadzone: float) -> float:
            if abs(value) < deadzone:
                return 0.0
            sign = 1.0 if value > 0 else -1.0
            return sign * ((abs(value) - deadzone) / (1.0 - deadzone))

        x_in = apply_deadzone(stick_x, JOYSTICK_DEADZONE)
        y_in = apply_deadzone(stick_y, JOYSTICK_DEADZONE)

        self.x_tilt += x_in * TILT_RATE * JOYSTICK_SENSITIVITY
        self.y_tilt += y_in * TILT_RATE * JOYSTICK_SENSITIVITY

        mag = math.sqrt(self.x_tilt * self.x_tilt + self.y_tilt * self.y_tilt)
        if mag > 0:
            if mag > MAX_TILT:
                scale = MAX_TILT / mag
                self.x_tilt *= scale
                self.y_tilt *= scale
                mag = MAX_TILT

            self.tilt_magnitude = mag
            self.tilt_direction = math.degrees(math.atan2(self.y_tilt, self.x_tilt))
            if self.tilt_direction < 0:
                self.tilt_direction += 360


class GameState(Enum):
    NOT_STARTED = 0
    RUNNING = 1
    PAUSED = 2
    GAME_OVER = 3


@dataclass
class PhysicsParams:
    initial_drag: float = INITIAL_DRAG
    drag_decrease_amount: float = DRAG_DECREASE_AMOUNT
    anti_stall_speed_threshold: float = ANTI_STALL_SPEED_THRESHOLD


class MultiPlateCore:
    def __init__(self, num_plates: int, seed: Optional[int] = None, physics_params: Optional[PhysicsParams] = None):
        self.num_plates = self._validate_num_plates(num_plates)
        if seed is not None:
            random.seed(seed)

        self.params = physics_params or PhysicsParams()
        self.plates: List[Plate] = [Plate() for _ in range(self.num_plates)]
        self.balls: List[Ball] = [Ball(0.0, 0.0) for _ in range(self.num_plates)]

        self.logical_to_internal: List[int] = list(range(self.num_plates))
        self.internal_to_logical: List[int] = list(range(self.num_plates))

        self.state = GameState.NOT_STARTED
        self.game_time = 0.0
        self.current_drag = self.params.initial_drag

        self.controlled_plate: int = 0
        self.last_switch_commit_time: float = 0.0
        self.last_take_control_time: List[float] = [0.0] * self.num_plates
        self.unattended_times: List[float] = [0.0] * self.num_plates
        self.unattended_times_ams: List[float] = [0.0] * self.num_plates
        self._last_ams_unattended_update: float = 0.0
        self.failure_plate_idx: Optional[int] = None

        self.ever_controlled: List[bool] = [False] * self.num_plates

        self.TILT_RATE = TILT_RATE
        self.JOYSTICK_SENSITIVITY = JOYSTICK_SENSITIVITY

    def _validate_num_plates(self, num: int) -> int:
        return max(2, min(12, int(num)))

    def _min_dwell_dt_s_for_N(self, N: int) -> float:
        """
        Minimum dwell time (seconds) after a switch commit before a new pre-cue may begin.

        This dwell time excludes the pre-cue duration itself.

        Values:
          - N = 2: 0.3
          - N in {3,4}: 0.6
          - N in {5,6}: 0.8
          - N >= 7: 0.9
        """
        if N <= 2:
            return 0.3
        elif N in (3, 4):
            return 0.6
        elif N in (5, 6):
            return 0.8
        else:
            return 0.9


    def set_logical_mapping(self, perm: List[int]) -> None:
        assert len(perm) == self.num_plates
        self.logical_to_internal = perm
        self.internal_to_logical = [0] * self.num_plates
        for logical_idx, internal_idx in enumerate(perm):
            self.internal_to_logical[internal_idx] = logical_idx

    def get_internal_plate_index(self, logical_idx: int) -> int:
        return self.logical_to_internal[logical_idx]

    def reset(self, seed: Optional[int] = None, training: bool = False) -> None:
        if seed is not None:
            random.seed(seed)
        for plate in self.plates:
            plate.reset()
            plate.apply_random_tilt()
        use_center_start = training and (random.random() < 0.5)

        for i, ball in enumerate(self.balls):
            if training and not use_center_start:
                r = random.uniform(0.0, 0.1 * R_USABLE)
                theta = random.uniform(0.0, 2.0 * math.pi)
                x = r * math.cos(theta)
                y = r * math.sin(theta)
                ball.reset(x, y)
                ball.vx = random.uniform(-0.5, 0.5)
                ball.vy = random.uniform(-0.5, 0.5)
            else:
                ball.reset(0.0, 0.0)

        self.state = GameState.NOT_STARTED
        self.game_time = 0.0
        self.current_drag = self.params.initial_drag
        self.controlled_plate = 0
        self.last_switch_commit_time = 0.0
        self.last_take_control_time = [0.0] * self.num_plates
        self.unattended_times = [0.0] * self.num_plates
        self.unattended_times_ams = [0.0] * self.num_plates
        self._last_ams_unattended_update = 0.0
        self.failure_plate_idx = None
        self.ever_controlled = [False] * self.num_plates
        if 0 <= self.controlled_plate < self.num_plates:
            self.ever_controlled[self.controlled_plate] = True


    def step_physics(self, dt: float) -> bool:
        if self.state == GameState.NOT_STARTED:
            self.state = GameState.RUNNING
        if self.state != GameState.RUNNING:
            return True

        self.game_time += dt
        self._update_difficulty()

        for i in range(self.num_plates):
            if i == self.controlled_plate:
                self.unattended_times[i] = 0.0
            else:
                self.unattended_times[i] = min(U_MAX, self.unattended_times[i] + dt)

        for i, (plate, ball) in enumerate(zip(self.plates, self.balls)):
            alive = ball.update(plate, self.current_drag, self.params.anti_stall_speed_threshold)
            if not alive:
                self.state = GameState.GAME_OVER
                self.failure_plate_idx = i
                return False
        return True

    def _update_difficulty(self) -> None:
        elapsed_intervals = int(self.game_time // DRAG_DECREASE_INTERVAL)
        self.current_drag = max(MIN_DRAG, self.params.initial_drag - elapsed_intervals * self.params.drag_decrease_amount)

    def commit_switch(self, target_idx: int, t_now: float) -> Dict[str, float]:
        source_idx = self.controlled_plate
        if target_idx == source_idx:
            return {
                "source": source_idx,
                "target": target_idx,
                "dt_since_last_switch": t_now - self.last_switch_commit_time,
            }

        dt_since_last = t_now - self.last_switch_commit_time
        self.last_switch_commit_time = t_now
        self.controlled_plate = target_idx
        self.last_take_control_time[target_idx] = t_now
        self.unattended_times[target_idx] = 0.0
        self.unattended_times_ams[target_idx] = 0.0
        if 0 <= target_idx < len(self.ever_controlled):
            self.ever_controlled[target_idx] = True

        return {"source": source_idx, "target": target_idx, "dt_since_last_switch": dt_since_last}
    
    def _compute_ttb_sec_for_plate(self, internal_idx: int) -> float:
        """
        Composite time-to-boundary (seconds), layout-independent.

        - Uses a ballistic circle-intersection estimate when speed is non-trivial.
        - Uses a tilt-driven acceleration estimate to handle v≈0 and center-start.
        - Includes a small inward-motion guard to avoid over-prioritizing plates
          that are clearly moving inward and buying time.
        """
        R = float(R_USABLE)
        ball = self.balls[internal_idx]
        plate = self.plates[internal_idx]

        x, y = float(ball.x), float(ball.y)
        vx, vy = float(ball.vx), float(ball.vy)
        r2 = x * x + y * y
        if r2 >= R * R:
            return 0.0

        r = math.sqrt(r2)
        r_eps = 0.05 * R

        v2 = vx * vx + vy * vy
        ttb_ballistic = float("inf")
        if v2 > 1e-12:
            pv = x * vx + y * vy
            A = v2
            B = 2.0 * pv
            C = r2 - R * R
            disc = B * B - 4.0 * A * C
            if disc > 0.0:
                s = math.sqrt(disc)
                t1 = (-B - s) / (2.0 * A)
                t2 = (-B + s) / (2.0 * A)
                candidates = [t for t in (t1, t2) if t > 0.0]
                if candidates:
                    ttb_ballistic = min(candidates) / float(FPS)

        angle_rad = math.radians(float(plate.tilt_magnitude))
        g_mag_frame = float(BASE_GRAVITY) * math.sin(angle_rad)
        ttb_acc = float("inf")

        if g_mag_frame > 1e-9:
            if r < r_eps:
                t_frames = math.sqrt(2.0 * R / max(g_mag_frame, 1e-9))
                ttb_acc = t_frames / float(FPS)
            else:
                rhx, rhy = x / r, y / r
                dir_rad = math.radians(float(plate.tilt_direction))
                ax_frame = g_mag_frame * math.cos(dir_rad)
                ay_frame = g_mag_frame * math.sin(dir_rad)
                a_r_frame = ax_frame * rhx + ay_frame * rhy

                v_r_frame = vx * rhx + vy * rhy
                d = R - r
                if d <= 0.0:
                    ttb_acc = 0.0
                else:
                    A = 0.5 * a_r_frame
                    B = v_r_frame
                    C = -d
                    if abs(A) < 1e-12:
                        if B > 1e-9:
                            ttb_acc = (d / B) / float(FPS)
                    else:
                        disc = B * B - 4.0 * A * C
                        if disc > 0.0:
                            s = math.sqrt(disc)
                            t1 = (-B - s) / (2.0 * A)
                            t2 = (-B + s) / (2.0 * A)
                            candidates = [t for t in (t1, t2) if t > 0.0]
                            if candidates:
                                ttb_acc = min(candidates) / float(FPS)

        if not math.isfinite(ttb_ballistic):
            ttb_ballistic = float("inf")
        if not math.isfinite(ttb_acc):
            ttb_acc = float("inf")

        if r < r_eps:
            ttb = min(ttb_ballistic, ttb_acc)
        else:
            rhx, rhy = x / r, y / r
            v_r_frame = vx * rhx + vy * rhy
            if v_r_frame >= -1e-6:
                ttb = min(ttb_ballistic, ttb_acc)
            else:
                inward_buffer = 1.0
                ttb = min(ttb_acc, max(ttb_ballistic, inward_buffer))

        if not math.isfinite(ttb):
            ttb = 60.0
        ttb = max(0.0, min(ttb, 60.0))
        return ttb


    def get_status_for_plate(self, i: int) -> Dict[str, float]:
        ball = self.balls[i]
        plate = self.plates[i]

        d = ball.get_distance_from_center()
        distance_norm = min(d / R_USABLE, 1.0)

        speed_px_per_frame = ball.get_speed()
        v_norm_per_s = speed_px_per_frame * FPS / R_USABLE
        speed_ratio = v_norm_per_s / V_MAX_NORMS if V_MAX_NORMS > 1e-12 else 0.0
        speed_norm = min(speed_ratio, 1.0)
        speed_over = max(0.0, speed_ratio - 1.0)
        speed_over_norm = speed_over / (1.0 + speed_over)

        speed_log_cap = 10.0
        speed_log_norm = math.log1p(min(speed_ratio, speed_log_cap)) / math.log1p(speed_log_cap)
        speed_log_norm = max(0.0, min(1.0, speed_log_norm))

        if d > 1e-6:
            rx, ry = ball.x / d, ball.y / d
            outward_px_per_frame = ball.vx * rx + ball.vy * ry
            outward_norm_per_s = outward_px_per_frame * FPS / R_USABLE
            outward_ratio = outward_norm_per_s / V_MAX_NORMS if V_MAX_NORMS > 1e-12 else 0.0

            out_vel_norm = max(-1.0, min(outward_ratio, 1.0))
            out_over = max(0.0, outward_ratio - 1.0)
            out_over_norm = out_over / (1.0 + out_over)
            out_log_cap = 10.0
            outward_log_norm = math.copysign(
                math.log1p(min(abs(outward_ratio), out_log_cap)) / math.log1p(out_log_cap),
                outward_ratio,
            )
            outward_log_norm = max(-1.0, min(1.0, outward_log_norm))
        else:
            out_vel_norm = 0.0
            out_over_norm = 0.0
            outward_log_norm = 0.0

        tilt_mag_norm = plate.tilt_magnitude / MAX_TILT if MAX_TILT > 0 else 0.0

        if d > 1e-6:
            rad_mag = math.radians(plate.tilt_magnitude)
            rad_dir = math.radians(plate.tilt_direction)
            ball_angle = math.atan2(ball.y, ball.x)
            delta = rad_dir - ball_angle
            outward_component = math.sin(rad_mag) * math.cos(delta)
            denom = math.sin(math.radians(MAX_TILT)) or 1.0
            tilt_outward_norm = outward_component / denom
        else:
            tilt_outward_norm = 0.0
        tilt_outward_norm = max(-1.0, min(1.0, tilt_outward_norm))

        unattended = self.unattended_times[i]
        unattended_norm = min(unattended / U_MAX, 1.0)

        ttb_sec = self._compute_ttb_sec_for_plate(i)
        ttb_scale = 5.0
        ttb_urgency = 1.0 / (1.0 + (ttb_sec / ttb_scale))
        ttb_sec_norm10 = min(ttb_sec, 10.0) / 10.0


        return {
            "distance_norm": distance_norm,
            "speed_norm": speed_norm,
            "speed_over_norm": speed_over_norm,
            "speed_log_norm": speed_log_norm,
            "outward_vel_norm": out_vel_norm,
            "outward_over_norm": out_over_norm,
            "outward_log_norm": outward_log_norm,
            "tilt_mag_norm": tilt_mag_norm,
            "tilt_outward_norm": tilt_outward_norm,
            "unattended_time_norm": unattended_norm,
            "ttb_sec": ttb_sec,
            "ttb_urgency": ttb_urgency,
            "ttb_sec_norm10": ttb_sec_norm10,
        }

    def get_status(self) -> Dict:
        denom = (INITIAL_DRAG - MIN_DRAG) if INITIAL_DRAG != MIN_DRAG else 0.0
        drag_norm = ((self.current_drag - MIN_DRAG) / denom) if denom != 0 else 0.0
        drag_norm = max(0.0, min(1.0, drag_norm))
        controlled_logical = self.internal_to_logical[self.controlled_plate]
        # Index mapping: controlled_plate is internal; returned controlled_plate is logical.
        # The returned "plates" list is in internal order (0..num_plates-1).
        return {
            "game_time": self.game_time,
            "state": self.state,
            "controlled_plate": controlled_logical,
            "current_drag": self.current_drag,
            "current_drag_norm": drag_norm,
            "plates": [self.get_status_for_plate(i) for i in range(self.num_plates)],
        }

    def build_ams_obs_and_mask(
        self,
        config: GameConfig,
        prompt_busy: bool = False,
        time_to_commit: float = 0.0,
    ) -> Tuple[List[float], List[int]]:
        """
        Build AMS observation vector and action mask.

        Observation layout:
          - Per-plate features (13) repeated for N_max slots; unused slots are zero-padded.
          - Global features appended after the per-plate block.

        Per-plate features (13):
          1) distance_norm            [0,1]
          2) speed_norm               [0,1]
          3) speed_over_norm          [0,1]
          4) speed_log_norm           [0,1]
          5) outward_vel_norm         [-1,1]
          6) outward_over_norm        [0,1]
          7) outward_log_norm         [-1,1]
          8) unattended_time_ams_norm [0,1]
          9) tilt_mag_norm            [0,1]
         10) tilt_outward_norm        [-1,1]
         11) ttb_urgency              [0,1]
         12) ttb_sec_norm10           [0,1]
         13) ever_controlled_flag     {0,1}

        Global features:
          - controlled_one_hot        (N_max)
          - current_drag_norm         [0,1]
          - dt_s_norm                 [0,1]  (time since last commit, capped at 3s)
          - N_norm                    [0,1]  (N / N_max)
          - prompt_busy_flag          {0,1}
          - time_to_commit_norm       [0,1]  (normalized by 0.3s)
          - min_ttb_norm10            [0,1]  (min TTB over plates, capped at 10s)
          - frac_ttb_lt2              [0,1]  (fraction of plates with TTB < 2s)

        The action mask is returned separately.
        """
        status = self.get_status()
        N = self.num_plates
        N_max = config.N_max

        delta = max(0.0, self.game_time - self._last_ams_unattended_update)
        for i in range(self.num_plates):
            if i == self.controlled_plate:
                self.unattended_times_ams[i] = 0.0
            else:
                self.unattended_times_ams[i] = min(U_MAX, self.unattended_times_ams[i] + delta)
        self._last_ams_unattended_update = self.game_time

        features: List[float] = []
        mask: List[int] = []
        ttb_secs: List[float] = []

        for logical_i in range(N_max):
            if logical_i < N:
                internal_i = self.get_internal_plate_index(logical_i)
                plate_status = self.get_status_for_plate(internal_i)

                unattended_norm = min(self.unattended_times_ams[internal_i] / U_MAX, 1.0)
                ever = 1.0 if (0 <= internal_i < len(self.ever_controlled) and self.ever_controlled[internal_i]) else 0.0

                ttb_secs.append(float(plate_status.get("ttb_sec", 60.0)))

                features.extend(
                    [
                        plate_status["distance_norm"],
                        plate_status["speed_norm"],
                        plate_status["speed_over_norm"],
                        plate_status["speed_log_norm"],      
                        plate_status["outward_vel_norm"],
                        plate_status["outward_over_norm"],
                        plate_status["outward_log_norm"],    
                        unattended_norm,
                        plate_status["tilt_mag_norm"],
                        plate_status["tilt_outward_norm"],
                        plate_status["ttb_urgency"],
                        plate_status["ttb_sec_norm10"],     
                        ever,
                    ]
                )

                mask.append(1)
            else:
                features.extend([0.0] * 13)
                mask.append(0)

        drag_norm = status["current_drag_norm"]
        dt_s = status["game_time"] - self.last_switch_commit_time

        dt_s_cap = 3.0
        dt_s_norm = min(max(dt_s / dt_s_cap, 0.0), 1.0)

        controlled_one_hot = [0.0] * N_max
        controlled_logical = status["controlled_plate"]
        if 0 <= controlled_logical < N_max:
            controlled_one_hot[controlled_logical] = 1.0

        min_dt_s = self._min_dwell_dt_s_for_N(N)
        first_switch_done = (self.last_switch_commit_time > 0.0)

        force_hold = prompt_busy

        # Action selection is gated while a pre-cue is active or within the minimum dwell window.
        if force_hold or (first_switch_done and dt_s < min_dt_s):
            gated_mask = [0] * N_max
            if 0 <= controlled_logical < N_max:
                gated_mask[controlled_logical] = 1
            mask = gated_mask

        prompt_busy_flag = 1.0 if prompt_busy else 0.0
        time_to_commit_norm = 0.0
        if time_to_commit > 0.0:
            time_to_commit_norm = min(time_to_commit / 0.3, 1.0)

        if ttb_secs:
            min_ttb = min(ttb_secs)
            min_ttb_norm10 = min(min_ttb, 10.0) / 10.0
            frac_ttb_lt2 = sum(1 for t in ttb_secs if t < 2.0) / float(max(1, N))
        else:
            min_ttb_norm10 = 1.0
            frac_ttb_lt2 = 0.0

        features.extend(controlled_one_hot)
        features.append(drag_norm)
        features.append(dt_s_norm)
        N_norm = float(N) / float(N_max) if N_max > 0 else 0.0
        features.append(N_norm)
        features.append(prompt_busy_flag)
        features.append(time_to_commit_norm)
        features.append(min_ttb_norm10)  
        features.append(frac_ttb_lt2)    

        return features, mask





class PygamePromptAPI(PromptAPI):
    def __init__(self, game: "GameInteractive"):
        self.game = game
        self._last_controlled_plate = game.core.controlled_plate

    def set_plate_background_color(self, plate_idx: int, color_name: str) -> None:
        if 0 <= plate_idx < len(self.game.plate_background_colors):
            try:
                idx = BACKGROUND_COLOR_NAMES.index(color_name)
            except ValueError:
                idx = BACKGROUND_COLOR_NAMES.index("default_background")
            self.game.plate_background_colors[plate_idx] = idx

    def set_plate_background_text(self, plate_idx: int, text: Optional[str]) -> None:
        if 0 <= plate_idx < len(self.game.plate_background_texts):
            self.game.plate_background_texts[plate_idx] = text

    def clear_plate_background(self, plate_idx: int) -> None:
        self.set_plate_background_color(plate_idx, "default_background")
        self.set_plate_background_text(plate_idx, None)

    def play_sound(self, key: str, left_vol: float = 1.0, right_vol: float = 1.0) -> None:
        self.game.play_sound(key, left_vol, right_vol)

    def set_controlled_plate(self, plate_idx: int) -> None:
        """
        Update background visuals for a control transfer.

        Resets the previous controlled plate to the default background and marks the
        new controlled plate with the active background color.
        """
        old_idx = self._last_controlled_plate

        if plate_idx == old_idx:
            return

        if 0 <= old_idx < len(self.game.plate_background_colors):
            self.set_plate_background_color(old_idx, "default_background")
            self.set_plate_background_text(old_idx, None)

        if 0 <= plate_idx < len(self.game.plate_background_colors):
            self.set_plate_background_color(plate_idx, "single_normal")
            self.set_plate_background_text(plate_idx, None)

        self._last_controlled_plate = plate_idx


    def show_precue_single_hand(self, source_idx: int, target_idx: int) -> None:
        self.set_plate_background_color(source_idx, "switch_from")
        self.set_plate_background_text(source_idx, str(target_idx + 1))

        self.set_plate_background_color(target_idx, "single_alert")
        self.set_plate_background_text(target_idx, None)

    def show_commit_single_hand(self, source_idx: int, target_idx: int) -> None:
        self.clear_plate_background(source_idx)
        self.set_plate_background_color(target_idx, "single_normal")
        self.set_plate_background_text(target_idx, None)




class MultiPlateEnv:
    def __init__(self, config: GameConfig, ams):
        assert config.mode == Mode.TRAINING
        self.config = config
        self.ams = ams

        self.core = MultiPlateCore(config.num_plates, seed=config.seed)
        self.core.reset(seed=config.seed, training=True)

        self.prompt_api = NullPromptAPI()
        self.metrics = MetricsManager(MetricsConfig(level=MetricLevel.NONE, log_to_csv=False))

        self.prompt_controller = SingleHandPromptController(
            core=self.core,
            api=self.prompt_api,
            pre_cue_duration=0.3,
            dwell_duration=0.0,
            metrics_manager=self.metrics,
            cognitive_agent=None,
        )

        params = CognitiveParams()
        self.cognitive_agent = CognitiveAgent(params, dt_micro=DT_MICRO, fps=FPS)
        self.prompt_controller.cognitive_agent = self.cognitive_agent

        self.t_now = 0.0
        self.decision_interval = 0.2
        self._episode_id = 0

        self._last_reward_controlled = self.core.controlled_plate

        self._prev_controlled_idx = self.core.controlled_plate
        self._prev_hazard_on_controlled = 0.0

        self._prev_max_hazard: float = 0.0

        self._first_visit_rewarded = [False] * self.core.num_plates
        self._first_visit_baseline_ttb = [None] * self.core.num_plates



    def reset(self, num_plates: Optional[int] = None, seed: Optional[int] = None):
        if seed is None:
            seed = self.config.seed
        if seed is not None:
            random.seed(seed)

        p = PhysicsParams(
            initial_drag=INITIAL_DRAG * random.uniform(0.95, 1.05),
            drag_decrease_amount=DRAG_DECREASE_AMOUNT * random.uniform(0.95, 1.05),
            anti_stall_speed_threshold=ANTI_STALL_SPEED_THRESHOLD * random.uniform(0.95, 1.05),
        )

        if num_plates is not None and num_plates != self.core.num_plates:
            self.core = MultiPlateCore(num_plates, seed=seed, physics_params=p)
            self.prompt_controller.core = self.core
            self.prompt_controller.state = PromptState.IDLE
            self.prompt_controller.pending_target = None
            self.prompt_controller.precue_start_time = None
            self.prompt_controller.commit_time = None
            self.prompt_controller.dwell_until = None
        else:
            self.core.params = p

        self.core.reset(seed=seed, training=True)
        perm = list(range(self.core.num_plates))
        random.shuffle(perm)
        self.core.set_logical_mapping(perm)
        self.prompt_controller.state = PromptState.IDLE
        self.prompt_controller.pending_target = None
        self.prompt_controller.precue_start_time = None
        self.prompt_controller.commit_time = None
        self.prompt_controller.dwell_until = None

        N = self.core.num_plates
        # N-scaled cognitive constraints:
        # - d_t models motor/action slip duration
        # - constant_reaction_time models visual reacquisition + planning time
        base_d_t = 0.10
        per_plate_d_t = 0.02
        self.cognitive_agent.params.d_t = min(0.25, base_d_t + per_plate_d_t * max(0, N - 2))

        base_rt = 0.00
        per_plate_rt = 0.05
        self.cognitive_agent.params.constant_reaction_time = min(0.45, base_rt + per_plate_rt * max(0, N - 2))

        self._last_reward_controlled = self.core.controlled_plate
        self._prev_controlled_idx = self.core.controlled_plate
        self._prev_hazard_on_controlled = 0.0

        max_h, _ = self._compute_plate_hazard()
        self._prev_max_hazard = float(max_h)

        self._first_visit_rewarded = [False] * self.core.num_plates
        self._first_visit_baseline_ttb = [None] * self.core.num_plates


        self.cognitive_agent.reset(num_plates=self.core.num_plates, seed=seed, core=self.core)

        self.t_now = 0.0
        self.core.state = GameState.RUNNING

        self.metrics.start_episode(
            participant_id=None,
            condition="Training",
            N_plates=self.core.num_plates,
            episode_id=self._episode_id,
            block_order=None,
            trial_index=None,
            layout_mapping_id=None,
            core=self.core,
        )
        self._episode_id += 1

        obs, _ = self.core.build_ams_obs_and_mask(self.config, prompt_busy=False, time_to_commit=0.0)
        return obs

    def _validate_ams_action(self, action: Optional[int], action_mask: List[int]) -> Optional[int]:
        """Ensure AMS actions respect mask and plate count."""
        if action is None:
            return None
        if not isinstance(action, int):
            raise ValueError(f"AMS returned non-integer action: {action!r}")
        if action < 0 or action >= len(action_mask):
            raise ValueError(f"AMS returned out-of-range action {action} with mask length {len(action_mask)}")
        if action >= self.core.num_plates:
            raise ValueError(f"AMS returned action {action} beyond active plates {self.core.num_plates}")
        if action_mask[action] == 0:
            raise ValueError(f"AMS returned masked-out action {action}")
        return action

    def step(self, action: int):
        """
        Advance the environment by one AMS decision interval (self.decision_interval seconds).

        - Physics and prompting run at 60 Hz (DT_MICRO); this method integrates them.
        - A requested switch starts a pre-cue (0.3s) and commits at the end of the pre-cue.
        - New switch requests are ignored while a pre-cue is active.
        - Reward is computed once per decision interval from timing and hazard-based shaping.
        """
        prompt_busy = (self.prompt_controller.state != PromptState.IDLE)
        time_to_commit = 0.0
        if self.prompt_controller.state == PromptState.PRE_CUE and self.prompt_controller.commit_time is not None:
            time_to_commit = max(0.0, self.prompt_controller.commit_time - self.t_now)

        _, pre_mask = self.core.build_ams_obs_and_mask(
            self.config, prompt_busy=prompt_busy, time_to_commit=time_to_commit
        )
        logical_target = self._validate_ams_action(action, pre_mask)
        if logical_target is None:
            logical_target = self.core.internal_to_logical[self.core.controlled_plate]
        internal_target = self.core.get_internal_plate_index(logical_target)
        if self.prompt_controller.state == PromptState.IDLE:
            self.prompt_controller.request_switch(internal_target, self.t_now)

        target_time = self.t_now + self.decision_interval
        done = False

        while self.t_now < target_time and not done:
            # If a pre-cue ends on this micro-step, the last control update applies to the previous plate;
            # the switch commits during prompt_controller.update(), and the next micro-step uses the new plate.
            cp_internal = self.core.controlled_plate
            cp_logical = self.core.internal_to_logical[cp_internal]
            jx, jy = self.cognitive_agent.act_for_training_control(cp_logical, self.t_now, self.core)
            self.core.plates[cp_internal].update_with_joystick(jx, jy)

            self.prompt_controller.update(self.t_now)
            alive = self.core.step_physics(DT_MICRO)
            self.metrics.on_micro_step(self.core, DT_MICRO, joystick_x=jx, joystick_y=jy)
            self.t_now += DT_MICRO
            if not alive:
                self.prompt_controller.abort_current_precue()
                done = True
                break

        prompt_busy = (self.prompt_controller.state != PromptState.IDLE)
        time_to_commit = 0.0
        if self.prompt_controller.state == PromptState.PRE_CUE and self.prompt_controller.commit_time is not None:
            time_to_commit = max(0.0, self.prompt_controller.commit_time - self.t_now)

        obs, action_mask = self.core.build_ams_obs_and_mask(
            self.config, prompt_busy=prompt_busy, time_to_commit=time_to_commit
        )
        reward = self._compute_timing_reward()

        if done:
            N = self.core.num_plates
            terminal_penalty = 1.0 * (1.0 + 0.1 * max(0, N - 2))
            reward -= terminal_penalty

        info = {"action_mask": action_mask}

        if done:
            self.metrics.end_episode(self.core, failure_plate_idx=self.core.failure_plate_idx)

        return obs, reward, done, info


    def _compute_plate_hazard(self) -> tuple[float, float]:
        """
        Compute hazard scores from the current true state.

        Returns:
            max_hazard: maximum hazard over all plates
            hazard_on_controlled: hazard for the currently controlled plate
        """
        status = self.core.get_status()
        drag_norm = status["current_drag_norm"]
        phase = 1.0 - drag_norm

        w_pos   = 0.5 - 0.2 * phase
        w_vel   = 0.25 + 0.15 * phase
        w_acc   = 0.15 + 0.15 * phase
        w_joint = 0.10 + 0.10 * phase

        max_hazard = 0.0
        hazard_on_controlled = 0.0
        N = self.core.num_plates

        for internal_idx in range(self.core.num_plates):
            plate_status = status["plates"][internal_idx]

            r = plate_status["distance_norm"]
            v = plate_status.get("speed_log_norm", plate_status["speed_norm"])
            v_out_raw = plate_status.get("outward_log_norm", plate_status["outward_vel_norm"])
            a_out_raw = plate_status["tilt_outward_norm"]
            u = plate_status["unattended_time_norm"]
            ttb_sec = plate_status.get("ttb_sec", 60.0)
            ttb_urg = plate_status.get("ttb_urgency", 0.0)


            v_out = max(v_out_raw, 0.0)
            a_out = max(a_out_raw, 0.0)

            H_pos = r ** 2

            H_vel = v_out * (r ** 2)

            H_acc = v_out * a_out

            H_joint = (r ** 3) * v_out * a_out

            H_unset = 0.1 * v

            H_raw = (
                w_pos   * H_pos +
                w_vel   * H_vel +
                w_acc   * H_acc +
                w_joint * H_joint +
                H_unset
            )

            H_raw += 0.20 * ttb_urg

            bump = 1.0 / (1.0 + math.exp(4.0 * (float(ttb_sec) - 1.0)))
            H_raw += 1.0 * bump

            extreme_outward = (r > 0.9) and (a_out_raw > 0.0) and (v_out_raw > -0.05)
            if extreme_outward:
                H_raw += 1.0

            braking = (v_out_raw < 0.0) and (a_out_raw < 0.0)
            if braking:
                if r < 0.9:
                    H_raw *= 0.3
                else:
                    H_raw *= 0.6

            neglect_factor = 1.0 + 2.0 * (u ** 2)

            H = H_raw * neglect_factor

            if internal_idx == self.core.controlled_plate:
                hazard_on_controlled = H

            if H > max_hazard:
                max_hazard = H

        return max_hazard, hazard_on_controlled


    def _compute_timing_reward(self) -> float:
        """
        Reward terms:
          - Timing term (anti-flicker / anti-camping)
          - Global hazard penalty (worst plate)
          - Stabilization bonus on the controlled plate
          - First-visit / coverage shaping
          - Neglect, overstay, and switch-cost shaping
          - Survival bonus (terminal penalty applied in step())
        """
        N = self.core.num_plates

        if not hasattr(self, "_prev_max_hazard"):
            max_h, _ = self._compute_plate_hazard()
            self._prev_max_hazard = float(max_h)

        dt_s = self.t_now - self.core.last_switch_commit_time
        t_d = self.decision_interval

        min_dt = self.core._min_dwell_dt_s_for_N(N) if hasattr(self.core, "_min_dwell_dt_s_for_N") else 0.5

        if N <= 2:
            max_dt = 2.5
        elif N <= 5:
            max_dt = 2.0
        elif N == 6:
            max_dt = 1.8
        else:
            max_dt = 1.7

        k = math.exp(2.0)
        sig_lo = 1.0 / (1.0 + math.exp(-k * (dt_s - min_dt)))
        sig_hi = 1.0 / (1.0 + math.exp(-k * (max_dt - dt_s)))
        timing_reward = t_d * sig_lo * sig_hi

        max_hazard, hazard_on_controlled = self._compute_plate_hazard()
        current = self.core.controlled_plate
        status = self.core.get_status()

        survival_bonus = 0.003 * (1.0 + 0.1 * max(0, N - 2))

        λ_global = 0.8 if N <= 3 else 1.0

        delta_global = self._prev_max_hazard - max_hazard
        progress_bonus = 0.0
        if delta_global > 0.0:
            progress_bonus = 0.08 * delta_global
        self._prev_max_hazard = max_hazard

        reward = survival_bonus + timing_reward + progress_bonus - λ_global * max_hazard
        reward += 0.10

        ttb_list_internal = [float(p.get("ttb_sec", 60.0)) for p in status["plates"]]
        if ttb_list_internal:
            min_ttb = min(ttb_list_internal)
            argmin_internal = int(ttb_list_internal.index(min_ttb))
        else:
            min_ttb = 60.0
            argmin_internal = current

        prompt_busy_now = (self.prompt_controller.state != PromptState.IDLE)
        first_switch_done = (self.core.last_switch_commit_time > 0.0)
        min_dt_now = self.core._min_dwell_dt_s_for_N(N)
        switch_allowed_now = (not prompt_busy_now) and ((not first_switch_done) or (dt_s >= min_dt_now))

        EMG_TTB = 2.0
        if min_ttb < EMG_TTB:
            severity = max(0.0, (EMG_TTB - min_ttb) / EMG_TTB)

            reward -= 0.40 * severity

            if current != argmin_internal:
                gate_factor = 1.0 if switch_allowed_now else 0.20
                reward -= gate_factor * (0.90 * severity)

            if min_ttb < 1.0:
                reward -= 0.30 * max(0.0, (1.0 - min_ttb))

        triage_margin = 0.08
        triage_gap = max(0.0, max_hazard - hazard_on_controlled - triage_margin)

        hazard_cap = 2.0
        crisis_level = min(1.0, max_hazard / hazard_cap)

        triage_weight = (0.06 + 0.06 * max(0, N - 2) / float(N)) * (0.5 + 0.5 * crisis_level)
        reward -= triage_weight * triage_gap

        scale_N = max(0, N - 2) / float(N)

        drag_norm = float(status.get("current_drag_norm", 1.0))

        ttb_list = [float(p.get("ttb_sec", 60.0)) for p in status["plates"]]
        min_ttb = min(ttb_list) if ttb_list else 60.0

        T_precue = self.prompt_controller.pre_cue_duration
        dt_slip = getattr(self.cognitive_agent.params, "d_t", 0.25) if self.cognitive_agent is not None else 0.25
        dt_react = getattr(self.cognitive_agent.params, "constant_reaction_time", 0.0) if self.cognitive_agent is not None else 0.0
        dt_reacq = max(float(dt_slip), float(dt_react))
        T_assign = T_precue + dt_reacq + 0.05

        sweep_allowed = (drag_norm > 0.60) and (min_ttb > (T_assign + 0.50))

        safe_thresh_coverage = 0.15
        safe_thresh_neglect  = 0.25
        current_safe_for_coverage = (hazard_on_controlled < safe_thresh_coverage)
        current_safe_for_neglect  = (hazard_on_controlled < safe_thresh_neglect)

        max_unattended = max(p["unattended_time_norm"] for p in status["plates"])
        if scale_N > 0.0 and max_unattended > 0.0 and current_safe_for_neglect:
            neglect_penalty = 0.03 * max_unattended * scale_N
            reward -= neglect_penalty

        if scale_N > 0.0 and (current_safe_for_coverage or sweep_allowed):
            unvisited_indices = [
                i for i in range(self.core.num_plates)
                if not self.core.ever_controlled[i]
            ]
            if unvisited_indices:
                max_unvisited_age_norm = max(
                    min(self.core.unattended_times[i] / U_MAX, 1.0)
                    for i in unvisited_indices
                )
                coverage_penalty = 0.04 * max_unvisited_age_norm * scale_N
                reward -= coverage_penalty

        if current == self._prev_controlled_idx:
            delta_h = self._prev_hazard_on_controlled - hazard_on_controlled
            if delta_h > 0.02:
                reward += 0.05 * delta_h

        if 0 <= current < len(self._first_visit_rewarded):
            cur_ttb = status["plates"][current].get("ttb_sec", 60.0)

            if self._first_visit_baseline_ttb[current] is None:
                self._first_visit_baseline_ttb[current] = cur_ttb

            if not self._first_visit_rewarded[current]:
                base_ttb = self._first_visit_baseline_ttb[current]
                if base_ttb is not None:
                    min_dt_s = self.core._min_dwell_dt_s_for_N(N)
                    if dt_s >= min_dt_s:
                        thresh = max(1.0, 0.15 * base_ttb)
                        if cur_ttb >= base_ttb + thresh:
                            sweep_bonus = 0.05 * (1.0 + 0.1 * max(0, N - 2))
                            reward += sweep_bonus
                            self._first_visit_rewarded[current] = True


        safe_thresh = 0.15
        danger_margin = 0.10
        if scale_N > 0.0 and hazard_on_controlled < safe_thresh:
            if max_hazard > hazard_on_controlled + danger_margin:
                overstay_penalty = 0.02 * scale_N
                reward -= overstay_penalty

        if current != self._last_reward_controlled:
            base_switch_penalty = 0.02 * (1.0 + 0.05 * max(0, N - 2))

            hazard_cap = 2.0
            crisis_level = min(1.0, max_hazard / hazard_cap)

            switch_penalty = base_switch_penalty * (1.0 - 0.70 * crisis_level)
            reward -= switch_penalty

            src_internal = self._last_reward_controlled
            if 0 <= src_internal < self.core.num_plates:
                src_status = self.core.get_status_for_plate(src_internal)
                src_v_out = max(src_status["outward_vel_norm"], 0.0)
                src_speed = src_status["speed_norm"]

                ttb_list = [p.get("ttb_sec", 60.0) for p in status["plates"]]
                min_ttb = min(ttb_list) if ttb_list else 60.0
                max_ttb = max(ttb_list) if ttb_list else 60.0

                T_precue = self.prompt_controller.pre_cue_duration
                dt_slip = getattr(self.cognitive_agent.params, "d_t", 0.25) if self.cognitive_agent is not None else 0.25
                dt_react = getattr(self.cognitive_agent.params, "constant_reaction_time", 0.0) if self.cognitive_agent is not None else 0.0
                dt_reacq = max(float(dt_slip), float(dt_react))
                T_assign = T_precue + dt_reacq + 0.05

                T_return = (N - 1) * (self.core._min_dwell_dt_s_for_N(N) + T_precue) + dt_reacq

                hazard_cap = 2.0
                crisis_level = min(1.0, max_hazard / hazard_cap)
                triage_relief = 0.35 if (min_ttb < T_assign) else 1.0
                triage_relief *= (1.0 - 0.50 * crisis_level)
                triage_relief = max(0.20, triage_relief)

                src_ttb = src_status.get("ttb_sec", 60.0)
                if src_ttb < T_return and T_return > 1e-6:
                    severity = max(0.0, 1.0 - (src_ttb / T_return))
                    reward -= triage_relief * (0.12 * severity)

                tgt_internal = current
                if 0 <= tgt_internal < self.core.num_plates:
                    tgt_status = self.core.get_status_for_plate(tgt_internal)
                    tgt_ttb = tgt_status.get("ttb_sec", 60.0)

                    if tgt_ttb < T_assign and (max_ttb > T_assign + 0.5) and T_assign > 1e-6:
                        severity = max(0.0, 1.0 - (tgt_ttb / T_assign))
                        reward -= triage_relief * (0.15 * severity)


                if src_v_out > 0.1 or src_speed > 0.4:
                    unsafe = src_v_out + 0.5 * src_speed
                    reward -= triage_relief * (0.2 * unsafe)

                tgt_internal = current
                if 0 <= tgt_internal < self.core.num_plates:
                    tgt_status = self.core.get_status_for_plate(tgt_internal)
                    src_R = 0.5 * src_status["distance_norm"] + 0.5 * src_status["speed_norm"]
                    tgt_R = 0.5 * tgt_status["distance_norm"] + 0.5 * tgt_status["speed_norm"]
                    margin = 0.05
                    if tgt_R + margin < src_R:
                        reward -= 0.05 * (src_R - tgt_R)

        self._last_reward_controlled = current
        self._prev_controlled_idx = current
        self._prev_hazard_on_controlled = hazard_on_controlled

        return reward






def derive_perm_from_seed(seed: int, num_plates: int) -> List[int]:
    rng = random.Random(seed)
    perm = list(range(num_plates))
    rng.shuffle(perm)
    return perm


def compute_layout_mapping_id(perm: List[int], num_plates: int, participant_id: Optional[int]) -> str:
    payload = {"participant_id": participant_id, "num_plates": num_plates, "perm": perm}
    s = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(s).hexdigest()[:16]


def write_layout_sidecar(
    layout_mapping_id: str,
    perm: List[int],
    num_plates: int,
    participant_id: Optional[int],
    out_dir: Optional[str] = None,
    plate_grid=None,
) -> None:

    if out_dir is None or not str(out_dir).strip():
        out_dir = os.path.join(os.path.dirname(__file__), "PlayLogs")

    os.makedirs(out_dir, exist_ok=True)

    index_to_rc = {}
    if plate_grid is not None:
        for r, row in enumerate(plate_grid):
            for c, internal_idx in enumerate(row):
                index_to_rc[internal_idx] = (r + 1, c + 1)

    plates = []
    for logical_idx, internal_idx in enumerate(perm):
        row_col = index_to_rc.get(internal_idx, (None, None))
        plates.append(
            {
                "logical_idx": logical_idx,
                "internal_idx": internal_idx,
                "row": row_col[0],
                "col": row_col[1],
            }
        )

    sidecar = {
        "layout_mapping_id": layout_mapping_id,
        "participant_id": participant_id,
        "num_plates": num_plates,
        "perm": perm,
        "plates": plates,
    }

    path = os.path.join(out_dir, f"layout_map_{layout_mapping_id}.json")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2)



class GameInteractive:
    def __init__(self, config: GameConfig, ams=None, metrics_config: Optional[MetricsConfig] = None):
        self.config = config
        self.ams = ams
        self._block_order = config.block_order
        self._trial_index = config.trial_index

        if self.config.mode in (Mode.HUMAN_BASELINE, Mode.AMS_PLAY):
            self.participant_id = self.config.participant_id if self.config.participant_id is not None else 0
        else:
            self.participant_id = self.config.participant_id

        pygame.init()
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        pygame.mixer.set_num_channels(32)
        pygame.joystick.init()

        self.info = pygame.display.Info()
        self.WINDOW_WIDTH = self.info.current_w
        self.WINDOW_HEIGHT = self.info.current_h
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        pygame.display.set_caption("Multi-Plate Balance Ball Game - Single Hand")
        self.clock = pygame.time.Clock()

        self.gamepad = None
        self.gamepad_name = None
        self._init_gamepad()

        self.single_hand_mode = SINGLE_HAND_MODE

        self.prev_button_states = {}
        self.current_button_states = {}

        self.core = MultiPlateCore(config.num_plates, seed=config.seed)
        self.core.reset(seed=config.seed)

        self.layout = self._get_plate_layout(self.core.num_plates)
        self.grid_rows = len(self.layout)
        self.grid_cols = max(len(row) for row in self.layout)

        self.h_margin = int(self.WINDOW_WIDTH * GRID_HORIZONTAL_MARGIN_PERCENT)
        self.left_margin = int(self.WINDOW_WIDTH * GRID_LEFT_MARGIN_PERCENT)
        self.v_margin = int(self.WINDOW_HEIGHT * GRID_VERTICAL_MARGIN_PERCENT)

        total_h_space = self.WINDOW_WIDTH - self.left_margin - 4 * self.h_margin
        self.plate_width = total_h_space // 4
        self.plate_height = (self.WINDOW_HEIGHT - 4 * self.v_margin) // 3

        self.plate_radius = min(self.plate_width, self.plate_height) // 2
        self.base_display_scale = self.plate_radius / PLATE_DISPLAY_RADIUS
        self.displayed_radius = int(self.plate_radius * PLATE_DISPLAY_SCALE_FACTOR)
        self.display_scale = self.base_display_scale * PLATE_DISPLAY_SCALE_FACTOR
        self.physics_to_display_scale = self.display_scale * (PLATE_DISPLAY_RADIUS / PLATE_RADIUS)

        grid_width = self.grid_cols * self.plate_width + (self.grid_cols - 1) * self.h_margin
        grid_height = self.grid_rows * self.plate_height + (self.grid_rows - 1) * self.v_margin
        self.grid_offset_x = (self.WINDOW_WIDTH - grid_width) // 2
        self.grid_offset_y = (self.WINDOW_HEIGHT - grid_height) // 2

        self.plate_positions: List[Tuple[int, int]] = []
        self.plate_background_colors: List[int] = []
        self.plate_background_texts: List[Optional[str]] = []
        self.plate_grid = []
        plate_index = 0
        for row_idx, row in enumerate(self.layout):
            grid_row = []
            for col_idx in range(len(row)):
                if row[col_idx]:
                    grid_row.append(plate_index)
                    center_x = self.grid_offset_x + col_idx * (self.plate_width + self.h_margin) + self.plate_width // 2
                    center_y = self.grid_offset_y + row_idx * (self.plate_height + self.v_margin) + self.plate_height // 2
                    self.plate_positions.append((center_x, center_y))
                    self.plate_background_colors.append(0)
                    self.plate_background_texts.append(None)
                    plate_index += 1
            if grid_row:
                self.plate_grid.append(grid_row)

        self.prompt_api = PygamePromptAPI(self)

        if metrics_config is None:
            if config.mode in (Mode.HUMAN_BASELINE, Mode.AMS_PLAY):
                metrics_level = MetricLevel.XAI
                log_to_csv = True
            else:
                metrics_level = MetricLevel.NONE
                log_to_csv = False

            if getattr(config, "log_dir", None):
                log_dir = str(config.log_dir)
                episodes_path = os.path.join(log_dir, "episodes.csv")
                switches_path = os.path.join(log_dir, "switches.csv")
                metrics_config = MetricsConfig(
                    level=metrics_level,
                    log_to_csv=log_to_csv,
                    episodes_csv_path=episodes_path,
                    switches_csv_path=switches_path,
                )
            else:
                metrics_config = MetricsConfig(level=metrics_level, log_to_csv=log_to_csv)


        if config.mode in (Mode.HUMAN_BASELINE, Mode.AMS_PLAY):
            base_seed = self.participant_id if self.participant_id is not None else (config.seed or 0)
            perm = derive_perm_from_seed(base_seed, num_plates=self.core.num_plates)
            self.core.set_logical_mapping(perm)
            self.layout_mapping_id = compute_layout_mapping_id(
                perm, self.core.num_plates, participant_id=self.participant_id
            )
            write_layout_sidecar(
                layout_mapping_id=self.layout_mapping_id,
                perm=perm,
                num_plates=self.core.num_plates,
                participant_id=self.participant_id,
                out_dir=self.config.log_dir,
                plate_grid=self.plate_grid,
            )
        else:
            self.layout_mapping_id = None

        # Playground modes disable all logging.
        if self.config.mode in (Mode.PLAYGROUND_AMS, Mode.PLAYGROUND_NO_AMS):
            metrics_config.log_to_csv = False
            metrics_config.level = MetricLevel.NONE

        self.metrics = MetricsManager(metrics_config)
        require_o = config.mode in (Mode.HUMAN_BASELINE, Mode.AMS_PLAY)
        self.logger = EpisodeLogger(metrics_config, require_o_confirmation=require_o)

        self._decision_trace_file = None
        self._decision_trace_writer = None
        self._decision_idx = 0
        self._decision_trace_path = None

        if (
            self.config.decision_trace
            and self.logger.config.log_to_csv
            and self.config.mode in (Mode.AMS_PLAY, Mode.HUMAN_BASELINE)
        ):
            self._init_decision_trace_logger()

        self.prompt_controller = SingleHandPromptController(
            core=self.core,
            api=self.prompt_api,
            pre_cue_duration=0.3,
            dwell_duration=0.0,
            metrics_manager=self.metrics,
            cognitive_agent=None,
        )

        self.cognitive_agent = None
        if self.config.display_belief and self.config.mode in (Mode.AMS_PLAY, Mode.PLAYGROUND_AMS):
            params = CognitiveParams()
            self.cognitive_agent = CognitiveAgent(params, dt_micro=DT_MICRO, fps=FPS)

            N = self.core.num_plates

            base_d_t = 0.10
            per_plate_d_t = 0.02
            self.cognitive_agent.params.d_t = min(0.25, base_d_t + per_plate_d_t * max(0, N - 2))

            base_rt = 0.00
            per_plate_rt = 0.05
            self.cognitive_agent.params.constant_reaction_time = min(0.45, base_rt + per_plate_rt * max(0, N - 2))

            self.prompt_controller.cognitive_agent = self.cognitive_agent

        self.font = pygame.font.Font(None, STATUS_FONT_SIZE)
        self.large_font = pygame.font.Font(None, MESSAGE_FONT_SIZE)
        self.plate_font = pygame.font.Font(None, PLATE_NUMBER_FONT_SIZE)
        self.background_font = pygame.font.Font(None, BACKGROUND_TEXT_FONT_SIZE)
        self.coordinate_font = pygame.font.Font(None, int(BACKGROUND_TEXT_FONT_SIZE * 0.4))

        self.sounds = self._load_sounds()
        self.state = GameState.NOT_STARTED
        self.t_now = 0.0
        self._physics_accumulator = 0.0
        self.ams_decision_interval = 0.2
        self.next_ams_decision_time = 0.0
        self._episode_id = 0

        self.apply_visual_prompts()
        self._current_condition_name: Optional[str] = None
        self._game_over_save_message: str = "Press O to log this episode"
        self._game_over_saved_count: int = 0

    def _init_decision_trace_logger(self) -> None:
        log_dir = os.path.dirname(self.logger.config.episodes_csv_path) or "."
        os.makedirs(log_dir, exist_ok=True)
        self._decision_trace_path = os.path.join(log_dir, "decision_trace.csv")

        N_max = self.config.N_max
        fieldnames = [
            "run_stamp",
            "participant_id",
            "condition",
            "episode_id",
            "episode_uid",
            "decision_idx",
            "t_now_sec",
            "controlled_plate_logical",

            "chosen_action_logical",

            "policy_action_logical",

            "override_applied",

            "top1_action_logical",
            "top1_prob",
            "top2_action_logical",
            "top2_prob",
            "prob_argmin_ttb",
            "prob_policy_action",
            "prob_chosen_action",
            "policy_entropy",

            "valid_action_count",
            "prompt_busy",
            "time_to_commit_sec",
            "min_ttb_sec",
            "argmin_ttb_plate",
            "count_ttb_lt1",
            "count_ttb_lt2",
            "max_hazard_proxy",
            "argmax_hazard_plate",
            "hazard_on_controlled",
        ]

        for i in range(N_max):
            fieldnames.append(f"ttb_sec_{i}")
        for i in range(N_max):
            fieldnames.append(f"hazard_{i}")

        _ensure_csv_header(self._decision_trace_path, fieldnames)

        is_new = (not os.path.exists(self._decision_trace_path)) or (os.path.getsize(self._decision_trace_path) == 0)
        self._decision_trace_file = open(self._decision_trace_path, "a", newline="", encoding="utf-8")

        self._decision_trace_writer = csv.DictWriter(self._decision_trace_file, fieldnames=fieldnames)
        if is_new:
            self._decision_trace_writer.writeheader()
            self._decision_trace_file.flush()

        self._run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _compute_hazard_proxy_for_logical_plate(self, logical_i: int, drag_norm: float) -> float:
        """
        A lightweight hazard proxy aligned with the training hazard structure.
        Uses per-plate status features already computed in MultiPlateCore.
        """
        internal_i = self.core.get_internal_plate_index(logical_i)
        st = self.core.get_status_for_plate(internal_i)

        r = float(st.get("distance_norm", 0.0))
        v = float(st.get("speed_log_norm", st.get("speed_norm", 0.0)))
        v_out_raw = float(st.get("outward_log_norm", st.get("outward_vel_norm", 0.0)))
        a_out_raw = float(st.get("tilt_outward_norm", 0.0))
        u = float(st.get("unattended_time_norm", 0.0))
        ttb_sec = float(st.get("ttb_sec", 60.0))
        ttb_urg = float(st.get("ttb_urgency", 0.0))

        phase = 1.0 - float(drag_norm)
        w_pos   = 0.5 - 0.2 * phase
        w_vel   = 0.25 + 0.15 * phase
        w_acc   = 0.15 + 0.15 * phase
        w_joint = 0.10 + 0.10 * phase

        v_out = max(v_out_raw, 0.0)
        a_out = max(a_out_raw, 0.0)

        H_pos = r ** 2
        H_vel = v_out * (r ** 2)
        H_acc = v_out * a_out
        H_joint = (r ** 3) * v_out * a_out
        H_unset = 0.1 * v

        H_raw = w_pos*H_pos + w_vel*H_vel + w_acc*H_acc + w_joint*H_joint + H_unset

        H_raw += 0.20 * ttb_urg

        bump = 1.0 / (1.0 + math.exp(4.0 * (ttb_sec - 1.0)))
        H_raw += 1.0 * bump

        neglect_factor = 1.0 + 2.0 * (u ** 2)
        return float(H_raw * neglect_factor)

    def _log_decision_trace(
        self,
        t_now: float,
        obs,
        action_mask,
        chosen_action: int,
        prompt_busy: bool,
        time_to_commit: float,
        policy_action: Optional[int] = None,
        override_applied: bool = False,
    ) -> None:
        if self._decision_trace_writer is None:
            return
        if self.metrics.current_episode is None:
            return

        ep = self.metrics.current_episode
        N = int(ep.N_plates)
        N_max = int(self.config.N_max)

        controlled_logical = int(self.core.internal_to_logical[self.core.controlled_plate])

        valid_count = int(sum(1 for x in action_mask if int(x) == 1))

        status = self.core.get_status()
        drag_norm = float(status.get("current_drag_norm", 1.0))

        ttb_list = [np.nan] * N_max
        hazard_list = [np.nan] * N_max

        active_ttb = []
        active_hz = []

        for logical_i in range(N):
            internal_i = self.core.get_internal_plate_index(logical_i)
            st = self.core.get_status_for_plate(internal_i)
            ttb = float(st.get("ttb_sec", 60.0))
            hz = self._compute_hazard_proxy_for_logical_plate(logical_i, drag_norm=drag_norm)

            ttb_list[logical_i] = ttb
            hazard_list[logical_i] = hz
            active_ttb.append((ttb, logical_i))
            active_hz.append((hz, logical_i))

        min_ttb_sec, argmin_ttb_plate = min(active_ttb, key=lambda x: x[0]) if active_ttb else (np.nan, np.nan)
        max_hz, argmax_hz_plate = max(active_hz, key=lambda x: x[0]) if active_hz else (np.nan, np.nan)

        count_lt1 = sum(1 for (ttb, _) in active_ttb if ttb < 1.0)
        count_lt2 = sum(1 for (ttb, _) in active_ttb if ttb < 2.0)

        hazard_on_ctrl = hazard_list[controlled_logical] if 0 <= controlled_logical < N_max else np.nan

        probs = None
        top1_a = ""
        top1_p = ""
        top2_a = ""
        top2_p = ""
        prob_argmin = ""
        prob_policy = ""
        prob_chosen = ""
        entropy = ""

        if self.ams is not None and hasattr(self.ams, "last_action_probs"):
            probs = getattr(self.ams, "last_action_probs", None)

        if isinstance(probs, list) and len(probs) >= N_max:
            p = np.array(probs[:N_max], dtype=np.float64)

            top_idx = np.argsort(p)[::-1][:2]
            if len(top_idx) >= 1:
                top1_a = int(top_idx[0])
                top1_p = float(p[top1_a])
            if len(top_idx) >= 2:
                top2_a = int(top_idx[1])
                top2_p = float(p[top2_a])

            pp = np.clip(p, 1e-12, 1.0)
            entropy = float(-np.sum(pp * np.log(pp)))

            if not np.isnan(argmin_ttb_plate):
                ai = int(argmin_ttb_plate)
                if 0 <= ai < N_max:
                    prob_argmin = float(p[ai])

            if policy_action is not None and 0 <= int(policy_action) < N_max:
                prob_policy = float(p[int(policy_action)])

            if chosen_action is not None and 0 <= int(chosen_action) < N_max:
                prob_chosen = float(p[int(chosen_action)])

        row = {
            "run_stamp": getattr(self, "_run_stamp", ""),
            "participant_id": ep.participant_id,
            "condition": ep.condition,
            "episode_id": ep.episode_id,
            "episode_uid": ep.episode_uid,
            "decision_idx": self._decision_idx,
            "t_now_sec": float(t_now),
            "controlled_plate_logical": controlled_logical,
            "chosen_action_logical": int(chosen_action) if chosen_action is not None else -1,
            "policy_action_logical": int(policy_action) if policy_action is not None else -1,
            "override_applied": int(bool(override_applied)),

            "top1_action_logical": top1_a,
            "top1_prob": top1_p,
            "top2_action_logical": top2_a,
            "top2_prob": top2_p,
            "prob_argmin_ttb": prob_argmin,
            "prob_policy_action": prob_policy,
            "prob_chosen_action": prob_chosen,
            "policy_entropy": entropy,

            "valid_action_count": valid_count,
            "prompt_busy": int(bool(prompt_busy)),
            "time_to_commit_sec": float(time_to_commit),
            "min_ttb_sec": float(min_ttb_sec),
            "argmin_ttb_plate": int(argmin_ttb_plate) if argmin_ttb_plate is not np.nan else "",
            "count_ttb_lt1": int(count_lt1),
            "count_ttb_lt2": int(count_lt2),
            "max_hazard_proxy": float(max_hz),
            "argmax_hazard_plate": int(argmax_hz_plate) if argmax_hz_plate is not np.nan else "",
            "hazard_on_controlled": float(hazard_on_ctrl) if hazard_on_ctrl is not np.nan else "",
        }

        for i in range(N_max):
            row[f"ttb_sec_{i}"] = "" if np.isnan(ttb_list[i]) else float(ttb_list[i])
        for i in range(N_max):
            row[f"hazard_{i}"] = "" if np.isnan(hazard_list[i]) else float(hazard_list[i])

        self._decision_trace_writer.writerow(row)
        self._decision_trace_file.flush()

    def _init_gamepad(self) -> None:
        self.gamepad = None
        self.gamepad_name = None
        if pygame.joystick.get_count() > 0:
            self.gamepad = pygame.joystick.Joystick(0)
            self.gamepad.init()
            self.gamepad_name = self.gamepad.get_name()

    def _check_gamepad_connection(self) -> None:
        current_count = pygame.joystick.get_count()
        if current_count == 0 and self.gamepad is not None:
            self.gamepad = None
            self.gamepad_name = None
        elif current_count > 0 and self.gamepad is None:
            self._init_gamepad()

    def _update_button_states(self) -> None:
        self.prev_button_states = self.current_button_states.copy()
        self.current_button_states = {}
        if self.gamepad:
            for i in range(self.gamepad.get_numbuttons()):
                self.current_button_states[i] = self.gamepad.get_button(i)
            if self.gamepad.get_numbuttons() >= 15:
                self.current_button_states["dpad_up"] = self.gamepad.get_button(11)
                self.current_button_states["dpad_down"] = self.gamepad.get_button(12)
                self.current_button_states["dpad_left"] = self.gamepad.get_button(13)
                self.current_button_states["dpad_right"] = self.gamepad.get_button(14)
            elif self.gamepad.get_numhats() > 0:
                hat = self.gamepad.get_hat(0)
                self.current_button_states["dpad_up"] = hat[1] == 1
                self.current_button_states["dpad_down"] = hat[1] == -1
                self.current_button_states["dpad_left"] = hat[0] == -1
                self.current_button_states["dpad_right"] = hat[0] == 1

            # Triggers are read from axes 4 (L2) and 5 (R2); axes 2/3 are reserved for the right stick.
            num_axes = self.gamepad.get_numaxes()
            l2_axis = 0.0
            r2_axis = 0.0

            if num_axes > 4:
                l2_axis = self.gamepad.get_axis(4)
            if num_axes > 5:
                r2_axis = self.gamepad.get_axis(5)

            l2_val = (l2_axis + 1.0) / 2.0
            r2_val = (r2_axis + 1.0) / 2.0

            self.current_button_states["L2_trigger"] = l2_val > 0.5
            self.current_button_states["R2_trigger"] = r2_val > 0.5


    def _get_button_pressed(self, button_id) -> bool:
        if button_id not in self.current_button_states:
            return False
        current = self.current_button_states.get(button_id, False)
        prev = self.prev_button_states.get(button_id, False)
        return current and not prev

    def _read_joystick_axes(self) -> Tuple[float, float]:
        if self.gamepad:
            lx = self.gamepad.get_axis(0) if self.gamepad.get_numaxes() > 0 else 0.0
            ly = self.gamepad.get_axis(1) if self.gamepad.get_numaxes() > 1 else 0.0
            rx = 0.0
            ry = 0.0
            if self.gamepad.get_numaxes() > 3:
                rx = self.gamepad.get_axis(2)
                ry = self.gamepad.get_axis(3)
            raw_x = lx + rx
            raw_y = ly + ry
            clamped_x = max(-1.0, min(1.0, raw_x))
            clamped_y = max(-1.0, min(1.0, raw_y))
            return (clamped_x, clamped_y)
        keys = pygame.key.get_pressed()
        x_axis = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * 0.8
        y_axis = (keys[pygame.K_DOWN] - keys[pygame.K_UP]) * 0.8
        return (x_axis, y_axis)

    def _condition_for_mode(self) -> str:
        return "Baseline" if self.config.mode == Mode.HUMAN_BASELINE else "AMS-Assisted"

    def _compute_saved_episode_count(self) -> int:
        if not self.logger.config.log_to_csv:
            return 0
        path = self.logger.config.episodes_csv_path
        if not os.path.exists(path):
            return 0

        target_pid = self.participant_id if self.participant_id is not None else 0
        target_condition = self._current_condition_name or self._condition_for_mode()
        count = 0
        try:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        row_pid = int(row.get("participant_id", "") or 0)
                        row_n = int(row.get("N_plates", "") or 0)
                    except ValueError:
                        continue
                    row_condition = row.get("condition", "")
                    if row_pid == target_pid and row_n == self.core.num_plates and row_condition == target_condition:
                        count += 1
        except FileNotFoundError:
            return 0
        return count

    def _reset_game_over_save_ui(self) -> None:
        self._game_over_save_message = "Press O to log this episode"
        self._game_over_saved_count = self._compute_saved_episode_count()

    def _start_new_episode_if_needed(self) -> None:
        self.t_now = 0.0
        self._physics_accumulator = 0.0
        self.next_ams_decision_time = 0.0
        self.apply_visual_prompts()
        self._current_condition_name = self._condition_for_mode()
        self._reset_game_over_save_ui()
        self._decision_idx = 0

        if self.cognitive_agent is not None:
            self.cognitive_agent.reset(num_plates=self.core.num_plates, core=self.core)

        if self.metrics.config.level == MetricLevel.NONE:
            return

        block_order = getattr(self, "_block_order", None)
        trial_index = getattr(self, "_trial_index", None)

        self.metrics.start_episode(
            participant_id=self.participant_id,
            condition=self._current_condition_name,
            N_plates=self.core.num_plates,
            episode_id=self._episode_id,
            block_order=block_order,
            trial_index=trial_index,
            layout_mapping_id=self.layout_mapping_id,
            core=self.core,
        )

    def _load_sounds(self) -> Dict[str, pygame.mixer.Sound]:
        sounds = {}
        sound_folder = os.path.join(os.path.dirname(__file__), "sounds")
        sound_files = {
            "single_switch": "single_switch.wav",
            "mode_switch": "mode_switch.wav",
            "left_switch": "left_switch.wav",
            "right_switch": "right_switch.wav",
        }
        for key, filename in sound_files.items():
            filepath = os.path.join(sound_folder, filename)
            if os.path.exists(filepath):
                try:
                    snd = pygame.mixer.Sound(filepath)
                    snd.set_volume(VOLUME_TUNING)
                    sounds[key] = snd
                except Exception as e:
                    print(f"Warning: Could not load sound {filepath}: {e}")
            else:
                print(f"Warning: Sound file {filepath} not found")
        return sounds


    def apply_visual_prompts(self) -> None:
        """Reset background visuals and highlight the controlled plate."""
        for i in range(len(self.plate_background_colors)):
            self.plate_background_colors[i] = BACKGROUND_COLOR_NAMES.index("default_background")
            self.plate_background_texts[i] = None
        
        if 0 <= self.core.controlled_plate < len(self.plate_background_colors):
            self.plate_background_colors[self.core.controlled_plate] = BACKGROUND_COLOR_NAMES.index("single_normal")

        # Keep PromptAPI state consistent with the current controlled plate.
        if isinstance(self.prompt_api, PygamePromptAPI):
            self.prompt_api._last_controlled_plate = self.core.controlled_plate


    def play_sound(self, key: str, left_vol: float = 1.0, right_vol: float = 1.0) -> None:
        snd = self.sounds.get(key)
        if snd is None:
            return

        l = max(0.0, min(1.0, left_vol * VOLUME_TUNING))
        r = max(0.0, min(1.0, right_vol * VOLUME_TUNING))

        channel = pygame.mixer.find_channel()
        if channel is not None:
            channel.set_volume(l, r)
            channel.play(snd)


    def _get_plate_layout(self, num_plates: int):
        layouts = {
            2: [[True, True]],
            3: [[True, True, True]],
            4: [[True, True, True, True]],
            5: [[True, True, True], [True, True]],
            6: [[True, True, True], [True, True, True]],
            7: [[True, True, True, True], [True, True, True]],
            8: [[True, True, True, True], [True, True, True, True]],
            9: [[True, True, True], [True, True, True], [True, True, True]],
            10: [[True, True, True, True], [True, True, True], [True, True, True]],
            11: [[True, True, True, True], [True, True, True, True], [True, True, True]],
            12: [[True, True, True, True], [True, True, True, True], [True, True, True, True]],
        }
        return layouts.get(num_plates, layouts[4])

    def get_plate_grid_position(self, internal_idx: int) -> Tuple[int, int]:
        for r, row in enumerate(self.plate_grid):
            for c, val in enumerate(row):
                if val == internal_idx:
                    return r, c
        return (0, 0)

    def _validate_ams_action(self, action: Optional[int], action_mask: List[int]) -> Optional[int]:
        if action is None:
            return None
        if not isinstance(action, int):
            raise ValueError(f"AMS returned non-integer action: {action!r}")
        if action < 0 or action >= len(action_mask):
            raise ValueError(f"AMS returned out-of-range action {action} with mask length {len(action_mask)}")
        if action >= self.core.num_plates:
            raise ValueError(f"AMS returned action {action} beyond active plates {self.core.num_plates}")
        if action_mask[action] == 0:
            raise ValueError(f"AMS returned masked-out action {action}")
        return action

    def _handle_keydown(self, key) -> None:
        if key == pygame.K_BACKSPACE:
            if self.state == GameState.NOT_STARTED:
                self._start_new_episode_if_needed()
                self.state = GameState.RUNNING
            elif self.state == GameState.RUNNING:
                self.state = GameState.PAUSED
            elif self.state == GameState.PAUSED:
                self.state = GameState.RUNNING

        if self.state == GameState.GAME_OVER:
            if key == pygame.K_o:
                success = self.logger.confirm_current_episode()
                if success:
                    self._game_over_save_message = "Episode successfully saved!"
                    self._game_over_saved_count = self._compute_saved_episode_count()
            elif key == pygame.K_r:
                self.logger.discard_current_episode()
                self._episode_id += 1
                self.core.reset(seed=self.config.seed)
                self.plate_background_colors = [0 for _ in self.plate_background_colors]
                self.plate_background_texts = [None for _ in self.plate_background_texts]
                self.apply_visual_prompts()
                self.state = GameState.NOT_STARTED
            elif key == pygame.K_p:
                self.logger.discard_current_episode()
                pygame.event.post(pygame.event.Event(pygame.QUIT))

        if self.config.mode in (Mode.HUMAN_BASELINE, Mode.PLAYGROUND_NO_AMS):
            if key >= pygame.K_1 and key < pygame.K_1 + self.core.num_plates:
                internal_idx = key - pygame.K_1
                self._handle_manual_switch(internal_idx)
        if (
            self.config.mode == Mode.PLAYGROUND_NO_AMS
            and self.config.debug_prompt
            and self.state == GameState.RUNNING
        ):
            if key == pygame.K_q:
                self.prompt_api.play_sound("single_switch")
            elif key == pygame.K_w:
                self.prompt_api.play_sound("mode_switch")
            elif key == pygame.K_e:
                self.prompt_api.play_sound("left_switch", left_vol=1.0, right_vol=0.2)
            elif key == pygame.K_r:
                self.prompt_api.play_sound("right_switch", left_vol=0.2, right_vol=1.0)

    def _handle_manual_switch(self, target_internal_idx: int) -> None:
        source_internal_idx = self.core.controlled_plate
        if target_internal_idx == source_internal_idx:
            return
        event_ms = int(round(self.t_now * 1000))
        if self.config.mode == Mode.HUMAN_BASELINE:
            self.metrics.on_baseline_switch(
                core=self.core, event_ms=event_ms, source_idx=source_internal_idx, target_idx=target_internal_idx
            )
        self.core.commit_switch(target_internal_idx, self.t_now)
        self.prompt_api.set_controlled_plate(target_internal_idx)

    def _find_directional_target(self, direction: str) -> Optional[int]:
        if not self.plate_grid:
            return None
        current = self.core.controlled_plate
        pos_map = {}
        for r, row in enumerate(self.plate_grid):
            for c, idx in enumerate(row):
                pos_map[idx] = (r, c)
        if current not in pos_map:
            return None
        r, c = pos_map[current]
        rows = len(self.plate_grid)
        cols = max(len(row) for row in self.plate_grid)

        def wrapped(next_r: int, next_c: int) -> Optional[int]:
            if 0 <= next_r < rows and 0 <= next_c < len(self.plate_grid[next_r]):
                return self.plate_grid[next_r][next_c]
            return None

        if direction == "left":
            next_c = (c - 1) % len(self.plate_grid[r])
            return wrapped(r, next_c)
        if direction == "right":
            next_c = (c + 1) % len(self.plate_grid[r])
            return wrapped(r, next_c)
        if direction == "up":
            next_r = (r - 1) % rows
            next_c = min(c, len(self.plate_grid[next_r]) - 1)
            return wrapped(next_r, next_c)
        if direction == "down":
            next_r = (r + 1) % rows
            next_c = min(c, len(self.plate_grid[next_r]) - 1)
            return wrapped(next_r, next_c)
        return None

    def _plate_at_position(self, x: int, y: int) -> Optional[int]:
        for idx, (cx, cy) in enumerate(self.plate_positions):
            if math.hypot(x - cx, y - cy) <= self.displayed_radius:
                return idx
        return None

    def _handle_mouse_debug(self, event) -> None:
        plate_idx = self._plate_at_position(*event.pos)
        if plate_idx is None:
            return
        if event.button == 1:
            current = self.plate_background_texts[plate_idx]
            sequence = [None] + [str(i + 1) for i in range(self.core.num_plates)] + ["R", "L"]
            try:
                idx = sequence.index(current)
            except ValueError:
                idx = 0
            next_val = sequence[(idx + 1) % len(sequence)]
            self.plate_background_texts[plate_idx] = next_val
        elif event.button == 3:
            current_idx = self.plate_background_colors[plate_idx]
            next_idx = (current_idx + 1) % len(BACKGROUND_COLOR_NAMES)
            self.plate_background_colors[plate_idx] = next_idx

    def run(self) -> None:
        running = True
        while running:
            dt_real = self.clock.tick(FPS) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    self._handle_keydown(event.key)
                elif (
                    event.type == pygame.MOUSEBUTTONDOWN
                    and self.state == GameState.RUNNING
                    and self.config.mode == Mode.PLAYGROUND_NO_AMS
                    and self.config.debug_prompt
                ):
                    self._handle_mouse_debug(event)

            self._check_gamepad_connection()
            self._update_button_states()

            if self._get_button_pressed(6) or self._get_button_pressed("L2_trigger"):
                if self.state == GameState.NOT_STARTED:
                    self._start_new_episode_if_needed()
                    self.state = GameState.RUNNING
                elif self.state == GameState.GAME_OVER:
                    self.logger.discard_current_episode()
                    pygame.event.post(pygame.event.Event(pygame.QUIT))
            if self._get_button_pressed("R2_trigger") and self.state == GameState.GAME_OVER:
                self.logger.discard_current_episode()
                self._episode_id += 1
                self.core.reset(seed=self.config.seed)
                self.plate_background_colors = [0 for _ in self.plate_background_colors]
                self.plate_background_texts = [None for _ in self.plate_background_texts]
                self.apply_visual_prompts()
                self.state = GameState.NOT_STARTED
            if self._get_button_pressed(7) and self.state == GameState.GAME_OVER:
                self.logger.discard_current_episode()
                self._episode_id += 1
                self.core.reset(seed=self.config.seed)
                self.apply_visual_prompts()
                self.state = GameState.NOT_STARTED

            if self.state == GameState.RUNNING and self.config.mode in (Mode.HUMAN_BASELINE, Mode.PLAYGROUND_NO_AMS):
                if self._get_button_pressed("dpad_left"):
                    target = self._find_directional_target("left")
                    if target is not None:
                        self._handle_manual_switch(target)
                if self._get_button_pressed("dpad_right"):
                    target = self._find_directional_target("right")
                    if target is not None:
                        self._handle_manual_switch(target)
                if self._get_button_pressed("dpad_up"):
                    target = self._find_directional_target("up")
                    if target is not None:
                        self._handle_manual_switch(target)
                if self._get_button_pressed("dpad_down"):
                    target = self._find_directional_target("down")
                    if target is not None:
                        self._handle_manual_switch(target)
                if self._get_button_pressed(2):
                    target = self._find_directional_target("left")
                    if target is not None:
                        self._handle_manual_switch(target)
                if self._get_button_pressed(1):
                    target = self._find_directional_target("right")
                    if target is not None:
                        self._handle_manual_switch(target)
                if self._get_button_pressed(3):
                    target = self._find_directional_target("up")
                    if target is not None:
                        self._handle_manual_switch(target)
                if self._get_button_pressed(0):
                    target = self._find_directional_target("down")
                    if target is not None:
                        self._handle_manual_switch(target)

            if self.state == GameState.RUNNING:
                self._update_control_and_physics(dt_real)
            self._draw()

        try:
            if self._decision_trace_file is not None:
                self._decision_trace_file.close()
        except Exception:
            pass

        pygame.quit()

    def _update_control_and_physics(self, dt_real: float) -> None:
        jx, jy = self._read_joystick_axes()
        self._physics_accumulator += dt_real

        while self._physics_accumulator >= DT_MICRO and self.state == GameState.RUNNING:
            if self.config.mode in (Mode.AMS_PLAY, Mode.PLAYGROUND_AMS) and self.t_now >= self.next_ams_decision_time:
                prompt_busy = (self.prompt_controller.state != PromptState.IDLE)
                time_to_commit = 0.0
                if self.prompt_controller.state == PromptState.PRE_CUE and self.prompt_controller.commit_time is not None:
                    time_to_commit = max(0.0, self.prompt_controller.commit_time - self.t_now)

                obs, action_mask = self.core.build_ams_obs_and_mask(
                    self.config, prompt_busy=prompt_busy, time_to_commit=time_to_commit
                )

                policy_action = None
                chosen_action = None
                override_applied = False

                if self.ams is not None:
                    policy_action = self.ams.select_action(obs, action_mask)
                    policy_action = self._validate_ams_action(policy_action, action_mask)
                    chosen_action = policy_action

                    valid_count = int(sum(1 for x in action_mask if int(x) == 1))
                    switch_allowed = (valid_count > 1) and (not prompt_busy)

                    if switch_allowed:
                        argmin_plate = None
                        min_ttb = float("inf")
                        for logical_i in range(self.core.num_plates):
                            internal_i = self.core.get_internal_plate_index(logical_i)
                            st = self.core.get_status_for_plate(internal_i)
                            ttb = float(st.get("ttb_sec", 60.0))
                            if ttb < min_ttb:
                                min_ttb = ttb
                                argmin_plate = logical_i

                        if (
                            argmin_plate is not None
                            and min_ttb < EMERGENCY_TTB_SEC
                            and 0 <= int(argmin_plate) < len(action_mask)
                            and int(action_mask[int(argmin_plate)]) == 1
                            and chosen_action != int(argmin_plate)
                        ):
                            chosen_action = int(argmin_plate)
                            override_applied = True

                    if self._decision_trace_writer is not None:
                        self._log_decision_trace(
                            t_now=self.t_now,
                            obs=obs,
                            action_mask=action_mask,
                            chosen_action=chosen_action,
                            prompt_busy=prompt_busy,
                            time_to_commit=time_to_commit,
                            policy_action=policy_action,
                            override_applied=override_applied,
                        )

                    if chosen_action is not None:
                        internal_target = self.core.get_internal_plate_index(int(chosen_action))
                        self.prompt_controller.request_switch(internal_target, self.t_now)

                self._decision_idx += 1
                self.next_ams_decision_time += self.ams_decision_interval

            cp_internal = self.core.controlled_plate
            self.core.plates[cp_internal].update_with_joystick(jx, jy)

            self.prompt_controller.update(self.t_now)
            alive = self.core.step_physics(DT_MICRO)

            if self.cognitive_agent is not None:
                self.cognitive_agent.update_belief_only(self.t_now, self.core)

            self.metrics.on_micro_step(self.core, DT_MICRO, joystick_x=jx, joystick_y=jy)

            self._physics_accumulator -= DT_MICRO
            self.t_now += DT_MICRO

            if not alive:
                self.prompt_controller.abort_current_precue()
                break

        if self.core.state == GameState.GAME_OVER:
            self.state = GameState.GAME_OVER
            ep = self.metrics.end_episode(self.core, failure_plate_idx=self.core.failure_plate_idx)
            if ep is not None:
                self.logger.on_episode_finished(ep)
                self._reset_game_over_save_ui()

    def _draw(self) -> None:
        self.screen.fill(BLACK)
        for internal_idx in range(self.core.num_plates):
            plate = self.core.plates[internal_idx]
            ball = self.core.balls[internal_idx]
            center_x, center_y = self.plate_positions[internal_idx]

            plate_color = YELLOW if internal_idx == self.core.controlled_plate else GREEN
            color_name = BACKGROUND_COLOR_NAMES[self.plate_background_colors[internal_idx]]
            bg_color = BACKGROUND_COLORS[color_name]
            if color_name != "default_background":
                pygame.draw.circle(self.screen, bg_color, (center_x, center_y), self.displayed_radius)

            text = self.plate_background_texts[internal_idx]
            if text:
                show_coordinates = text.isdigit() and int(text) <= self.core.num_plates
                if show_coordinates:
                    text_surface = self.background_font.render(text, True, BLACK)
                    text_spacing = int(15 * BACKGROUND_TEXT_SCALE)
                    text_rect = text_surface.get_rect(center=(center_x, center_y - text_spacing))
                    self.screen.blit(text_surface, text_rect)

                    plate_num = int(text)
                    row, col = self.get_plate_grid_position(plate_num - 1)
                    coord_text = f"({row + 1}, {col + 1})"
                    coord_surface = self.coordinate_font.render(coord_text, True, BLACK)
                    coord_rect = coord_surface.get_rect(center=(center_x, center_y + text_spacing))
                    self.screen.blit(coord_surface, coord_rect)
                else:
                    text_surface = self.background_font.render(text, True, BLACK)
                    text_rect = text_surface.get_rect(center=(center_x, center_y))
                    self.screen.blit(text_surface, text_rect)

            pygame.draw.circle(self.screen, plate_color, (center_x, center_y), self.displayed_radius, 2)

            alpha_value = int(PLATE_INTERNAL_TRANSPARENCY * 2.55)
            plate_surface = pygame.Surface(
                (self.displayed_radius * 2 + 20, self.displayed_radius * 2 + 20), pygame.SRCALPHA
            )
            surface_center_x = self.displayed_radius + 10
            surface_center_y = self.displayed_radius + 10

            reference_angles = [15, 30, 45]
            for angle in reference_angles:
                radius = PLATE_DISPLAY_RADIUS * (angle / MAX_TILT)
                draw_radius = int(radius * self.display_scale)
                color_with_alpha = plate_color + (alpha_value,)
                pygame.draw.circle(
                    plate_surface, color_with_alpha, (surface_center_x, surface_center_y), draw_radius, 1
                )

            for angle in range(0, 360, 90):
                line_length = self.plate_radius * PLATE_DISPLAY_SCALE_FACTOR
                end_x = surface_center_x + int(line_length * math.cos(math.radians(angle)))
                end_y = surface_center_y + int(line_length * math.sin(math.radians(angle)))
                color_with_alpha = plate_color + (alpha_value,)
                pygame.draw.line(
                    plate_surface, color_with_alpha, (surface_center_x, surface_center_y), (end_x, end_y), 1
                )

            self.screen.blit(plate_surface, (center_x - surface_center_x, center_y - surface_center_y))

            if plate.tilt_magnitude > 0:
                arrow_length = self.displayed_radius * (plate.tilt_magnitude / MAX_TILT)
                angle_rad = math.radians(plate.tilt_direction)
                end_x = center_x + arrow_length * math.cos(angle_rad)
                end_y = center_y + arrow_length * math.sin(angle_rad)
                pygame.draw.line(self.screen, YELLOW, (center_x, center_y), (end_x, end_y), 3)
                head_length = 15
                head_angle = math.pi / 6
                for offset in (-head_angle, head_angle):
                    head_x = end_x - head_length * math.cos(angle_rad + offset)
                    head_y = end_y - head_length * math.sin(angle_rad + offset)
                    pygame.draw.line(self.screen, YELLOW, (end_x, end_y), (head_x, head_y), 3)

            # Rendering uses an enlarged ball radius for visibility; physics uses BALL_RADIUS.
            ball_screen_x = center_x + int(ball.x * self.physics_to_display_scale)
            ball_screen_y = center_y + int(ball.y * self.physics_to_display_scale)
            ball_display_radius = int(BALL_RADIUS * self.display_scale * (PLATE_DISPLAY_RADIUS / PLATE_RADIUS) * 2)
            pygame.draw.circle(self.screen, RED, (ball_screen_x, ball_screen_y), ball_display_radius * 2)

            if self.config.display_belief and self.cognitive_agent is not None:
                logical_idx = self.core.internal_to_logical[internal_idx]
                bx, by = self.cognitive_agent.get_belief_position_px(logical_idx)
                belief_screen_x = center_x + int(bx * self.physics_to_display_scale)
                belief_screen_y = center_y + int(by * self.physics_to_display_scale)
                pygame.draw.circle(
                    self.screen, WHITE, (belief_screen_x, belief_screen_y), ball_display_radius * 2, 2
                )

            plate_number = str(internal_idx + 1)
            number_surface = self.plate_font.render(plate_number, True, WHITE)
            number_pos = (
                center_x - self.plate_radius + int(self.plate_radius * PLATE_NUMBER_X_OFFSET),
                center_y - self.plate_radius + int(self.plate_radius * PLATE_NUMBER_Y_OFFSET),
            )
            self.screen.blit(number_surface, number_pos)

            if internal_idx == self.core.controlled_plate:
                label_surface = self.plate_font.render("C", True, WHITE)
                label_pos = (
                    number_pos[0],
                    number_pos[1] + number_surface.get_height() * PLATE_LABEL_Y_SPACING,
                )
                self.screen.blit(label_surface, label_pos)

        if self.state in (GameState.RUNNING, GameState.PAUSED, GameState.GAME_OVER):
            time_text = f"Time: {self.core.game_time:.1f}s"
            time_surface = self.font.render(time_text, True, WHITE)
            self.screen.blit(time_surface, (20, 20))


        if self.state == GameState.NOT_STARTED:
            text = self.large_font.render("Press BACKSPACE or L2 to Start", True, WHITE)
            rect = text.get_rect(center=(self.WINDOW_WIDTH // 2, self.WINDOW_HEIGHT // 2))
            self.screen.blit(text, rect)
        elif self.state == GameState.PAUSED:
            text = self.large_font.render("PAUSED", True, WHITE)
            rect = text.get_rect(center=(self.WINDOW_WIDTH // 2, self.WINDOW_HEIGHT // 2))
            self.screen.blit(text, rect)
        elif self.state == GameState.GAME_OVER:
            text1 = self.large_font.render("Game Over. Press R or R2 to Restart", True, RED)
            rect1 = text1.get_rect(center=(self.WINDOW_WIDTH // 2, self.WINDOW_HEIGHT // 2 - 40))
            self.screen.blit(text1, rect1)
            text2 = self.large_font.render("Press P or L2 to Exit", True, RED)
            rect2 = text2.get_rect(center=(self.WINDOW_WIDTH // 2, self.WINDOW_HEIGHT // 2 + 40))
            self.screen.blit(text2, rect2)
            if self.config.mode in (Mode.HUMAN_BASELINE, Mode.AMS_PLAY):
                text3 = self.large_font.render(self._game_over_save_message, True, WHITE)
                rect3 = text3.get_rect(center=(self.WINDOW_WIDTH // 2, self.WINDOW_HEIGHT // 2 + 120))
                self.screen.blit(text3, rect3)
                count_text = f"Save episodes: {self._game_over_saved_count}"
                text4 = self.large_font.render(count_text, True, WHITE)
                rect4 = text4.get_rect(center=(self.WINDOW_WIDTH // 2, self.WINDOW_HEIGHT // 2 + 170))
                self.screen.blit(text4, rect4)
        pygame.display.flip()

def load_trained_ams(config: GameConfig):
    """
    Load a trained AMS checkpoint from AMSTrained/ and return a runtime AMSInterface.

    config.ams_checkpoint may be "auto", "AMS3", "AMS6", or "AMS9". "auto" selects the
    highest available stage (AMS9 → AMS6 → AMS3).
    """
    def _resolve_stage(ams_checkpoint: str) -> str:
        if ams_checkpoint == "auto":
            for stage in ("AMS9", "AMS6", "AMS3"):
                try:
                    PPOAMS.from_stage(stage)
                    return stage
                except FileNotFoundError:
                    continue
            raise RuntimeError(
                "AMS checkpoint selection 'auto' was requested, "
                "but no AMS3/AMS6/AMS9 checkpoint could be found in AMSTrained/."
            )
        else:
            return ams_checkpoint

    stage = _resolve_stage(config.ams_checkpoint)

    ams = PPOAMS.from_stage(
        stage=stage,
        deterministic=True,
        tie_break_eps=0.03,
        tie_break_sample=True,
    )
    return ams




def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="PlaygroundNoAMS",
        choices=[m.value for m in Mode],
        help="Execution mode for the environment.",
    )
    parser.add_argument("--num-plates", type=int, default=4, help="Number of active plates (2–12).")
    parser.add_argument(
        "--participant-id",
        type=int,
        default=None,
        help="Participant identifier (used in HumanBaseline / AMSPlay; ignored otherwise).",
    )
    parser.add_argument("--display-belief", action="store_true", help="Visualize the cognitive belief ball (AMS modes).")
    parser.add_argument(
        "--debug-prompt",
        action="store_true",
        help="Enable manual prompt debug keys (only meaningful in PlaygroundNoAMS).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for layout and physics randomization.")
    parser.add_argument(
        "--ams-checkpoint",
        type=str,
        default="auto",
        choices=["auto", "AMS3", "AMS6", "AMS9"],
        help="Which trained AMS checkpoint to load in AMS modes.",
    )
    parser.add_argument(
        "--block-order",
        type=str,
        default=None,
        help=(
            "Block order label to log in episodes, e.g. 'BaselineFirst' or "
            "'AMSFirst'. If omitted, the block_order field is left blank."
        ),
    )
    parser.add_argument(
        "--trial-index",
        type=int,
        default=None,
        help=(
            "Within-block trial index to log in episodes (e.g., 1..8 for N=2..9). "
            "If omitted, the trial_index field is left blank."
        ),
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help=(
            "Optional directory for writing PlayLogs CSVs (episodes.csv, switches.csv). "
            "If omitted, PlateGame uses its default log paths."
        ),
    )
    parser.add_argument(
        "--decision-trace",
        action="store_true",
        help=(
            "Opt-in: write per-AMS-decision debug trace to decision_trace.csv in --log-dir. "
            "Off by default. Allowed only in AMSPlay / HumanBaseline. Never allowed in Playground."
        ),
    )

    return parser.parse_args(argv)



def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    mode_val = Mode(args.mode)
    if getattr(args, "decision_trace", False) and mode_val in (Mode.PLAYGROUND_AMS, Mode.PLAYGROUND_NO_AMS):
        raise ValueError("--decision-trace is not allowed in Playground modes (no logging permitted).")
    config = GameConfig(
        mode=Mode(args.mode),
        num_plates=args.num_plates,
        participant_id=args.participant_id,
        display_belief=args.display_belief,
        debug_prompt=args.debug_prompt,
        headless=(args.mode == "Training"),
        seed=args.seed,
        ams_checkpoint=args.ams_checkpoint,
        block_order=args.block_order,
        trial_index=args.trial_index,
        log_dir=args.log_dir,
        decision_trace=bool(getattr(args, "decision_trace", False)),
    )


    if config.mode == Mode.TRAINING:
        raise NotImplementedError("Training loop integration should be added externally.")
    else:
        ams_model = None
        if config.mode in (Mode.AMS_PLAY, Mode.PLAYGROUND_AMS) and ams_model is None:
            ams_model = load_trained_ams(config)
        game = GameInteractive(config, ams=ams_model)
        game.run()


if __name__ == "__main__":
    main()
