from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Physics constants (keep aligned with PlateGame)
PLATE_RADIUS = 1500
BALL_RADIUS = 30
MAX_TILT = 45.0

R_USABLE = PLATE_RADIUS - BALL_RADIUS


@dataclass
class CognitiveParams:
    # Observation probability used in the multiplicative reweighting step.
    o_p_active: float = 0.2185

    # Belief update period (seconds).
    u: float = 0.4852

    # Standard deviation used for mean drift when not visible.
    sigma_mean0: float = 0.05

    # Standard deviation used for sampling displacement hypotheses.
    sigma_P: float = 0.0140

    # Distribution persistence time (seconds).
    d_t: float = 0.25

    # Constant reaction time (seconds).
    constant_reaction_time: float = 0.0

    # Discretization: requested bins, coerced to the next perfect square.
    number_of_bins: int = 1000

    # Number of displacement samples per belief update.
    num_samples: int = 64


class CognitiveAgent:
    """
    Cognitive gameplay agent with a discretized belief state.

    Belief mechanism:
      - Square support [-R, R) x [-R, R) with margin beyond the physical plate.
      - Out-of-support maps to bin -1 and contributions are dropped.
      - Interior bins use visitor-bin propagation.
      - Edge bins use crossed-bin enumeration; empty crossed sets contribute 0.
      - Observation update uses multiplicative o_p reweighting; when the true bin has zero propagated mass, a small floor mass is assigned to enable recovery.
      - Believed position is MAP bin center.

    Cognitive constraints:
      - Visibility gating uses constant_reaction_time and d_t.
      - Action slip uses source-belief substitution for d_t.
    """

    def __init__(self, params: CognitiveParams, dt_micro: float, fps: int = 60):
        self.params = params
        self.dt_micro = float(dt_micro)
        self.fps = int(fps)

        self._rng = random.Random()
        self._np_rng = np.random.default_rng()

        self._num_plates: int = 0

        # Unit conversion: belief uses a "radius=5" physical scale, with belief margin +1.
        self._R_phys_units: float = 5.0
        self._R_belief_units: float = 6.0
        self._scale_px_per_unit: float = float(R_USABLE) / self._R_phys_units

        # Discretization derived from params.number_of_bins
        self._N: int = 0
        self._N_bins: int = 0
        self._bin_size: float = 0.0

        # Precomputed per-bin centers in belief units
        self._centers_x: np.ndarray = np.zeros(1, dtype=np.float64)
        self._centers_y: np.ndarray = np.zeros(1, dtype=np.float64)
        self._is_edge: np.ndarray = np.zeros(1, dtype=bool)
        self._interior_idx: np.ndarray = np.zeros(1, dtype=np.int32)
        self._edge_idx: np.ndarray = np.zeros(1, dtype=np.int32)

        # Fractions for crossed-bin sampling along segment (length S+1, S=2N)
        self._cross_s: np.ndarray = np.zeros(1, dtype=np.float64)

        # Per-plate belief state
        self._belief: List[np.ndarray] = []
        self._belief_tmp: List[np.ndarray] = []
        self._estimated_d: List[Tuple[float, float]] = []
        self._sigma_mean_current: List[float] = []
        self._avg_d: List[Tuple[float, float]] = []
        self._map_pos_units: List[Tuple[float, float]] = []

        # True position cache for displacement estimation across consecutive visible updates.
        self._last_visible_pos_units: List[Tuple[float, float]] = []
        self._was_visible_last_update: List[bool] = []

        # Timers (seconds)
        self._update_timer: List[float] = []
        self._visibility_timer: List[float] = []
        self._persistence_timer: List[float] = []

        # Slip source (logical index)
        self._slip_source: List[Optional[int]] = []

        # Tick bookkeeping
        self._last_tick_time: Optional[float] = None

        # Diagnostic counters
        self.stats: Dict[str, int] = {
            "belief_updates_total": 0,
            "belief_visible_updates_total": 0,
            "belief_sum_invalid": 0,
            "true_bin_out_of_support_visible": 0,
            "true_bin_zero_mass_visible": 0,
            "crossed_bins_empty": 0,
        }

    # ---------------------------------------------------------------- Reset / state
    def reset(self, num_plates: int, seed: Optional[int] = None, core=None) -> None:
        if seed is not None:
            self._rng.seed(seed)
            self._np_rng = np.random.default_rng(seed)
        else:
            self._np_rng = np.random.default_rng()

        self._num_plates = int(num_plates)

        for k in list(self.stats.keys()):
            self.stats[k] = 0

        # Coerce bin count to next perfect square
        requested = max(4, int(self.params.number_of_bins))
        n_side = int(math.ceil(math.sqrt(requested)))
        self._N = n_side
        self._N_bins = self._N * self._N
        self._bin_size = (2.0 * self._R_belief_units) / float(self._N)

        # Precompute centers and edge flags
        bins = np.arange(self._N_bins, dtype=np.int32)
        rows = bins // self._N
        cols = bins % self._N
        self._centers_x = rows.astype(np.float32) * self._bin_size - self._R_belief_units + (self._bin_size / 2.0)
        self._centers_y = cols.astype(np.float32) * self._bin_size - self._R_belief_units + (self._bin_size / 2.0)

        self._is_edge = (rows == 0) | (rows == (self._N - 1)) | (cols == 0) | (cols == (self._N - 1))
        self._edge_idx = np.nonzero(self._is_edge)[0].astype(np.int32)
        self._interior_idx = np.nonzero(~self._is_edge)[0].astype(np.int32)

        # Crossed-bin sampling resolution (S=2N)
        S = 2 * self._N
        self._cross_s = np.linspace(0.0, 1.0, S + 1, dtype=np.float32)

        # Initialize per-plate structures
        self._belief = []
        self._estimated_d = []
        self._belief_tmp = []
        self._sigma_mean_current = []
        self._avg_d = []
        self._map_pos_units = []

        self._update_timer = [0.0] * self._num_plates
        self._visibility_timer = [0.0] * self._num_plates
        self._persistence_timer = [0.0] * self._num_plates
        self._slip_source = [None] * self._num_plates

        self._last_visible_pos_units = [(0.0, 0.0)] * self._num_plates
        self._was_visible_last_update = [False] * self._num_plates

        self._last_tick_time = None

        # Initialize belief to delta at true start position when core is available
        for logical_idx in range(self._num_plates):
            b = np.zeros(self._N_bins, dtype=np.float32)
            b_tmp = np.zeros(self._N_bins, dtype=np.float32)

            if core is not None:
                internal_idx = core.get_internal_plate_index(logical_idx)
                ball = core.balls[internal_idx]
                x_u = float(ball.x) / self._scale_px_per_unit
                y_u = float(ball.y) / self._scale_px_per_unit
                self._last_visible_pos_units[logical_idx] = (x_u, y_u)
                start_bin = self._coords_to_bin(x_u, y_u)
            else:
                start_bin = self._coords_to_bin(0.0, 0.0)

            if start_bin == -1:
                start_bin = self._coords_to_bin(0.0, 0.0)

            b[start_bin] = 1.0
            self._belief.append(b)
            self._belief_tmp.append(b_tmp)

            self._estimated_d.append((0.0, 0.0))
            self._sigma_mean_current.append(float(self.params.sigma_mean0))
            self._avg_d.append((0.0, 0.0))

            mx, my = self._bin_center_units(start_bin)
            self._map_pos_units.append((mx, my))

    def on_switch_commit(self, source_idx: int, target_idx: int, t_now: float) -> None:
        if 0 <= target_idx < self._num_plates:
            self._slip_source[target_idx] = int(source_idx)
            self._update_timer[target_idx] = 0.0

    # ---------------------------------------------------------------- Control loop
    def act_for_training_control(self, plate_logical_idx: int, t_now: float, core) -> Tuple[float, float]:
        self._tick(t_now=t_now, active_logical_idx=plate_logical_idx, core=core)

        pos_u = self._get_effective_belief_pos_units(active_logical_idx=plate_logical_idx)
        r_hat_px = (pos_u[0] * self._scale_px_per_unit, pos_u[1] * self._scale_px_per_unit)

        v_hat_px_per_micro = self._get_velocity_estimate_px_per_micro(plate_logical_idx)

        return self._pd_control(plate_logical_idx, core, r_hat_px, v_hat_px_per_micro)

    def update_belief_only(self, t_now: float, core) -> None:
        if self._num_plates <= 0:
            return
        cp_logical = int(core.internal_to_logical[core.controlled_plate])
        self._tick(t_now=t_now, active_logical_idx=cp_logical, core=core)

    def get_belief_position_px(self, plate_logical_idx: int) -> Tuple[float, float]:
        if plate_logical_idx < 0 or plate_logical_idx >= self._num_plates:
            return (0.0, 0.0)
        x_u, y_u = self._map_pos_units[plate_logical_idx]
        return (x_u * self._scale_px_per_unit, y_u * self._scale_px_per_unit)

    def get_effective_belief_position_px(self, active_logical_idx: int) -> Tuple[float, float]:
        """Return slip-substituted believed position (argmax), in plate-local pixels."""
        x_u, y_u = self._get_effective_belief_pos_units(active_logical_idx=active_logical_idx)
        return (x_u * self._scale_px_per_unit, y_u * self._scale_px_per_unit)

    # ---------------------------------------------------------------- Tick / timers
    def _tick(self, t_now: float, active_logical_idx: int, core) -> None:
        if self._last_tick_time is None:
            dt = 0.0
        else:
            dt = max(0.0, float(t_now) - float(self._last_tick_time))

        self._last_tick_time = float(t_now)

        u = float(self.params.u)
        for logical_idx in range(self._num_plates):
            # Update cadence timer runs for all plates.
            self._update_timer[logical_idx] += dt

            # Visibility/persistence timers reset while inactive.
            if logical_idx == active_logical_idx:
                self._visibility_timer[logical_idx] += dt
                self._persistence_timer[logical_idx] += dt
            else:
                self._visibility_timer[logical_idx] = 0.0
                self._persistence_timer[logical_idx] = 0.0

            if self._update_timer[logical_idx] > u:
                self._update_timer[logical_idx] = 0.0
                visible = self._is_visible(logical_idx, active_logical_idx)
                self._belief_update_for_plate(logical_idx, visible, core)

    def _is_visible(self, logical_idx: int, active_logical_idx: int) -> bool:
        if logical_idx != active_logical_idx:
            return False
        vt = float(self._visibility_timer[logical_idx])
        return (vt > float(self.params.constant_reaction_time)) and (vt > float(self.params.d_t))

    # ---------------------------------------------------------------- Belief update

    
    def _belief_update_for_plate(self, logical_idx: int, visible: bool, core) -> None:
        self.stats["belief_updates_total"] += 1
        if visible:
            self.stats["belief_visible_updates_total"] += 1
        current = self._belief[logical_idx]
        new = self._belief_tmp[logical_idx]
        new.fill(0.0)

        u = float(self.params.u)
        K = max(1, int(self.params.num_samples))

        internal_idx = core.get_internal_plate_index(logical_idx)
        ball = core.balls[internal_idx]

        # True state in belief units
        x_u = float(ball.x) / self._scale_px_per_unit
        y_u = float(ball.y) / self._scale_px_per_unit
        vx_u_per_sec = (float(ball.vx) * float(self.fps)) / self._scale_px_per_unit
        vy_u_per_sec = (float(ball.vy) * float(self.fps)) / self._scale_px_per_unit

        d_true_x = vx_u_per_sec * u
        d_true_y = vy_u_per_sec * u

        # When visibility is sustained across updates, use observed displacement between updates.
        if visible and self._was_visible_last_update[logical_idx]:
            px_prev, py_prev = self._last_visible_pos_units[logical_idx]
            d_true_x = x_u - px_prev
            d_true_y = y_u - py_prev

        # Drift of mean displacement
        est_dx, est_dy = self._estimated_d[logical_idx]
        sigma_mean = float(self._sigma_mean_current[logical_idx])

        if visible:
            est_dx, est_dy = d_true_x, d_true_y
            sigma_mean = float(self.params.sigma_mean0)
        else:
            mu_x = self._rng.gauss(est_dx, sigma_mean) if sigma_mean > 0.0 else est_dx
            mu_y = self._rng.gauss(est_dy, sigma_mean) if sigma_mean > 0.0 else est_dy
            sigma_mean = sigma_mean * 0.5

            est_norm = math.hypot(est_dx, est_dy)
            true_norm = math.hypot(d_true_x, d_true_y)
            if est_norm <= 1e-12:
                ratio = 0.0
            else:
                ratio = true_norm / est_norm

            est_dx, est_dy = mu_x * ratio, mu_y * ratio

        self._estimated_d[logical_idx] = (est_dx, est_dy)
        self._sigma_mean_current[logical_idx] = sigma_mean

        # Sample displacements
        sigma_P = float(self.params.sigma_P)
        sigma_P = max(sigma_P, 0.10 * float(self._bin_size))
        dx_samples = self._np_rng.normal(loc=est_dx, scale=sigma_P, size=K).astype(np.float32)
        dy_samples = self._np_rng.normal(loc=est_dy, scale=sigma_P, size=K).astype(np.float32)

        avg_dx = float(dx_samples.mean()) if K > 0 else 0.0
        avg_dy = float(dy_samples.mean()) if K > 0 else 0.0
        self._avg_d[logical_idx] = (avg_dx, avg_dy)

        # Propagation
        R = float(self._R_belief_units)
        bin_size = float(self._bin_size)
        N = int(self._N)

        interior = self._interior_idx
        cx = self._centers_x[interior]
        cy = self._centers_y[interior]

        invK = np.float32(1.0 / float(K))

        vx = cx[None, :] - dx_samples[:, None]
        vy = cy[None, :] - dy_samples[:, None]

        in_support = (vx >= -R) & (vx < R) & (vy >= -R) & (vy < R)

        xp = vx + R
        yp = vy + R

        ix = np.floor(xp / bin_size).astype(np.int32)
        iy = np.floor(yp / bin_size).astype(np.int32)

        in_grid = in_support & (ix >= 0) & (ix < N) & (iy >= 0) & (iy < N)

        visitor_bins = (ix * N + iy).astype(np.int32)
        visitor_bins = np.where(in_grid, visitor_bins, 0)

        vals = current[visitor_bins]
        vals = np.where(in_grid, vals, 0.0)

        new[interior] += vals.sum(axis=0, dtype=np.float32) * invK

        edge_bins = self._edge_idx  # int32 from reset()
        E = int(edge_bins.size)
        if E > 0:
            cx_e = self._centers_x[edge_bins]
            cy_e = self._centers_y[edge_bins]

            vx_e = cx_e[None, :] - dx_samples[:, None]
            vy_e = cy_e[None, :] - dy_samples[:, None]

            in_support_e = (vx_e >= -R) & (vx_e < R) & (vy_e >= -R) & (vy_e < R)

            xp_e = vx_e + R
            yp_e = vy_e + R

            ix_e = np.floor(xp_e / bin_size).astype(np.int32)
            iy_e = np.floor(yp_e / bin_size).astype(np.int32)

            in_grid_e = in_support_e & (ix_e >= 0) & (ix_e < N) & (iy_e >= 0) & (iy_e < N)

            visitor_e = (ix_e * N + iy_e).astype(np.int32)
            visitor_e = np.where(in_grid_e, visitor_e, 0)

            vals_e = current[visitor_e]
            vals_e = np.where(in_grid_e, vals_e, vals_e * 0.0)

            new[edge_bins] += vals_e.sum(axis=0, dtype=np.float32) * invK

        # Observation reweighting when visible
        if visible:
            true_bin = self._coords_to_bin(x_u, y_u)
            if true_bin == -1:
                self.stats["true_bin_out_of_support_visible"] += 1

            true_val = 0.0
            if true_bin != -1:
                true_val = float(new[true_bin])
                if true_val == 0.0:
                    self.stats["true_bin_zero_mass_visible"] += 1

                    # If the propagated belief assigns zero probability to the true bin,
                    # inject a small floor mass so multiplicative reweighting can recover.
                    max_mass = float(new.max())
                    floor_mass = max(max_mass, 1.0 / float(K))
                    if floor_mass > 0.0:
                        new[true_bin] = floor_mass
                        true_val = floor_mass

            mask_nonzero = new != 0.0
            not_true_factor = (1.0 - float(self.params.o_p_active)) / float(self._N_bins - 1)

            new[mask_nonzero] *= not_true_factor

            if true_bin != -1 and true_val != 0.0:
                new[true_bin] = true_val * float(self.params.o_p_active)

        # Normalize
        s = float(new.sum())
        if (not math.isfinite(s)) or (s <= 1e-12):
            self.stats["belief_sum_invalid"] += 1
            self._was_visible_last_update[logical_idx] = bool(visible)
            return

        new /= s
        self._belief[logical_idx], self._belief_tmp[logical_idx] = new, current

        # MAP position
        map_idx = int(np.argmax(new))
        self._map_pos_units[logical_idx] = (float(self._centers_x[map_idx]), float(self._centers_y[map_idx]))
        if visible:
            self._last_visible_pos_units[logical_idx] = (x_u, y_u)
        self._was_visible_last_update[logical_idx] = bool(visible)


    # ---------------------------------------------------------------- Slip and velocity helpers
    def _get_effective_belief_pos_units(self, active_logical_idx: int) -> Tuple[float, float]:
        if active_logical_idx < 0 or active_logical_idx >= self._num_plates:
            return (0.0, 0.0)

        if float(self._persistence_timer[active_logical_idx]) < float(self.params.d_t):
            src = self._slip_source[active_logical_idx]
            if src is not None and 0 <= int(src) < self._num_plates and int(src) != int(active_logical_idx):
                return self._map_pos_units[int(src)]

        return self._map_pos_units[active_logical_idx]

    def _get_velocity_estimate_px_per_micro(self, plate_logical_idx: int) -> Tuple[float, float]:
        if plate_logical_idx < 0 or plate_logical_idx >= self._num_plates:
            return (0.0, 0.0)

        u = float(self.params.u)
        if u <= 1e-12:
            return (0.0, 0.0)

        dx_u, dy_u = self._avg_d[plate_logical_idx]
        vx_u_per_sec = dx_u / u
        vy_u_per_sec = dy_u / u

        vx_px_per_micro = (vx_u_per_sec * self._scale_px_per_unit) / float(self.fps)
        vy_px_per_micro = (vy_u_per_sec * self._scale_px_per_unit) / float(self.fps)
        return (vx_px_per_micro, vy_px_per_micro)

    # ---------------------------------------------------------------- PD controller
    def _pd_control(
        self,
        plate_logical_idx: int,
        core,
        r_hat_px: Tuple[float, float],
        v_hat_px_per_micro: Tuple[float, float],
    ) -> Tuple[float, float]:
        k_p = 0.02
        k_d = 0.6
        alpha = 0.8
        epsilon_tilt = 0.5

        if plate_logical_idx < 0 or plate_logical_idx >= self._num_plates:
            return (0.0, 0.0)

        internal_idx = core.get_internal_plate_index(plate_logical_idx)
        plate = core.plates[internal_idx]

        u_star_x = -k_p * float(r_hat_px[0]) - k_d * float(v_hat_px_per_micro[0])
        u_star_y = -k_p * float(r_hat_px[1]) - k_d * float(v_hat_px_per_micro[1])

        mag_star = math.hypot(u_star_x, u_star_y)
        if mag_star > MAX_TILT and mag_star > 1e-6:
            scale = MAX_TILT / mag_star
            u_star_x *= scale
            u_star_y *= scale

        delta_x = u_star_x - float(plate.x_tilt)
        delta_y = u_star_y - float(plate.y_tilt)
        E = math.hypot(delta_x, delta_y)
        if E <= epsilon_tilt:
            return (0.0, 0.0)

        dir_x = delta_x / E
        dir_y = delta_y / E

        delta_max_per_frame = (
            float(core.TILT_RATE) * float(core.JOYSTICK_SENSITIVITY)
            if hasattr(core, "TILT_RATE")
            else 0.48 * 3.4
        )
        m = min(1.0, alpha * E / max(delta_max_per_frame, 1e-6))
        return (m * dir_x, m * dir_y)

    # ---------------------------------------------------------------- Mapping utilities
    def _coords_to_bin(self, x_u: float, y_u: float) -> int:
        R = float(self._R_belief_units)
        xp = float(x_u) + R
        yp = float(y_u) + R
        if (xp < 0.0) or (xp >= 2.0 * R) or (yp < 0.0) or (yp >= 2.0 * R):
            return -1

        ix = int(math.floor(xp / float(self._bin_size)))
        iy = int(math.floor(yp / float(self._bin_size)))
        if ix < 0 or ix >= self._N or iy < 0 or iy >= self._N:
            return -1
        return ix * self._N + iy

    def _bin_center_units(self, bin_idx: int) -> Tuple[float, float]:
        if bin_idx < 0 or bin_idx >= self._N_bins:
            return (0.0, 0.0)
        return (float(self._centers_x[bin_idx]), float(self._centers_y[bin_idx]))

    def _get_crossed_bins(self, edge_bin: int, px: float, py: float, dx: float, dy: float) -> List[int]:
        R = float(self._R_belief_units)
        D = 2.0 * R

        m = max(abs(dx), abs(dy))
        if m > D and m > 1e-12:
            s = D / m
            dx *= s
            dy *= s

        svec = self._cross_s
        xs = px + svec * dx
        ys = py + svec * dy

        xp = xs + R
        yp = ys + R

        valid = (xp >= 0.0) & (xp < 2.0 * R) & (yp >= 0.0) & (yp < 2.0 * R)
        if not np.any(valid):
            return []

        xp_v = xp[valid]
        yp_v = yp[valid]

        ix = np.floor(xp_v / float(self._bin_size)).astype(np.int32)
        iy = np.floor(yp_v / float(self._bin_size)).astype(np.int32)

        in_grid = (ix >= 0) & (ix < self._N) & (iy >= 0) & (iy < self._N)
        if not np.any(in_grid):
            return []

        bins = (ix[in_grid] * self._N + iy[in_grid]).astype(np.int32)
        bins = bins[bins != int(edge_bin)]
        if bins.size == 0:
            return []

        uniq = np.unique(bins)
        return uniq.tolist()
