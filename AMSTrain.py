"""
Train the AMS supervisor policy with PPO against the cognitive gameplay agent
in the headless MultiPlateEnv (PlateGame.py) using a simple curriculum:

  - AMS3: N in {2,3}
  - AMS6: N in {2,3,4,6}
  - AMS9: N in {2,3,4,6,8,9}

Artifacts per stage are written to AMSTrainLogs/<stage>/:
  - config.json
  - progress.csv
  - metrics_per_N.csv
  - curves_main.png (if matplotlib is available)

Stage checkpoints are written to AMSTrained/{AMS3,AMS6,AMS9}.pt.

This script does not write user-study CSV logs (episodes.csv, switches.csv).
"""


from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Matplotlib is used only for saving PNG plots. We use Agg backend for headless.
try:
    import matplotlib

    matplotlib.use("Agg")  # no GUI
    import matplotlib.pyplot as plt

    _HAVE_MPL = True
except Exception:  # pragma: no cover - plotting is optional
    _HAVE_MPL = False

from PlateGame import GameConfig, Mode, MultiPlateEnv, R_USABLE  # type: ignore
from ControlMapping import PromptState  # type: ignore
from AMSCore import AMSNet, save_ams_checkpoint, _default_device

# --------------------------------------------------------------------------------------
# Curriculum specification


@dataclass
class StageSpec:
    name: str  # "AMS3", "AMS6", "AMS9"
    sampling_probs: Dict[int, float]  # N -> probability


STAGE_SPECS: Dict[str, StageSpec] = {
    "AMS3": StageSpec(
        name="AMS3",
        sampling_probs={
            2: 0.5,
            3: 0.5,
        },
    ),
    "AMS6": StageSpec(
        name="AMS6",
        sampling_probs={
            2: 0.10,
            3: 0.10,
            4: 0.40,
            6: 0.40,
        },
    ),
    "AMS9": StageSpec(
        name="AMS9",
        sampling_probs={
            2: 0.05,
            3: 0.05,
            4: 0.10,
            6: 0.10,
            8: 0.35,
            9: 0.35,
        },
    ),
}


# --------------------------------------------------------------------------------------
# Training hyperparameters & config


@dataclass
class PPOHyperParams:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    rollout_steps: int = 1024   # per env per PPO update
    num_epochs: int = 10
    minibatch_size: int = 1024 # across all envs and time steps


@dataclass
class TrainConfig:
    curricula: List[str]  # e.g., ["AMS3", "AMS6", "AMS9"]
    load_from_stage: Optional[str]
    num_envs: int
    total_env_steps_per_stage: int  # total AMS decisions across all envs
    device: str
    base_seed: int
    ppo: PPOHyperParams


# --------------------------------------------------------------------------------------
# Logging & metrics


LOG_ROOT = os.path.join(os.path.dirname(__file__), "AMSTrainLogs")


class EpisodeStatsAggregator:
    """Collects episodic statistics for one curriculum stage.

    We track:
        - global episode return & duration;
        - per-N stats (N = number of plates for that episode).

    The trainer calls `record_episode()` from the vectorized env whenever
    an episode finishes. At the *end* of each PPO rollout, the trainer
    calls `summarize_and_reset()` to get aggregate stats for that window.
    """

    def __init__(self) -> None:
        self._episodes: List[Tuple[int, float, float]] = []
        # each entry is (N, return, duration_sec)

    def record_episode(self, N: int, ep_return: float, duration_sec: float) -> None:
        self._episodes.append((N, ep_return, duration_sec))

    def summarize_and_reset(self) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
        """Summarize and clear the collected stats.

        Returns:
            global_stats: dict with keys
                - num_episodes
                - mean_return
                - std_return
                - mean_duration_sec
                - std_duration_sec

            per_N_stats: dict mapping N -> dict with keys
                - num_episodes
                - mean_return
                - mean_duration_sec
        """
        eps = self._episodes
        self._episodes = []

        global_stats: Dict[str, float] = {
            "num_episodes": 0.0,
            "mean_return": float("nan"),
            "std_return": float("nan"),
            "mean_duration_sec": float("nan"),
            "std_duration_sec": float("nan"),
        }
        per_N_stats: Dict[int, Dict[str, float]] = {}

        if not eps:
            return global_stats, per_N_stats

        returns = np.array([e[1] for e in eps], dtype=np.float32)
        durations = np.array([e[2] for e in eps], dtype=np.float32)
        Ns = np.array([e[0] for e in eps], dtype=np.int32)

        global_stats["num_episodes"] = float(len(eps))
        global_stats["mean_return"] = float(returns.mean())
        global_stats["std_return"] = float(returns.std())
        global_stats["mean_duration_sec"] = float(durations.mean())
        global_stats["std_duration_sec"] = float(durations.std())

        unique_Ns = np.unique(Ns)
        for N in unique_Ns:
            mask = Ns == N
            ret_N = returns[mask]
            dur_N = durations[mask]
            per_N_stats[int(N)] = {
                "num_episodes": float(len(ret_N)),
                "mean_return": float(ret_N.mean()),
                "mean_duration_sec": float(dur_N.mean()),
            }

        return global_stats, per_N_stats


class TrainLogger:
    """Handles writing training logs and plots for one curriculum stage.

    All artifacts go into AMSTrainLogs/<stage>/, and are overwritten by
    default when a new training run for that stage begins.
    """

    def __init__(self, stage: str, train_cfg: TrainConfig) -> None:
        self.stage = stage
        self.stage_dir = os.path.join(LOG_ROOT, stage)
        # Clear existing stage artifacts for this run.
        if os.path.isdir(self.stage_dir):
            for fname in os.listdir(self.stage_dir):
                try:
                    os.remove(os.path.join(self.stage_dir, fname))
                except OSError:
                    # If it's a subdir or something unexpected, ignore for now
                    pass
        else:
            os.makedirs(self.stage_dir, exist_ok=True)

        self.config_path = os.path.join(self.stage_dir, "config.json")
        self.progress_csv_path = os.path.join(self.stage_dir, "progress.csv")
        self.metrics_per_N_csv_path = os.path.join(self.stage_dir, "metrics_per_N.csv")

        # Save config.json once at the beginning
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "stage": stage,
                    "train_config": {
                        "curricula": train_cfg.curricula,
                        "load_from_stage": train_cfg.load_from_stage,
                        "num_envs": train_cfg.num_envs,
                        "total_env_steps_per_stage": train_cfg.total_env_steps_per_stage,
                        "device": train_cfg.device,
                        "base_seed": train_cfg.base_seed,
                    },
                    "ppo_hyperparams": asdict(train_cfg.ppo),
                },
                f,
                indent=2,
            )

        # Prepare CSV writers
        self._progress_file = open(self.progress_csv_path, "w", newline="", encoding="utf-8")
        self._metricsN_file = open(self.metrics_per_N_csv_path, "w", newline="", encoding="utf-8")

        self._progress_writer = None  # type: ignore
        self._metricsN_writer = None  # type: ignore

        self._progress_fieldnames = [
            "stage",
            "total_env_steps",
            "updates_done",
            "wall_time_sec",
            "steps_per_sec",
            "mean_ep_return",
            "std_ep_return",
            "mean_ep_duration_sec",
            "std_ep_duration_sec",
            "num_episodes",
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clip_fraction",
            "value_explained_var",
            "learning_rate",
            "mean_switch_rate",
            "mean_stay_rate",
            "belief_updates_total",
            "belief_visible_updates_total",
            "belief_sum_invalid_rate",
            "true_bin_oos_visible_rate",
            "true_bin_zero_mass_visible_rate",
            "crossed_bins_empty_per_update",
            "active_map_err_norm",
            "active_effective_err_norm",
            "inactive_map_err_norm",
        ]
        self._metricsN_fieldnames = [
            "stage",
            "total_env_steps",
            "N",
            "num_episodes",
            "mean_return",
            "mean_duration_sec",
        ]

        self._progress_writer = csv.DictWriter(self._progress_file, fieldnames=self._progress_fieldnames)
        self._progress_writer.writeheader()
        self._metricsN_writer = csv.DictWriter(self._metricsN_file, fieldnames=self._metricsN_fieldnames)
        self._metricsN_writer.writeheader()

        # For plotting the main curves
        self._steps_history: List[int] = []
        self._mean_return_history: List[float] = []
        self._mean_duration_history: List[float] = []

    def log_progress(
        self,
        total_env_steps: int,
        updates_done: int,
        wall_time_sec: float,
        steps_per_sec: float,
        global_stats: Dict[str, float],
        ppo_stats: Dict[str, float],
        lr: float,
        mean_switch_rate: float,
        mean_stay_rate: float,
        belief_debug: Dict[str, float],
    ) -> None:
        row = {
            "stage": self.stage,
            "total_env_steps": total_env_steps,
            "updates_done": updates_done,
            "wall_time_sec": wall_time_sec,
            "steps_per_sec": steps_per_sec,
            "mean_ep_return": global_stats.get("mean_return", float("nan")),
            "std_ep_return": global_stats.get("std_return", float("nan")),
            "mean_ep_duration_sec": global_stats.get("mean_duration_sec", float("nan")),
            "std_ep_duration_sec": global_stats.get("std_duration_sec", float("nan")),
            "num_episodes": global_stats.get("num_episodes", 0.0),
            "policy_loss": ppo_stats.get("policy_loss", float("nan")),
            "value_loss": ppo_stats.get("value_loss", float("nan")),
            "entropy": ppo_stats.get("entropy", float("nan")),
            "approx_kl": ppo_stats.get("approx_kl", float("nan")),
            "clip_fraction": ppo_stats.get("clip_fraction", float("nan")),
            "value_explained_var": ppo_stats.get("value_explained_var", float("nan")),
            "learning_rate": lr,
            "mean_switch_rate": mean_switch_rate,
            "mean_stay_rate": mean_stay_rate,
            "belief_updates_total": belief_debug.get("belief_updates_total", 0.0),
            "belief_visible_updates_total": belief_debug.get("belief_visible_updates_total", 0.0),
            "belief_sum_invalid_rate": belief_debug.get("belief_sum_invalid_rate", float("nan")),
            "true_bin_oos_visible_rate": belief_debug.get("true_bin_oos_visible_rate", float("nan")),
            "true_bin_zero_mass_visible_rate": belief_debug.get("true_bin_zero_mass_visible_rate", float("nan")),
            "crossed_bins_empty_per_update": belief_debug.get("crossed_bins_empty_per_update", float("nan")),
            "active_map_err_norm": belief_debug.get("active_map_err_norm", float("nan")),
            "active_effective_err_norm": belief_debug.get("active_effective_err_norm", float("nan")),
            "inactive_map_err_norm": belief_debug.get("inactive_map_err_norm", float("nan")),
        }
        self._progress_writer.writerow(row)
        self._progress_file.flush()

        # store for plotting
        self._steps_history.append(total_env_steps)
        self._mean_return_history.append(global_stats.get("mean_return", float("nan")))
        self._mean_duration_history.append(global_stats.get("mean_duration_sec", float("nan")))

    def log_metrics_per_N(
        self,
        total_env_steps: int,
        per_N_stats: Dict[int, Dict[str, float]],
    ) -> None:
        for N, stats in per_N_stats.items():
            row = {
                "stage": self.stage,
                "total_env_steps": total_env_steps,
                "N": N,
                "num_episodes": stats.get("num_episodes", 0.0),
                "mean_return": stats.get("mean_return", float("nan")),
                "mean_duration_sec": stats.get("mean_duration_sec", float("nan")),
            }
            self._metricsN_writer.writerow(row)
        if per_N_stats:
            self._metricsN_file.flush()

    def close(self) -> None:
        try:
            self._progress_file.close()
        except Exception: 
            pass
        try:
            self._metricsN_file.close()
        except Exception: 
            pass

        if not _HAVE_MPL or not self._steps_history:
            return
        try:
            self._plot_main_curves()
        except Exception:
            pass

    def _plot_main_curves(self) -> None:
        # Main curves: mean episode duration and mean return vs steps
        steps = np.array(self._steps_history, dtype=np.int64)
        mean_return = np.array(self._mean_return_history, dtype=np.float32)
        mean_dur = np.array(self._mean_duration_history, dtype=np.float32)

        finite_mask = np.isfinite(mean_return) & np.isfinite(mean_dur)
        steps = steps[finite_mask]
        mean_return = mean_return[finite_mask]
        mean_dur = mean_dur[finite_mask]

        # If nothing is finite, skip plotting entirely.
        if len(steps) == 0:
            return

        # Simple moving average for smoother visualization
        def moving_average(x, window: int = 5) -> np.ndarray:
            x = np.asarray(x, dtype=np.float32)
            if len(x) < window:
                return x
            # We assume x contains only finite values here (filtered above).
            cumsum = np.cumsum(np.insert(x, 0, 0.0))
            return (cumsum[window:] - cumsum[:-window]) / float(window)

        # We align smoothed curves with the last window positions
        window = 5
        if len(steps) >= window:
            sm_steps = steps[window - 1 :]
            sm_return = moving_average(mean_return, window=window)
            sm_dur = moving_average(mean_dur, window=window)
        else:
            sm_steps = steps
            sm_return = mean_return
            sm_dur = mean_dur

        if len(sm_steps) == 0:
            # Nothing meaningful to plot after smoothing
            return

        # curves_main.png
        plt.figure(figsize=(8, 5))
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax1.plot(sm_steps, sm_dur, label="Mean episode duration (s)", color="tab:blue")
        ax2.plot(sm_steps, sm_return, label="Mean return", color="tab:orange")
        ax1.set_xlabel("Env steps")
        ax1.set_ylabel("Episode duration (s)", color="tab:blue")
        ax2.set_ylabel("Return", color="tab:orange")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        plt.title(f"{self.stage}: Training curves")
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        plt.tight_layout()
        out_path = os.path.join(self.stage_dir, "curves_main.png")
        plt.savefig(out_path)
        plt.close()


# --------------------------------------------------------------------------------------
# Vectorized headless env wrapper


class AMSVecEnv:
    """Simple vectorized wrapper around multiple MultiPlateEnv instances.

    Provides:
        - reset() -> (obs_batch, mask_batch)
        - step(actions) -> (obs_batch, mask_batch, rewards, dones)

    Also:
        - handles curriculum sampling of N according to StageSpec.sampling_probs.
        - records episodic stats via EpisodeStatsAggregator.
        - tracks switch vs stay rates for debugging.
    """

    def __init__(
        self,
        num_envs: int,
        stage_spec: StageSpec,
        base_seed: int,
        episode_stats: EpisodeStatsAggregator,
        device: torch.device,
    ) -> None:
        self.num_envs = num_envs
        self.stage_spec = stage_spec
        self.base_seed = base_seed
        self.episode_stats = episode_stats
        self.device = device

        self.envs: List[MultiPlateEnv] = []
        self.current_Ns: List[int] = [0] * num_envs
        self.ep_returns: np.ndarray = np.zeros(num_envs, dtype=np.float32)
        self.ep_durations_sec: np.ndarray = np.zeros(num_envs, dtype=np.float32)

        # For switch/stay stats:
        self._switch_count = 0
        self._stay_count = 0
        self._step_count = 0
        self._env_reset_counts: List[int] = [0] * num_envs
        self._env_seed_stride: int = 1000003


        self._belief_stat_keys = (
            "belief_updates_total",
            "belief_visible_updates_total",
            "belief_sum_invalid",
            "true_bin_out_of_support_visible",
            "true_bin_zero_mass_visible",
            "crossed_bins_empty",
        )

        self._belief_prev_stats: List[Dict[str, int]] = [
            {k: 0 for k in self._belief_stat_keys} for _ in range(num_envs)
        ]

        self._belief_window_counts: Dict[str, int] = {k: 0 for k in self._belief_stat_keys}

        self._active_map_err_sum: float = 0.0
        self._active_effective_err_sum: float = 0.0
        self._inactive_map_err_sum: float = 0.0
        self._map_err_count: int = 0

        # Build configs and envs
        for i in range(num_envs):
            cfg = GameConfig(
                mode=Mode.TRAINING,
                num_plates=2,  # initial; overridden by reset(num_plates=...)
                participant_id=None,
                display_belief=False,
                debug_prompt=False,
                headless=True,
                seed=None,  # leave seed None; global RNG is seeded in main()
            )
            env = MultiPlateEnv(config=cfg, ams=None)
            self.envs.append(env)

    def _next_reset_seed(self, env_i: int) -> int:
        self._env_reset_counts[env_i] += 1
        s = int(self.base_seed) + int(env_i) * int(self._env_seed_stride) + int(self._env_reset_counts[env_i])
        return int(s & 0xFFFFFFFF)

    def _sample_N(self) -> int:
        # Sample N according to the fixed discrete distribution for this stage
        Ns = list(self.stage_spec.sampling_probs.keys())
        ps = np.array(list(self.stage_spec.sampling_probs.values()), dtype=np.float64)
        ps = ps / ps.sum()
        return int(np.random.choice(Ns, p=ps))

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Reset all envs with freshly sampled Ns.

        Returns:
            obs_batch: [num_envs, obs_dim] float32
            mask_batch: [num_envs, n_actions] int/0-1
        """
        obs_list: List[np.ndarray] = []
        mask_list: List[np.ndarray] = []

        self.ep_returns[:] = 0.0
        self.ep_durations_sec[:] = 0.0
        self.reset_belief_debug_window()

        for i, env in enumerate(self.envs):
            N = self._sample_N()
            self.current_Ns[i] = N
            seed_i = self._next_reset_seed(i)
            obs = env.reset(num_plates=N, seed=seed_i)
            self._belief_prev_stats[i] = self._read_belief_stats(env)
            prompt_busy = (env.prompt_controller.state != PromptState.IDLE)
            time_to_commit = 0.0
            if env.prompt_controller.state == PromptState.PRE_CUE and env.prompt_controller.commit_time is not None:
                time_to_commit = max(0.0, env.prompt_controller.commit_time - env.t_now)

            obs2, mask2 = env.core.build_ams_obs_and_mask(
                env.config, prompt_busy=prompt_busy, time_to_commit=time_to_commit
            )
            obs_arr = np.asarray(obs2, dtype=np.float32)
            mask_arr = np.asarray(mask2, dtype=np.int64)
            obs_list.append(obs_arr)
            mask_list.append(mask_arr)

        obs_batch = np.stack(obs_list, axis=0)
        mask_batch = np.stack(mask_list, axis=0)
        return obs_batch, mask_batch

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Step all envs once with the given actions.

        Args:
            actions: [num_envs] int array of logical plate indices.

        Returns:
            obs_batch: [num_envs, obs_dim] float32
            mask_batch: [num_envs, n_actions] int/0-1
            rewards: [num_envs] float32
            dones: [num_envs] bool
        """
        obs_list: List[np.ndarray] = []
        mask_list: List[np.ndarray] = []
        reward_list: List[float] = []
        done_list: List[bool] = []

        self._step_count += self.num_envs

        for i, (env, act) in enumerate(zip(self.envs, actions)):
            cp_logical = env.core.internal_to_logical[env.core.controlled_plate]
            if int(act) == int(cp_logical):
                self._stay_count += 1
            else:
                self._switch_count += 1

            obs, reward, done, info = env.step(int(act))
            self._accumulate_belief_debug_step(i, env, done)
            self.ep_returns[i] += float(reward)
            self.ep_durations_sec[i] = float(env.core.game_time)

            if done:
                # Record finished episode
                N_finished = self.current_Ns[i]
                self.episode_stats.record_episode(
                    N=N_finished,
                    ep_return=self.ep_returns[i],
                    duration_sec=self.ep_durations_sec[i],
                )
                # Reset this env with a new N
                N_new = self._sample_N()
                self.current_Ns[i] = N_new
                seed_i = self._next_reset_seed(i)
                obs = env.reset(num_plates=N_new, seed=seed_i)
                self._belief_prev_stats[i] = self._read_belief_stats(env)
                self.ep_returns[i] = 0.0
                self.ep_durations_sec[i] = 0.0
                # For new episode, build obs+mask from core
                prompt_busy = (env.prompt_controller.state != PromptState.IDLE)
                time_to_commit = 0.0
                if env.prompt_controller.state == PromptState.PRE_CUE and env.prompt_controller.commit_time is not None:
                    time_to_commit = max(0.0, env.prompt_controller.commit_time - env.t_now)

                obs2, mask2 = env.core.build_ams_obs_and_mask(
                    env.config, prompt_busy=prompt_busy, time_to_commit=time_to_commit
                )
                obs_arr = np.asarray(obs2, dtype=np.float32)
                mask_arr = np.asarray(mask2, dtype=np.int64)
            else:
                # Use obs + mask from step()
                mask_arr = np.asarray(info["action_mask"], dtype=np.int64)
                obs_arr = np.asarray(obs, dtype=np.float32)

            obs_list.append(obs_arr)
            mask_list.append(mask_arr)
            reward_list.append(float(reward))
            done_list.append(bool(done))

        obs_batch = np.stack(obs_list, axis=0)
        mask_batch = np.stack(mask_list, axis=0)
        rewards = np.asarray(reward_list, dtype=np.float32)
        dones = np.asarray(done_list, dtype=bool)
        return obs_batch, mask_batch, rewards, dones

    def get_switch_stay_rates(self) -> Tuple[float, float]:
        """Return (switch_rate, stay_rate) over all steps so far."""
        if self._step_count == 0:
            return 0.0, 0.0
        switch_rate = self._switch_count / float(self._step_count)
        stay_rate = self._stay_count / float(self._step_count)
        return switch_rate, stay_rate

    def reset_belief_debug_window(self) -> None:
        for k in self._belief_window_counts.keys():
            self._belief_window_counts[k] = 0
        self._active_map_err_sum = 0.0
        self._active_effective_err_sum = 0.0
        self._inactive_map_err_sum = 0.0
        self._map_err_count = 0

    def summarize_belief_debug_window(self) -> Dict[str, float]:
        counts = self._belief_window_counts
        updates = float(counts.get("belief_updates_total", 0))
        visible_updates = float(counts.get("belief_visible_updates_total", 0))

        def safe_div(num: float, den: float) -> float:
            if den <= 0.0:
                return float("nan")
            return num / den

        out: Dict[str, float] = {}
        out["belief_updates_total"] = updates
        out["belief_visible_updates_total"] = visible_updates

        out["belief_sum_invalid_rate"] = safe_div(float(counts.get("belief_sum_invalid", 0)), max(1.0, updates))
        out["true_bin_oos_visible_rate"] = safe_div(float(counts.get("true_bin_out_of_support_visible", 0)), max(1.0, visible_updates))
        out["true_bin_zero_mass_visible_rate"] = safe_div(float(counts.get("true_bin_zero_mass_visible", 0)), max(1.0, visible_updates))
        out["crossed_bins_empty_per_update"] = safe_div(float(counts.get("crossed_bins_empty", 0)), max(1.0, updates))

        if self._map_err_count > 0:
            out["active_map_err_norm"] = self._active_map_err_sum / float(self._map_err_count)
            out["active_effective_err_norm"] = self._active_effective_err_sum / float(self._map_err_count)
            out["inactive_map_err_norm"] = self._inactive_map_err_sum / float(self._map_err_count)
        else:
            out["active_map_err_norm"] = float("nan")
            out["active_effective_err_norm"] = float("nan")
            out["inactive_map_err_norm"] = float("nan")

        return out

    def _read_belief_stats(self, env: MultiPlateEnv) -> Dict[str, int]:
        s = getattr(env.cognitive_agent, "stats", {}) if hasattr(env, "cognitive_agent") else {}
        out: Dict[str, int] = {}
        for k in self._belief_stat_keys:
            try:
                out[k] = int(s.get(k, 0))
            except Exception:
                out[k] = 0
        return out

    def _accumulate_belief_debug_step(self, env_i: int, env: MultiPlateEnv, done: bool) -> None:
        cur = self._read_belief_stats(env)
        prev = self._belief_prev_stats[env_i]

        for k in self._belief_stat_keys:
            d = int(cur.get(k, 0)) - int(prev.get(k, 0))
            if d < 0:
                d = int(cur.get(k, 0))
            self._belief_window_counts[k] += d

        self._belief_prev_stats[env_i] = cur

        if done:
            return

        core = env.core
        cog = env.cognitive_agent
        N = int(core.num_plates)
        if N <= 0:
            return

        cp_internal = int(core.controlled_plate)
        cp_logical = int(core.internal_to_logical[cp_internal])

        bx, by = cog.get_belief_position_px(cp_logical)
        bex, bey = cog.get_effective_belief_position_px(cp_logical)

        ball = core.balls[cp_internal]
        tx, ty = float(ball.x), float(ball.y)

        active_err = math.hypot(bx - tx, by - ty) / float(R_USABLE)
        active_eff_err = math.hypot(bex - tx, bey - ty) / float(R_USABLE)

        inactive_sum = 0.0
        inactive_count = 0
        for logical_j in range(N):
            if logical_j == cp_logical:
                continue
            internal_j = int(core.get_internal_plate_index(logical_j))
            bj = core.balls[internal_j]
            bjx, bjy = cog.get_belief_position_px(logical_j)
            inactive_sum += math.hypot(bjx - float(bj.x), bjy - float(bj.y)) / float(R_USABLE)
            inactive_count += 1

        inactive_err = (inactive_sum / float(inactive_count)) if inactive_count > 0 else 0.0

        self._active_map_err_sum += float(active_err)
        self._active_effective_err_sum += float(active_eff_err)
        self._inactive_map_err_sum += float(inactive_err)
        self._map_err_count += 1

# --------------------------------------------------------------------------------------
# PPO training logic


class PPOTrainer:
    """PPO trainer for the AMS policy.

    This class is specific to the AMS + MultiPlateEnv setting but keeps
    algorithmic logic reasonably generic.
    """

    def __init__(
        self,
        policy: AMSNet,
        vec_env: AMSVecEnv,
        train_cfg: TrainConfig,
        stage_spec: StageSpec,
        logger: TrainLogger,
        device: torch.device,
    ) -> None:
        self.policy = policy
        self.vec_env = vec_env
        self.train_cfg = train_cfg
        self.stage_spec = stage_spec
        self.logger = logger
        self.device = device

        self.hp = train_cfg.ppo

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.hp.lr)

        self.global_env_steps = 0  # across all envs
        self.num_updates_done = 0

        self.start_wall_time = time.perf_counter()

    def train_stage(self) -> None:
        """Train the policy for one curriculum stage."""
        num_envs = self.train_cfg.num_envs
        rollout_steps = self.hp.rollout_steps
        total_steps_target = self.train_cfg.total_env_steps_per_stage

        obs, mask = self.vec_env.reset()
        obs = obs.astype(np.float32)
        mask = mask.astype(np.int64)

        obs_dim = self.policy.obs_dim
        n_actions = self.policy.n_actions

        obs_buf = np.zeros((rollout_steps, num_envs, obs_dim), dtype=np.float32)
        mask_buf = np.zeros((rollout_steps, num_envs, n_actions), dtype=np.int64)
        actions_buf = np.zeros((rollout_steps, num_envs), dtype=np.int64)
        logprobs_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        rewards_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        dones_buf = np.zeros((rollout_steps, num_envs), dtype=np.bool_)
        values_buf = np.zeros((rollout_steps + 1, num_envs), dtype=np.float32)

        while self.global_env_steps < total_steps_target:
            rollout_start_time = time.perf_counter()
            episode_stats = self.vec_env.episode_stats
            episode_stats._episodes = []  # clear window stats
            self.vec_env.reset_belief_debug_window()

            for t in range(rollout_steps):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device)

                with torch.no_grad():
                    dist, values = self.policy.masked_action_distribution(obs_t, mask_t)
                    actions = dist.sample()
                    logprobs = dist.log_prob(actions)

                actions_np = np.asarray(actions.cpu().tolist(), dtype=np.int64)
                logprobs_np = np.asarray(logprobs.cpu().tolist(), dtype=np.float32)
                values_np = np.asarray(values.cpu().tolist(), dtype=np.float32)

                next_obs, next_mask, rewards, dones = self.vec_env.step(actions_np)
                next_obs = next_obs.astype(np.float32)
                next_mask = next_mask.astype(np.int64)

                obs_buf[t] = obs
                mask_buf[t] = mask
                actions_buf[t] = actions_np
                logprobs_buf[t] = logprobs_np
                rewards_buf[t] = rewards
                dones_buf[t] = dones
                values_buf[t] = values_np

                obs, mask = next_obs, next_mask
                self.global_env_steps += num_envs

            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            with torch.no_grad():
                _, next_values = self.policy.masked_action_distribution(obs_t, mask_t)
            values_buf[rollout_steps] = np.asarray(
                next_values.cpu().tolist(), dtype=np.float32
            )

            advantages, returns = self._compute_gae(
                rewards_buf, dones_buf, values_buf
            )

            batch_obs = obs_buf.reshape(-1, obs_dim)
            batch_mask = mask_buf.reshape(-1, n_actions)
            batch_actions = actions_buf.reshape(-1)
            batch_logprobs = logprobs_buf.reshape(-1)
            batch_advantages = advantages.reshape(-1)
            batch_returns = returns.reshape(-1)
            batch_values = values_buf[:-1].reshape(-1)

            adv_mean = batch_advantages.mean()
            adv_std = batch_advantages.std() + 1e-8
            batch_advantages = (batch_advantages - adv_mean) / adv_std

            ppo_stats = self._ppo_update(
                batch_obs=batch_obs,
                batch_mask=batch_mask,
                batch_actions=batch_actions,
                batch_old_logprobs=batch_logprobs,
                batch_advantages=batch_advantages,
                batch_returns=batch_returns,
                batch_old_values=batch_values,
            )
            self.num_updates_done += 1

            global_stats, per_N_stats = episode_stats.summarize_and_reset()
            rollout_wall_time = time.perf_counter() - rollout_start_time
            steps_this_rollout = num_envs * rollout_steps
            steps_per_sec = steps_this_rollout / max(rollout_wall_time, 1e-8)

            switch_rate, stay_rate = self.vec_env.get_switch_stay_rates()
            belief_debug = self.vec_env.summarize_belief_debug_window()

            elapsed_wall = time.perf_counter() - self.start_wall_time

            progress = min(1.0, self.global_env_steps / float(total_steps_target))
            base_lr = self.hp.lr
            new_lr = base_lr * (1.0 - progress)
            for pg in self.optimizer.param_groups:
                pg["lr"] = new_lr

            current_lr = new_lr
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.logger.log_progress(
                total_env_steps=self.global_env_steps,
                updates_done=self.num_updates_done,
                wall_time_sec=elapsed_wall,
                steps_per_sec=steps_per_sec,
                global_stats=global_stats,
                ppo_stats=ppo_stats,
                lr=current_lr,
                mean_switch_rate=switch_rate,
                mean_stay_rate=stay_rate,
                belief_debug=belief_debug,
            )
            self.logger.log_metrics_per_N(
                total_env_steps=self.global_env_steps,
                per_N_stats=per_N_stats,
            )

            mean_ret = global_stats.get("mean_return", float("nan"))
            mean_dur = global_stats.get("mean_duration_sec", float("nan"))
            print(
                f"[{self.stage_spec.name}] steps={self.global_env_steps:,} "
                f"upd={self.num_updates_done} "
                f"mean_ret={mean_ret:.3f} "
                f"dur={mean_dur:.2f}s "
                f"KL={ppo_stats.get('approx_kl', float('nan')):.4f} "
                f"clip={ppo_stats.get('clip_fraction', float('nan')):.2f} "
                f"ent={ppo_stats.get('entropy', float('nan')):.2f} "
                f"sw={switch_rate:.2f} st={stay_rate:.2f}"
            )

        print(f"[{self.stage_spec.name}] Training finished at {self.global_env_steps:,} env steps.")

    def _compute_gae(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and returns from rollout buffers.

        Args:
            rewards: [T, N] float32
            dones: [T, N] bool
            values: [T+1, N] float32

        Returns:
            advantages: [T, N]
            returns: [T, N]
        """
        T, N = rewards.shape
        advantages = np.zeros((T, N), dtype=np.float32)
        last_adv = np.zeros(N, dtype=np.float32)

        gamma = self.hp.gamma
        lam = self.hp.gae_lambda

        for t in reversed(range(T)):
            mask = 1.0 - dones[t].astype(np.float32)
            delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
            last_adv = delta + gamma * lam * mask * last_adv
            advantages[t] = last_adv

        returns = advantages + values[:-1]
        return advantages, returns

    def _ppo_update(
        self,
        batch_obs: np.ndarray,
        batch_mask: np.ndarray,
        batch_actions: np.ndarray,
        batch_old_logprobs: np.ndarray,
        batch_advantages: np.ndarray,
        batch_returns: np.ndarray,
        batch_old_values: np.ndarray,
    ) -> Dict[str, float]:
        """One PPO update over the collected rollout batch.

        Returns a dict of aggregate stats (for logging).
        """
        device = self.device
        hp = self.hp

        num_samples = batch_obs.shape[0]
        batch_inds = np.arange(num_samples)

        # Convert everything to torch tensors on the proper device
        obs_t = torch.as_tensor(batch_obs, dtype=torch.float32, device=device)
        mask_t = torch.as_tensor(batch_mask, dtype=torch.bool, device=device)
        actions_t = torch.as_tensor(batch_actions, dtype=torch.long, device=device)
        old_logprobs_t = torch.as_tensor(batch_old_logprobs, dtype=torch.float32, device=device)
        advantages_t = torch.as_tensor(batch_advantages, dtype=torch.float32, device=device)
        returns_t = torch.as_tensor(batch_returns, dtype=torch.float32, device=device)
        old_values_t = torch.as_tensor(batch_old_values, dtype=torch.float32, device=device)

        # Stats we accumulate
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clip_frac = 0.0
        total_num_mb = 0

        for epoch in range(hp.num_epochs):
            np.random.shuffle(batch_inds)
            for start in range(0, num_samples, hp.minibatch_size):
                end = start + hp.minibatch_size
                mb_inds_np = batch_inds[start:end]
                # Convert numpy indices to a torch LongTensor for safe indexing
                mb_inds = torch.as_tensor(mb_inds_np, dtype=torch.long, device=device)

                mb_obs = obs_t[mb_inds]
                mb_mask = mask_t[mb_inds]
                mb_actions = actions_t[mb_inds]
                mb_old_logprobs = old_logprobs_t[mb_inds]
                mb_advantages = advantages_t[mb_inds]
                mb_returns = returns_t[mb_inds]
                mb_old_values = old_values_t[mb_inds]

                dist, values = self.policy.masked_action_distribution(mb_obs, mb_mask)
                new_logprobs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # Policy loss
                log_ratio = new_logprobs - mb_old_logprobs
                ratio = log_ratio.exp()
                # Surrogate objectives
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - hp.clip_coef, 1.0 + hp.clip_coef) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (no clipping for simplicity)
                value_loss = 0.5 * (values - mb_returns).pow(2).mean()

                # Combine
                loss = policy_loss + hp.vf_coef * value_loss - hp.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), hp.max_grad_norm)
                self.optimizer.step()

                # Bookkeeping for stats
                with torch.no_grad():
                    approx_kl = (
                        (mb_old_logprobs - new_logprobs).mean().cpu().item()
                    )  # positive if new bigger than old on average
                    clip_frac = (torch.abs(ratio - 1.0) > hp.clip_coef).float().mean().cpu().item()

                total_policy_loss += policy_loss.detach().cpu().item()
                total_value_loss += value_loss.detach().cpu().item()
                total_entropy += entropy.detach().cpu().item()
                total_approx_kl += approx_kl
                total_clip_frac += clip_frac
                total_num_mb += 1

        # Compute explained variance of value function
        with torch.no_grad():
            v_pred = old_values_t
            v_true = returns_t
            var_y = torch.var(v_true)
            if var_y.item() > 1e-8:
                value_explained_var = 1.0 - torch.var(v_true - v_pred) / var_y
                value_explained_var = value_explained_var.cpu().item()
            else:
                value_explained_var = float("nan")

        # Aggregate stats
        if total_num_mb > 0:
            policy_loss_mean = total_policy_loss / total_num_mb
            value_loss_mean = total_value_loss / total_num_mb
            entropy_mean = total_entropy / total_num_mb
            approx_kl_mean = total_approx_kl / total_num_mb
            clip_frac_mean = total_clip_frac / total_num_mb
        else:
            policy_loss_mean = float("nan")
            value_loss_mean = float("nan")
            entropy_mean = float("nan")
            approx_kl_mean = float("nan")
            clip_frac_mean = float("nan")

        return {
            "policy_loss": policy_loss_mean,
            "value_loss": value_loss_mean,
            "entropy": entropy_mean,
            "approx_kl": approx_kl_mean,
            "clip_fraction": clip_frac_mean,
            "value_explained_var": value_explained_var,
        }


# --------------------------------------------------------------------------------------
# CLI & main
#
# Checkpoint selection differs between training and gameplay:
# - Training (this script) starts from scratch unless --load-from is provided.
# - Gameplay (PlateGame.py) may use "auto" to pick the best available stage.



def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train AMS supervisor with PPO")

    parser.add_argument(
        "--curricula",
        type=str,
        default="3,6,9",
        help="Comma-separated list of curricula stages to train (subset of 3,6,9). Default: '3,6,9'",
    )
    parser.add_argument(
        "--load-from",
        type=str,
        default=None,
        choices=["3", "6", "9"],
        help="Optional stage to load initial AMS weights from (3, 6, or 9). "
        "If omitted, training starts from scratch.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=30,
        help="Number of parallel MultiPlateEnv instances. Default: 30",
    )
    parser.add_argument(
        "--total-env-steps-per-stage",
        type=int,
        default=3_000_000,
        help="Total number of environment steps per curriculum stage "
        "(across all envs). Default: 3,000,000",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device for AMS policy (e.g., 'cuda', 'cpu'). "
        "Default: auto (prefer GPU if available).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Base random seed for training. Used to seed environments. Default: 12345",
    )

    # PPO-specific overrides (optional)
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=1024,
        help="Rollout horizon (steps per env per PPO update). Default: 1024",
    )
    parser.add_argument(
        "--ppo-epochs",
        type=int,
        default=10,
        help="Number of PPO epochs per update. Default: 10",
    )
    parser.add_argument(
        "--ppo-minibatch-size",
        type=int,
        default=1024,
        help="PPO minibatch size (across time and envs). Default: 1024",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="PPO learning rate. Default: 3e-4",
    )

    args = parser.parse_args()

    # Resolve curricula list
    raw_stages = [s.strip() for s in args.curricula.split(",") if s.strip()]
    valid_stage_nums = {"3": "AMS3", "6": "AMS6", "9": "AMS9"}
    curricula: List[str] = []
    for s in raw_stages:
        if s not in valid_stage_nums:
            raise ValueError(f"Invalid curricula stage '{s}'. Expected subset of 3,6,9.")
        curricula.append(valid_stage_nums[s])

    # Deduplicate and sort in ascending order of 3,6,9
    order_map = {"AMS3": 3, "AMS6": 6, "AMS9": 9}
    curricula = sorted(set(curricula), key=lambda name: order_map[name])

    load_from_stage: Optional[str]
    if args.load_from is None:
        load_from_stage = None
    else:
        load_from_stage = valid_stage_nums[args.load_from]

    device = _default_device(args.device)

    ppo_hp = PPOHyperParams(
        lr=args.learning_rate,
        rollout_steps=args.rollout_steps,
        num_epochs=args.ppo_epochs,
        minibatch_size=args.ppo_minibatch_size,
    )

    cfg = TrainConfig(
        curricula=curricula,
        load_from_stage=load_from_stage,
        num_envs=args.num_envs,
        total_env_steps_per_stage=args.total_env_steps_per_stage,
        device=str(device),
        base_seed=args.seed,
        ppo=ppo_hp,
    )
    return cfg


def build_initial_policy(train_cfg: TrainConfig) -> AMSNet:
    """
    Create a new AMSNet from scratch.

    Observations are assumed to be pre-normalized by the environment
    (MultiPlateCore.build_ams_obs_and_mask), so no running normalization
    is applied in the model.
    """

    probe_cfg = GameConfig(
        mode=Mode.TRAINING,
        num_plates=2,
        participant_id=None,
        display_belief=False,
        debug_prompt=False,
        headless=True,
        seed=train_cfg.base_seed,
    )
    probe_env = MultiPlateEnv(config=probe_cfg, ams=None)
    obs = probe_env.reset(num_plates=2, seed=train_cfg.base_seed)
    obs_arr = np.asarray(obs, dtype=np.float32)
    obs_dim = obs_arr.shape[0]

    n_actions = probe_cfg.N_max  # corresponds to maximum logical plate slots
    policy = AMSNet(obs_dim=obs_dim, n_actions=n_actions)
    return policy


def main() -> None:
    train_cfg = parse_args()
    device = _default_device(train_cfg.device)
    print(f"Using device: {device}")
    print(f"Curricula to train: {train_cfg.curricula}")
    if train_cfg.load_from_stage is not None:
        print(f"Will initialize AMS policy from stage: {train_cfg.load_from_stage}")
    else:
        print("Training AMS policy from scratch (no initial checkpoint).")

    # Build initial policy
    if train_cfg.load_from_stage is None:
        policy = build_initial_policy(train_cfg)
        policy.to(device)
    else:
        from AMSCore import load_ams_checkpoint

        policy, meta = load_ams_checkpoint(
            stage=train_cfg.load_from_stage,
            device=str(device),
            path=None,
        )
        print(
            f"Loaded AMS checkpoint for stage {meta.stage}: "
            f"obs_dim={meta.obs_dim}, n_actions={meta.n_actions}"
        )

    # Make sure policy is on the correct device
    policy.to(device)

    # Fix seeds for Python and NumPy for reproducibility
    random.seed(train_cfg.base_seed)
    np.random.seed(train_cfg.base_seed)
    torch.manual_seed(train_cfg.base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_cfg.base_seed)

    # Train each requested curriculum stage in order
    for stage_name in train_cfg.curricula:
        stage_spec = STAGE_SPECS[stage_name]
        print(f"\n=== Training stage {stage_name} (Ns={list(stage_spec.sampling_probs.keys())}) ===")

        # Fresh episode stats aggregator and logger per stage
        ep_stats = EpisodeStatsAggregator()
        logger = TrainLogger(stage=stage_name, train_cfg=train_cfg)

        # Build vectorized env for this stage
        vec_env = AMSVecEnv(
            num_envs=train_cfg.num_envs,
            stage_spec=stage_spec,
            base_seed=train_cfg.base_seed,
            episode_stats=ep_stats,
            device=device,
        )

        trainer = PPOTrainer(
            policy=policy,
            vec_env=vec_env,
            train_cfg=train_cfg,
            stage_spec=stage_spec,
            logger=logger,
            device=device,
        )

        trainer.train_stage()
        logger.close()

        # Save checkpoint for this stage
        ckpt_path = save_ams_checkpoint(policy, stage=stage_name)
        print(f"[{stage_name}] Saved AMS checkpoint to: {ckpt_path}")

    print("\nAll requested curricula have been trained.")


if __name__ == "__main__":
    main()
