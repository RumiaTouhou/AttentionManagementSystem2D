"""
Core AMS components:

- AMSNet: PyTorch MLP policy/value network.
- PPOAMS: Runtime wrapper implementing PlateGame.AMSInterface.
- Checkpoint save/load helpers.

Checkpoints are stored under AMSTrained/ and are discovered by stage name
("AMS3", "AMS6", "AMS9") using the convention {stage} or {stage}.*.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # PlateGame.py defines AMSInterface; we reuse it for runtime integration.
    from PlateGame import AMSInterface  # type: ignore
except ImportError:  # pragma: no cover
    class AMSInterface:  # Fallback when PlateGame is unavailable.
        def select_action(self, obs, action_mask) -> int:
            raise NotImplementedError


# --------------------------------------------------------------------------------------
# Constants & small helpers


AMSTRAINED_DIR = os.path.join(os.path.dirname(__file__), "AMSTrained")

# We standardize on these three stage labels for curricula.
VALID_AMS_STAGES = ("AMS3", "AMS6", "AMS9")


def _normalize_stage_name(stage: str) -> str:
    """Validate and normalize the AMS stage name.

    Allowed values (case-insensitive): "AMS3", "AMS6", "AMS9".
    Raises ValueError if an unsupported name is given.
    """
    if not isinstance(stage, str):
        raise ValueError(f"Stage name must be string, got {type(stage)!r}")
    upper = stage.upper()
    if upper not in VALID_AMS_STAGES:
        raise ValueError(
            f"Unsupported AMS stage '{stage}'. Expected one of {VALID_AMS_STAGES}."
        )
    return upper


def _ensure_amstrained_dir() -> None:
    """Ensure the AMSTrained directory exists."""
    os.makedirs(AMSTRAINED_DIR, exist_ok=True)


def _default_device(device: Optional[str] = None) -> torch.device:
    """Resolve a device string to a torch.device, with a sane default."""
    if device is None:
        # Prefer GPU if available.
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _find_checkpoint_path_for_stage(stage: str) -> str:
    """Find a checkpoint file for a given stage in AMSTrained/.

    This mirrors the discovery logic in PlateGame.load_trained_ams:

    - Look in AMSTrained/ for a file whose basename either:
        - equals the stage name exactly, or
        - starts with `stage + "."` (e.g., "AMS3.pt", "AMS3.ckpt")

    Returns the absolute path to the first matching file (sorted order).
    Raises FileNotFoundError if nothing is found.
    """
    _ensure_amstrained_dir()
    base = _normalize_stage_name(stage)
    if not os.path.isdir(AMSTRAINED_DIR):
        raise FileNotFoundError(
            f"AMSTrained directory '{AMSTRAINED_DIR}' does not exist."
        )

    candidates = sorted(os.listdir(AMSTRAINED_DIR))
    for fname in candidates:
        if fname == base or fname.startswith(base + "."):
            return os.path.join(AMSTRAINED_DIR, fname)

    raise FileNotFoundError(
        f"No AMS checkpoint found for stage '{base}' in '{AMSTRAINED_DIR}'. "
        f"Expected a file named '{base}' or starting with '{base}.'"
    )


# --------------------------------------------------------------------------------------
# Policy/Value network


class AMSNet(nn.Module):
    """MLP policy/value network for the AMS.

    - Input: flat observation vector (obs_dim).
    - Output:
        - Policy logits over n_actions.
        - Scalar state-value estimate.

    This network is deliberately small and general-purpose; training
    code can choose its own optimizer and PPO hyperparameters.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_size: int = 128,
        num_hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        if obs_dim <= 0:
            raise ValueError(f"obs_dim must be positive, got {obs_dim}")
        if n_actions <= 0:
            raise ValueError(f"n_actions must be positive, got {n_actions}")
        if num_hidden_layers < 1:
            raise ValueError(f"num_hidden_layers must be >=1, got {num_hidden_layers}")

        layers = []
        in_dim = obs_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            in_dim = hidden_size
        self.backbone = nn.Sequential(*layers)

        self.policy_head = nn.Linear(hidden_size, n_actions)
        self.value_head = nn.Linear(hidden_size, 1)

        self.obs_dim = obs_dim
        self.n_actions = n_actions

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            obs: tensor of shape [B, obs_dim]

        Returns:
            logits: tensor of shape [B, n_actions]
            values: tensor of shape [B] (flattened value estimates)
        """
        if obs.dim() != 2 or obs.size(-1) != self.obs_dim:
            raise ValueError(
                f"Expected obs shape [B, {self.obs_dim}], got {tuple(obs.shape)}"
            )
        x = self.backbone(obs)
        logits = self.policy_head(x)
        values = self.value_head(x).squeeze(-1)
        return logits, values

    def masked_action_distribution(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> Tuple[torch.distributions.Categorical, torch.Tensor]:
        """Build a Categorical distribution over valid actions, given a mask.

        Args:
            obs: [B, obs_dim] float32 tensor.
            action_mask: [B, n_actions] bool or {0,1} tensor. True/1 => valid.

        Returns:
            dist: Categorical distribution over actions after masking.
            values: [B] value estimates.

        Notes:
            - Invalid actions get logits of -1e9 before softmax.
            - If a row mask has no valid actions (all zeros), all actions are
              treated as valid for that row (defensive fallback).
        """
        if action_mask.shape != (obs.shape[0], self.n_actions):
            raise ValueError(
                f"Expected action_mask shape [B, {self.n_actions}], got {tuple(action_mask.shape)}"
            )

        logits, values = self(obs)

        mask_bool = action_mask.bool()
        # Detect rows with no valid actions
        valid_counts = mask_bool.sum(dim=-1)  # [B]
        # For rows with zero valid actions, treat all as valid
        all_invalid = valid_counts == 0
        if all_invalid.any():
            # Fallback for corrupted masks: treat all actions as valid.
            mask_bool[all_invalid] = True

        # Very negative logits on invalid actions => probability ~ 0
        masked_logits = logits.masked_fill(~mask_bool, -1e9)

        probs = F.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        return dist, values


# --------------------------------------------------------------------------------------
# Checkpoint save/load


@dataclass
class AMSCheckpointMeta:
    """Metadata stored alongside the AMS model in a checkpoint."""

    stage: str
    obs_dim: int
    n_actions: int
    version: int = 1


def save_ams_checkpoint(
    policy: AMSNet,
    stage: str,
    *,
    path: Optional[str] = None,
) -> str:
    """Save an AMSNet policy checkpoint for a given curriculum stage.

    Args:
        policy: the trained AMSNet instance.
        stage: one of "AMS3", "AMS6", "AMS9" (case-insensitive).
        path: optional explicit path to save to. If None, defaults to
              AMSTrained/AMS{3,6,9}.pt.

    Returns:
        The absolute path to the written checkpoint file.

    Behavior:
        - Creates the AMSTrained/ directory if needed.
        - Overwrites any existing file at the chosen path.
        - Stores a dict with:
            - "meta": AMSCheckpointMeta as a dict
            - "state_dict": model state_dict
    """
    _ensure_amstrained_dir()
    stage_norm = _normalize_stage_name(stage)

    if path is None:
        filename = f"{stage_norm}.pt"
        path = os.path.join(AMSTRAINED_DIR, filename)
    else:
        # If caller passed a relative path, anchor it under AMSTrained/ for consistency
        if not os.path.isabs(path):
            path = os.path.join(AMSTRAINED_DIR, path)

    meta = AMSCheckpointMeta(
        stage=stage_norm,
        obs_dim=policy.obs_dim,
        n_actions=policy.n_actions,
        version=1,
    )

    payload: Dict[str, Any] = {
        "meta": meta.__dict__,
        "state_dict": policy.state_dict(),
    }
    torch.save(payload, path)
    return path


def load_ams_checkpoint(
    stage: str,
    *,
    device: Optional[str] = None,
    path: Optional[str] = None,
) -> Tuple[AMSNet, AMSCheckpointMeta]:
    """Load an AMSNet policy and its metadata from a checkpoint.

    Args:
        stage: one of "AMS3", "AMS6", "AMS9" (case-insensitive).
               Used to find the checkpoint if `path` is None.
        device: optional device string ("cpu", "cuda", ...). If None,
                a default is chosen (prefer GPU if available).
        path: optional explicit checkpoint path. If None, we search
              under AMSTrained/ using the same convention as
              PlateGame.load_trained_ams (stage or stage.*).

    Returns:
        (policy, meta) where:
            - policy is an AMSNet instance loaded onto the requested device.
            - meta is an AMSCheckpointMeta.

    Raises:
        FileNotFoundError if no checkpoint is found.
        ValueError if checkpoint contents are malformed.
    """
    dev = _default_device(device)
    stage_norm = _normalize_stage_name(stage)

    if path is None:
        path = _find_checkpoint_path_for_stage(stage_norm)

    payload = torch.load(path, map_location=dev)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Checkpoint at '{path}' is not a dict; got type {type(payload)!r}"
        )
    if "meta" not in payload or "state_dict" not in payload:
        raise ValueError(
            f"Checkpoint at '{path}' missing 'meta' or 'state_dict' keys."
        )

    meta_raw = payload["meta"]
    if not isinstance(meta_raw, dict):
        raise ValueError(
            f"Checkpoint meta at '{path}' is not a dict; got type {type(meta_raw)!r}"
        )

    try:
        meta = AMSCheckpointMeta(**meta_raw)
    except TypeError as exc:
        raise ValueError(f"Invalid checkpoint meta in '{path}': {exc}") from exc

    if meta.stage != stage_norm:
        # Stage mismatch; raise to avoid silent confusion.
        raise ValueError(
            f"Checkpoint at '{path}' was saved for stage '{meta.stage}', "
            f"but load_ams_checkpoint was called with stage '{stage_norm}'."
        )

    state_dict = payload["state_dict"]
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"Checkpoint state_dict at '{path}' is not a dict; got {type(state_dict)!r}"
        )

    policy = AMSNet(
        obs_dim=meta.obs_dim,
        n_actions=meta.n_actions,
    )
    policy.load_state_dict(state_dict)
    policy.to(dev)
    policy.eval()  # runtime usage is inference-only by default
    return policy, meta


# --------------------------------------------------------------------------------------
# Runtime wrapper for GameInteractive (implements AMSInterface)


class PPOAMS(AMSInterface):
    """Runtime AMS implementation that wraps an AMSNet policy.

    This class implements the `select_action(obs, action_mask)` method
    expected by `GameInteractive` and `SingleHandPromptController`:

        - `obs`: flat numpy array or list of floats (AMS observation).
        - `action_mask`: list/array of 0/1 flags indicating valid actions.

    The wrapper:
        - Moves data to the configured device.
        - Applies the action mask via AMSNet.masked_action_distribution.
        - Samples an action (stochastic or deterministic).
        - Returns the action index as a Python int (logical plate index).
    """

    def __init__(
        self,
        policy: AMSNet,
        *,
        device: Optional[str] = None,
        deterministic: bool = False,
        tie_break_eps: float = 0.03,
        tie_break_sample: bool = True,
    ) -> None:
        self.device = _default_device(device)
        self.policy = policy.to(self.device)
        self.policy.eval()
        self.deterministic = deterministic

        # If deterministic=True and the top-2 probabilities are within tie_break_eps, sample instead of argmax.
        self.tie_break_eps = float(tie_break_eps)
        self.tie_break_sample = bool(tie_break_sample)
        # Cached probabilities for optional logging/debugging.
        self.last_action_probs: Optional[List[float]] = None
        self.last_top_actions: Optional[List[int]] = None
        self.last_top_probs: Optional[List[float]] = None


    @classmethod
    def from_stage(
        cls,
        stage: str,
        *,
        device: Optional[str] = None,
        deterministic: bool = False,
        tie_break_eps: float = 0.03,
        tie_break_sample: bool = True,
    ) -> PPOAMS:
        """Convenience constructor that loads a checkpointed policy by stage."""
        policy, _ = load_ams_checkpoint(stage, device=device)
        return cls(
            policy=policy,
            device=device,
            deterministic=deterministic,
            tie_break_eps=tie_break_eps,
            tie_break_sample=tie_break_sample,
        )

    def select_action(self, obs, action_mask) -> int:  # type: ignore[override]
        """Select a single action for the current AMS observation.

        Args:
            obs: 1D array-like of floats; must match the obs_dim the policy was
                 trained on.
            action_mask: 1D array-like of {0,1} or bools, length n_actions.

        Returns:
            action index (int) in [0, n_actions).

        Notes:
            - If all entries in `action_mask` are 0/False, we fall back to
              treating all actions as valid.
            - If `deterministic=True`, we pick the argmax action by default,
              but may sample instead if tie_break_sample=True and the top-1 and
              top-2 action probabilities are within tie_break_eps (to prevent
              argmax lock-in when the policy is uncertain).
        """

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(
            1, -1
        )
        if obs_t.size(-1) != self.policy.obs_dim:
            raise ValueError(
                f"AMS policy expects obs_dim={self.policy.obs_dim}, "
                f"but got obs of length {obs_t.size(-1)}"
            )


        mask_arr = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
        mask_arr = mask_arr.view(1, -1)
        if mask_arr.size(-1) != self.policy.n_actions:
            raise ValueError(
                f"AMS policy expects n_actions={self.policy.n_actions}, "
                f"but got action_mask of length {mask_arr.size(-1)}"
            )

        with torch.no_grad():
            logits, _ = self.policy(obs_t)
            mask_bool = mask_arr
            valid_count = mask_bool.sum().item()
            if valid_count == 0:
                # No valid actions according to the mask; treat all as valid.
                mask_bool = torch.ones_like(mask_bool, dtype=torch.bool, device=self.device)

            masked_logits = logits.masked_fill(~mask_bool, -1e9)

            probs = F.softmax(masked_logits, dim=-1)

            # Cache masked probabilities for optional logging.
            self.last_action_probs = probs.squeeze(0).detach().cpu().tolist()

            topk = torch.topk(probs, k=min(2, probs.size(-1)), dim=-1)
            self.last_top_actions = topk.indices.squeeze(0).detach().cpu().tolist()
            self.last_top_probs = topk.values.squeeze(0).detach().cpu().tolist()

            if self.deterministic:
                # Greedy by default; optionally sample when top-2 are nearly tied.
                top2 = torch.topk(probs, k=2, dim=-1)
                p1 = float(top2.values[0, 0].item())
                p2 = float(top2.values[0, 1].item())
                if self.tie_break_sample and (p1 - p2) < self.tie_break_eps:
                    dist = torch.distributions.Categorical(probs=probs)
                    action = int(dist.sample().item())
                else:
                    action = int(torch.argmax(probs, dim=-1).item())
            else:
                dist = torch.distributions.Categorical(probs=probs)
                action = int(dist.sample().item())

        return action
