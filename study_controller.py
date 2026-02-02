#!/usr/bin/env python3
import csv
import datetime
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple
import json

from flask import (
    Flask,
    redirect,
    render_template,
    request,
    url_for,
    send_from_directory,
)

# ---------------------------------------------------------------------------
# Paths and basic configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
PLAYLOGS_DIR = os.path.join(BASE_DIR, "PlayLogs")
ENABLE_DECISION_TRACE = True

EPISODES_CSV = os.path.join(PLAYLOGS_DIR, "episodes.csv")
SWITCHES_CSV = os.path.join(PLAYLOGS_DIR, "switches.csv")
SURVEY_CSV = os.path.join(PLAYLOGS_DIR, "surveyinfo.csv")
STATE_JSON = os.path.join(PLAYLOGS_DIR, "current_state.json")

# Make sure the PlayLogs directory exists
os.makedirs(PLAYLOGS_DIR, exist_ok=True)

app = Flask(__name__, template_folder=TEMPLATES_DIR)


@app.route("/pics/<path:filename>")
def pics(filename: str) -> object:
    """Serve image files from the templates/pics directory."""
    return send_from_directory(os.path.join(TEMPLATES_DIR, "pics"), filename)


# ---------------------------------------------------------------------------
# Study state management (single participant per server session)
# ---------------------------------------------------------------------------


class StudyState:
    """In-memory state for the current participant and block order."""

    def __init__(self) -> None:
        self.participant_id: Optional[int] = None
        # "BaselineFirst" or "AMSFirst"
        self.block_order: Optional[str] = None

        self.study_date: Optional[str] = None  # ISO yyyy-mm-dd
        self.metadata_confirmed: bool = False  # True after first submit

    def reset(self) -> None:
        self.participant_id = None
        self.block_order = None
        self.study_date = None
        self.metadata_confirmed = False



STATE = StudyState()

def _load_state_from_disk() -> None:
    """Best-effort restore of STATE after server restart/reload."""
    if not os.path.exists(STATE_JSON):
        return
    try:
        with open(STATE_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        pid = data.get("participant_id", None)
        block = data.get("block_order", None)
        date_val = data.get("study_date", None)
        confirmed = data.get("metadata_confirmed", False)

        # Restore participant_id
        if STATE.participant_id is None:
            if isinstance(pid, int):
                STATE.participant_id = pid
            elif isinstance(pid, str) and pid.strip().isdigit():
                STATE.participant_id = int(pid.strip())

        # Restore block order (or recompute if pid exists)
        if STATE.block_order is None:
            if block in ("BaselineFirst", "AMSFirst"):
                STATE.block_order = block
            elif STATE.participant_id is not None:
                STATE.block_order = compute_block_order(STATE.participant_id)

        # Restore study_date
        if STATE.study_date is None and isinstance(date_val, str) and date_val.strip():
            STATE.study_date = date_val.strip()

        # Restore confirmed flag
        if not STATE.metadata_confirmed and isinstance(confirmed, bool):
            STATE.metadata_confirmed = confirmed

    except Exception:
        # best-effort: ignore corrupt or partial file
        return


def _save_state_to_disk() -> None:
    """Persist STATE so participant_id/date survive reloads."""
    try:
        payload = {
            "participant_id": STATE.participant_id,
            "block_order": STATE.block_order,
            "study_date": STATE.study_date,
            "metadata_confirmed": STATE.metadata_confirmed,
        }
        with open(STATE_JSON, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        return


def _clear_state_file() -> None:
    try:
        if os.path.exists(STATE_JSON):
            os.remove(STATE_JSON)
    except Exception:
        pass


def compute_block_order(participant_id: int) -> str:
    """
    Compute block order deterministically from participant_id.

    This uses a simple rule for counterbalancing:
    - Odd participant_id: BaselineFirst
    - Even participant_id: AMSFirst
    """
    if participant_id % 2 == 1:
        return "BaselineFirst"
    return "AMSFirst"


def get_today_iso() -> str:
    return datetime.date.today().isoformat()

def compute_next_participant_id() -> int:
    """
    Suggest the next participant id by scanning existing logs and returning max+1.
    If no valid participant ids exist, return 1.

    - Ignores non-numeric values and participant_id <= 0
    - Scans episodes.csv, switches.csv, surveyinfo.csv
    """
    max_pid = 0
    for path in (EPISODES_CSV, SWITCHES_CSV, SURVEY_CSV):
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    raw = (row.get("participant_id", "") or "").strip()
                    try:
                        pid = int(raw)
                    except ValueError:
                        continue
                    if pid > max_pid:
                        max_pid = pid
        except Exception:
            continue

    return (max_pid + 1) if max_pid > 0 else 1


def ensure_metadata(participant_id_str: str, date_str: str) -> Tuple[int, str, str]:
    """
    Confirm participant_id + study_date exactly once (on first POST that includes metadata).
    After confirmation, ignore later edits and always return the confirmed values.

    Returns: (participant_id, block_order, study_date_iso)
    """
    _load_state_from_disk()

    if STATE.metadata_confirmed and STATE.participant_id is not None:
        pid = STATE.participant_id
        block = STATE.block_order or compute_block_order(pid)
        date_val = STATE.study_date or get_today_iso()
        STATE.block_order = block
        STATE.study_date = date_val
        _save_state_to_disk()
        return pid, block, date_val

    pid_input: Optional[int] = None
    s = (participant_id_str or "").strip()
    if s:
        pid_input = int(s)
    else:
        pid_input = compute_next_participant_id()

    chosen_date = (date_str or "").strip() or get_today_iso()

    STATE.participant_id = pid_input
    STATE.block_order = compute_block_order(pid_input)
    STATE.study_date = chosen_date
    STATE.metadata_confirmed = True

    _save_state_to_disk()
    return pid_input, STATE.block_order, chosen_date


def ensure_participant_and_order(participant_id_str: str) -> Tuple[int, str]:
    """
    Backward-compatible wrapper for code paths that only supply participant_id.
    If metadata isn't confirmed yet, this will confirm using today's date.
    """
    pid, block, _ = ensure_metadata(participant_id_str, "")
    return pid, block



# ---------------------------------------------------------------------------
# Survey logging helpers
# ---------------------------------------------------------------------------

# Unified column set for surveyinfo.csv
SURVEY_COLUMNS: List[str] = [
    "participant_id",
    "date",
    "condition",  # block order: BaselineFirst / AMSFirst
    "mode",       # Baseline / AMS / empty
    "survey_type",
    # Demographics
    "age_group",
    "gender",
    "gamepad_experience",
    # Gaming
    "gaming_skill",
    "gaming_hours_per_week",
    # Consent
    "consent",
    # NASA
    "mental_demand",
    "physical_demand",
    "temporal_demand",
    "self_performance",
    "effort",
    "frustration",
    "overall_experience",
    # AMS experience
    "ams_helped_performance",
    "ams_trust",
    "ams_prompt_clarity",
    "ams_prefer_over_manual",
    "ams_comments",
]


def append_survey_row(row: Dict[str, str]) -> None:
    """Append a single survey row to surveyinfo.csv, creating the file with header if needed."""
    file_exists = os.path.exists(SURVEY_CSV)
    with open(SURVEY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SURVEY_COLUMNS)
        if not file_exists:
            writer.writeheader()
        full_row = {col: "" for col in SURVEY_COLUMNS}
        for k, v in row.items():
            if k in full_row:
                full_row[k] = v
        writer.writerow(full_row)


# ---------------------------------------------------------------------------
# Game launching helpers
# ---------------------------------------------------------------------------


def plategame_path() -> str:
    """Return the path to PlateGame.py."""
    return os.path.join(BASE_DIR, "PlateGame.py")


def run_game_subprocess(args: List[str]) -> None:
    """Run PlateGame.py as a blocking subprocess with the given argument list."""
    try:
        subprocess.run(args, check=False)
    except FileNotFoundError:
        # If PlateGame.py or Python is not found, do not raise here,
        # as this controller should still respond; the experimenter
        # will see that the game did not start.
        pass

def launch_game_subprocess(args: List[str]) -> None:
    """Launch PlateGame.py as a non-blocking subprocess (returns immediately)."""
    try:
        subprocess.Popen(args)  # non-blocking
    except FileNotFoundError:
        pass


def run_practice_baseline() -> None:
    """Run a single practice session in PlaygroundNoAMS mode."""
    _load_state_from_disk()
    if STATE.participant_id is None:
        participant_id = 0
        block_order = ""
    else:
        participant_id = STATE.participant_id
        block_order = STATE.block_order or ""
    args = [
        sys.executable,
        plategame_path(),
        "--mode",
        "PlaygroundNoAMS",
        "--num-plates",
        "4",
        "--participant-id",
        str(participant_id),
        "--block-order",
        block_order,
    ]
    run_game_subprocess(args)


def run_practice_ams() -> None:
    """Run a single practice session in PlaygroundAMS mode with AMS9."""
    _load_state_from_disk()
    if STATE.participant_id is None:
        participant_id = 0
        block_order = ""
    else:
        participant_id = STATE.participant_id
        block_order = STATE.block_order or ""
    args = [
        sys.executable,
        plategame_path(),
        "--mode",
        "PlaygroundAMS",
        "--num-plates",
        "4",
        "--participant-id",
        str(participant_id),
        "--block-order",
        block_order,
        "--ams-checkpoint",
        "AMS9",
    ]
    run_game_subprocess(args)


def trial_index_for_n(n_plate: int) -> int:
    """Map number of plates N in [2..9] to trial_index [1..8]."""
    mapping = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8}
    return mapping.get(n_plate, 0)


def run_main_block(mode: str) -> None:
    """
    Run the full block for one mode ("Baseline" or "AMS").

    For each N in 2..9 this starts PlateGame once, and the experimenter
    controls episodes inside the game itself.
    """
    _load_state_from_disk()
    if STATE.participant_id is None:
        participant_id = 0
        block_order = ""
    else:
        participant_id = STATE.participant_id
        block_order = STATE.block_order or ""

    for n in range(2, 10):
        trial_idx = trial_index_for_n(n)
        if mode == "Baseline":
            args = [
                sys.executable,
                plategame_path(),
                "--mode",
                "HumanBaseline",
                "--num-plates",
                str(n),
                "--participant-id",
                str(participant_id),
                "--block-order",
                block_order,
                "--trial-index",
                str(trial_idx),
                "--log-dir",
                PLAYLOGS_DIR,
            ]
        else:  # AMS
            args = [
                sys.executable,
                plategame_path(),
                "--mode",
                "AMSPlay",
                "--num-plates",
                str(n),
                "--participant-id",
                str(participant_id),
                "--block-order",
                block_order,
                "--trial-index",
                str(trial_idx),
                "--ams-checkpoint",
                "AMS9",
                "--log-dir",
                PLAYLOGS_DIR,
            ]

            if ENABLE_DECISION_TRACE:
                args.append("--decision-trace")

        run_game_subprocess(args)


def run_retry_session(participant_id: int, mode: str, n_plates: int) -> None:
    """
    Launch a single retry session for a given mode ("Baseline" or "AMS") and N plates.
    NON-BLOCKING: returns immediately; the game runs as a separate process.
    """
    block_order = STATE.block_order or ""
    trial_idx = trial_index_for_n(n_plates)

    if mode == "Baseline":
        args = [
            sys.executable,
            plategame_path(),
            "--mode",
            "HumanBaseline",
            "--num-plates",
            str(n_plates),
            "--participant-id",
            str(participant_id),
            "--block-order",
            block_order,
            "--trial-index",
            str(trial_idx),
            "--log-dir",
            PLAYLOGS_DIR,
        ]
    else:
        args = [
            sys.executable,
            plategame_path(),
            "--mode",
            "AMSPlay",
            "--num-plates",
            str(n_plates),
            "--participant-id",
            str(participant_id),
            "--block-order",
            block_order,
            "--trial-index",
            str(trial_idx),
            "--ams-checkpoint",
            "AMS9",
            "--log-dir",
            PLAYLOGS_DIR,
        ]

        if ENABLE_DECISION_TRACE:
            args.append("--decision-trace")

    launch_game_subprocess(args)



# ---------------------------------------------------------------------------
# Force terminate and retry helpers
# ---------------------------------------------------------------------------


def delete_participant_from_csv(path: str, participant_id: int) -> None:
    """Remove all rows with the given participant_id from a CSV file, if it exists."""
    if not os.path.exists(path):
        return
    rows: List[Dict[str, str]] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            try:
                pid = int(row.get("participant_id", "0"))
            except ValueError:
                pid = 0
            if pid != participant_id:
                rows.append(row)
    with open(path, "w", newline="", encoding="utf-8") as f:
        if not rows:
            # Write only header if there were fieldnames but no remaining rows
            if fieldnames:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            return
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def delete_layout_sidecars_for_participant(participant_id: int) -> None:
    """Delete layout_map_*.json files that belong to the given participant."""
    for root in (PLAYLOGS_DIR, BASE_DIR):
        if not os.path.isdir(root):
            continue

        for fname in os.listdir(root):
            if not fname.startswith("layout_map_") or not fname.endswith(".json"):
                continue

            full_path = os.path.join(root, fname)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                pid = data.get("participant_id", None)
                if pid == participant_id:
                    os.remove(full_path)
            except Exception:
                # If parsing fails, leave the file in place.
                continue


def force_terminate_participant(participant_id: int) -> None:
    """
    Remove all data for a participant from episodes, switches, and surveys,
    and also delete layout sidecar files.
    """
    delete_participant_from_csv(EPISODES_CSV, participant_id)
    delete_participant_from_csv(SWITCHES_CSV, participant_id)
    delete_participant_from_csv(SURVEY_CSV, participant_id)
    delete_layout_sidecars_for_participant(participant_id)
    _clear_state_file()

    STATE.reset()


def retry_delete_cell(participant_id: int, mode_label: str, n_plates: int) -> None:
    """
    Remove all episode and switch rows for (participant, mode, N_plates).

    mode_label: "Baseline" or "AMS"
    """
    mode_condition = "Baseline" if mode_label == "Baseline" else "AMS-Assisted"

    # Filter episodes
    if os.path.exists(EPISODES_CSV):
        kept_rows: List[Dict[str, str]] = []
        removed_episode_ids: List[str] = []      # legacy fallback
        removed_episode_uids: List[str] = []     # preferred unique key
        with open(EPISODES_CSV, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            for row in reader:
                try:
                    pid = int(row.get("participant_id", "0"))
                except ValueError:
                    pid = 0
                cond = row.get("condition", "")
                try:
                    n_val = int(row.get("N_plates", "0"))
                except ValueError:
                    n_val = 0
                if pid == participant_id and cond == mode_condition and n_val == n_plates:
                    ep_uid = (row.get("episode_uid", "") or "").strip()
                    if ep_uid:
                        removed_episode_uids.append(ep_uid)
                    else:
                        removed_episode_ids.append(row.get("episode_id", ""))
                    continue
                kept_rows.append(row)
        with open(EPISODES_CSV, "w", newline="", encoding="utf-8") as f:
            if kept_rows and fieldnames:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(kept_rows)
            elif fieldnames:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

        # Filter switches for removed episodes (prefer episode_uid; fallback to episode_id for legacy rows)
        if (removed_episode_uids or removed_episode_ids) and os.path.exists(SWITCHES_CSV):
            kept_sw: List[Dict[str, str]] = []
            with open(SWITCHES_CSV, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames_sw = reader.fieldnames or []
                for row in reader:
                    try:
                        pid = int(row.get("participant_id", "0"))
                    except ValueError:
                        pid = 0

                    ep_id = (row.get("episode_id", "") or "").strip()
                    ep_uid = (row.get("episode_uid", "") or "").strip()

                    # Prefer unique episode_uid when available.
                    if pid == participant_id:
                        if removed_episode_uids and ep_uid and ep_uid in removed_episode_uids:
                            continue

                        # Legacy fallback: only apply episode_id matching to rows that have no episode_uid.
                        # This prevents accidental deletion of newer rows where episode_id may repeat.
                        if removed_episode_ids and (not ep_uid) and ep_id in removed_episode_ids:
                            continue

                    kept_sw.append(row)
            with open(SWITCHES_CSV, "w", newline="", encoding="utf-8") as f:
                if kept_sw and fieldnames_sw:
                    writer = csv.DictWriter(f, fieldnames=fieldnames_sw)
                    writer.writeheader()
                    writer.writerows(kept_sw)
                elif fieldnames_sw:
                    writer = csv.DictWriter(f, fieldnames=fieldnames_sw)
                    writer.writeheader()


# ---------------------------------------------------------------------------
# Integrity check helpers
# ---------------------------------------------------------------------------


def load_episodes_for_participant(participant_id: int) -> List[Dict[str, str]]:
    if not os.path.exists(EPISODES_CSV):
        return []
    rows: List[Dict[str, str]] = []
    with open(EPISODES_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                pid = int(row.get("participant_id", "0"))
            except ValueError:
                pid = 0
            if pid == participant_id:
                rows.append(row)
    return rows


def load_survey_for_participant(participant_id: int) -> List[Dict[str, str]]:
    if not os.path.exists(SURVEY_CSV):
        return []
    rows: List[Dict[str, str]] = []
    with open(SURVEY_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                pid = int(row.get("participant_id", "0"))
            except ValueError:
                pid = 0
            if pid == participant_id:
                rows.append(row)
    return rows


def compute_integrity_summary(participant_id: int) -> Dict[str, object]:
    """
    Compute a simple integrity summary for the participant.

    Returns a dict with:
        - baseline_episodes[N] = bool
        - ams_episodes[N] = bool
        - nasa_baseline_done = bool
        - nasa_ams_done = bool
        - ams_experience_done = bool
        - demographics_done = bool
        - gaming_done = bool
        - consent_done = bool
    """
    episodes = load_episodes_for_participant(participant_id)
    surveys = load_survey_for_participant(participant_id)

    baseline_counts: Dict[int, int] = {n: 0 for n in range(2, 10)}
    ams_counts: Dict[int, int] = {n: 0 for n in range(2, 10)}

    for row in episodes:
        cond = row.get("condition", "")
        try:
            n_val = int(row.get("N_plates", "0"))
        except ValueError:
            n_val = 0
        if n_val not in baseline_counts:
            continue
        if cond == "Baseline":
            baseline_counts[n_val] += 1
        elif cond == "AMS-Assisted":
            ams_counts[n_val] += 1

    baseline_ok = {n: (baseline_counts[n] >= 2) for n in baseline_counts}
    ams_ok = {n: (ams_counts[n] >= 2) for n in ams_counts}

    nasa_baseline_done = False
    nasa_ams_done = False
    ams_experience_done = False
    demographics_done = False
    gaming_done = False
    consent_done = False

    for row in surveys:
        stype = row.get("survey_type", "")
        mode = row.get("mode", "")
        if stype == "NASA":
            if mode == "Baseline":
                nasa_baseline_done = True
            elif mode == "AMS":
                nasa_ams_done = True
        elif stype == "AMS_experience":
            ams_experience_done = True
        elif stype == "DEMO":
            demographics_done = True
        elif stype == "GAMING":
            gaming_done = True
        elif stype == "CONSENT":
            consent_done = True

    return {
        "baseline_episodes": baseline_ok,
        "ams_episodes": ams_ok,
        "nasa_baseline_done": nasa_baseline_done,
        "nasa_ams_done": nasa_ams_done,
        "ams_experience_done": ams_experience_done,
        "demographics_done": demographics_done,
        "gaming_done": gaming_done,
        "consent_done": consent_done,
    }


# ---------------------------------------------------------------------------
# Common context creator for templates
# ---------------------------------------------------------------------------


def common_context(mode: str = "") -> Dict[str, object]:
    """
    Base context with participant_id, date, condition (block_order), and mode.
    - Before first submit: suggest participant_id=max+1 (or 1) and date=today, editable.
    - After first submit: show confirmed values, locked.
    """
    _load_state_from_disk()

    if STATE.metadata_confirmed and STATE.participant_id is not None:
        pid = STATE.participant_id
        cond = STATE.block_order or compute_block_order(pid)
        date_val = STATE.study_date or get_today_iso()
        STATE.block_order = cond
        STATE.study_date = date_val
        _save_state_to_disk()
    else:
        pid = compute_next_participant_id()
        cond = ""  # do not set condition until confirmed
        date_val = get_today_iso()

    return {
        "participant_id": pid,
        "date": date_val,
        "condition": cond,
        "mode": mode,
        "metadata_confirmed": STATE.metadata_confirmed,
    }



# ---------------------------------------------------------------------------
# Routes for the 17-step pipeline
# ---------------------------------------------------------------------------


@app.route("/")
def index() -> object:
    return redirect(url_for("intro"))


@app.route("/intro", methods=["GET", "POST"])
def intro() -> object:
    if request.method == "POST":
        return redirect(url_for("demographics"))
    ctx = common_context()
    return render_template("intro.html", **ctx)


@app.route("/demographics", methods=["GET", "POST"])
def demographics() -> object:
    if request.method == "POST":
        pid_str = request.form.get("participant_id", "")
        date_input = request.form.get("date", "")

        age_group = request.form.get("age_group", "").strip()
        gender = request.form.get("gender", "").strip()
        gamepad_exp = request.form.get("gamepad_experience", "").strip()

        participant_id, block_order, study_date = ensure_metadata(pid_str, date_input)

        row = {
            "participant_id": str(participant_id),
            "date": study_date,
            "condition": block_order,
            "mode": "",
            "survey_type": "DEMO",
            "age_group": age_group,
            "gender": gender,
            "gamepad_experience": gamepad_exp,
        }
        append_survey_row(row)

        return redirect(url_for("gaming_skill"))


    ctx = common_context()
    return render_template("demographics.html", **ctx)


@app.route("/gaming_skill", methods=["GET", "POST"])
def gaming_skill() -> object:
    if request.method == "POST":
        pid_str = request.form.get("participant_id", "")
        date_input = request.form.get("date", "")

        gaming_skill_val = request.form.get("gaming_skill", "").strip()
        gaming_hours = request.form.get("gaming_hours_per_week", "").strip()

        participant_id, block_order, study_date = ensure_metadata(pid_str, date_input)

        row = {
            "participant_id": str(participant_id),
            "date": study_date,
            "condition": block_order,
            "mode": "",
            "survey_type": "GAMING",
            "gaming_skill": gaming_skill_val,
            "gaming_hours_per_week": gaming_hours,
        }
        append_survey_row(row)

        return redirect(url_for("consent_page"))

    ctx = common_context()
    return render_template("gaming_skill.html", **ctx)


@app.route("/consent", methods=["GET", "POST"])
def consent_page() -> object:
    if request.method == "POST":
        pid_str = request.form.get("participant_id", "")
        date_input = request.form.get("date", "")

        consent_val = request.form.get("consent", "").strip()

        participant_id, block_order, study_date = ensure_metadata(pid_str, date_input)

        row = {
            "participant_id": str(participant_id),
            "date": study_date,
            "condition": block_order,
            "mode": "",
            "survey_type": "CONSENT",
            "consent": consent_val,
        }
        append_survey_row(row)


        # After consent, go to first block according to block order.
        block_order = STATE.block_order or "BaselineFirst"
        if block_order == "BaselineFirst":
            return redirect(url_for("baseline_intro"))
        else:
            return redirect(url_for("ams_intro"))

    ctx = common_context()
    return render_template("consent.html", **ctx)


@app.route("/baseline_intro", methods=["GET", "POST"])
def baseline_intro() -> object:
    if request.method == "POST":
        return redirect(url_for("practice_baseline"))
    ctx = common_context(mode="Baseline")
    return render_template("baseline_intro.html", **ctx)


@app.route("/ams_intro", methods=["GET", "POST"])
def ams_intro() -> object:
    if request.method == "POST":
        return redirect(url_for("practice_ams"))
    ctx = common_context(mode="AMS")
    return render_template("ams_intro.html", **ctx)


@app.route("/practice_baseline", methods=["GET", "POST"])
def practice_baseline() -> object:
    if request.method == "POST":
        action = request.form.get("action", "")
        if action == "start_practice":
            run_practice_baseline()
            # Return to the same page so that the experimenter can start again or continue.
            return redirect(url_for("practice_baseline"))
        elif action == "continue":
            return redirect(url_for("ready_baseline"))
    ctx = common_context(mode="Baseline")
    return render_template("practice_baseline.html", **ctx)


@app.route("/practice_ams", methods=["GET", "POST"])
def practice_ams() -> object:
    if request.method == "POST":
        action = request.form.get("action", "")
        if action == "start_practice":
            run_practice_ams()
            return redirect(url_for("practice_ams"))
        elif action == "continue":
            return redirect(url_for("ready_ams"))
    ctx = common_context(mode="AMS")
    return render_template("practice_ams.html", **ctx)


@app.route("/ready_baseline", methods=["GET", "POST"])
def ready_baseline() -> object:
    if request.method == "POST":
        pid_str = request.form.get("participant_id", "").strip()
        if pid_str:
            ensure_participant_and_order(pid_str)
        else:
            _load_state_from_disk()
        run_main_block("Baseline")
        return redirect(url_for("nasa_baseline"))
    ctx = common_context(mode="Baseline")
    return render_template("ready_baseline.html", **ctx)


@app.route("/ready_ams", methods=["GET", "POST"])
def ready_ams() -> object:
    if request.method == "POST":
        pid_str = request.form.get("participant_id", "").strip()
        if pid_str:
            ensure_participant_and_order(pid_str)
        else:
            _load_state_from_disk()
        run_main_block("AMS")
        return redirect(url_for("nasa_ams"))
    ctx = common_context(mode="AMS")
    return render_template("ready_ams.html", **ctx)


@app.route("/nasa_baseline", methods=["GET", "POST"])
def nasa_baseline() -> object:
    if request.method == "POST":
        pid_str = request.form.get("participant_id", "")
        date_input = request.form.get("date", "")

        mental = request.form.get("mental_demand", "").strip()
        physical = request.form.get("physical_demand", "").strip()
        temporal = request.form.get("temporal_demand", "").strip()
        performance = request.form.get("self_performance", "").strip()
        effort = request.form.get("effort", "").strip()
        frustration = request.form.get("frustration", "").strip()
        overall = request.form.get("overall_experience", "").strip()

        participant_id, block_order, study_date = ensure_metadata(pid_str, date_input)

        row = {
            "participant_id": str(participant_id),
            "date": study_date,
            "condition": block_order,
            "mode": "Baseline",
            "survey_type": "NASA",
            "mental_demand": mental,
            "physical_demand": physical,
            "temporal_demand": temporal,
            "self_performance": performance,
            "effort": effort,
            "frustration": frustration,
            "overall_experience": overall,
        }
        append_survey_row(row)

        # After NASA baseline, go either to AMS intro (if BaselineFirst)
        # or directly to AMS experience (if AMSFirst and AMS block is already done).
        block_order = STATE.block_order or "BaselineFirst"
        if block_order == "BaselineFirst":
            return redirect(url_for("ams_intro"))
        else:
            return redirect(url_for("ams_experience"))

    ctx = common_context(mode="Baseline")
    return render_template("nasa_baseline.html", **ctx)


@app.route("/nasa_ams", methods=["GET", "POST"])
def nasa_ams() -> object:
    if request.method == "POST":
        pid_str = request.form.get("participant_id", "")
        date_input = request.form.get("date", "")

        mental = request.form.get("mental_demand", "").strip()
        physical = request.form.get("physical_demand", "").strip()
        temporal = request.form.get("temporal_demand", "").strip()
        performance = request.form.get("self_performance", "").strip()
        effort = request.form.get("effort", "").strip()
        frustration = request.form.get("frustration", "").strip()
        overall = request.form.get("overall_experience", "").strip()

        participant_id, block_order, study_date = ensure_metadata(pid_str, date_input)

        row = {
            "participant_id": str(participant_id),
            "date": study_date,
            "condition": block_order,
            "mode": "AMS",
            "survey_type": "NASA",
            "mental_demand": mental,
            "physical_demand": physical,
            "temporal_demand": temporal,
            "self_performance": performance,
            "effort": effort,
            "frustration": frustration,
            "overall_experience": overall,
        }
        append_survey_row(row)

        # After NASA AMS, go either to Baseline intro (if AMSFirst)
        # or to AMS experience (if BaselineFirst and baseline block is already done).
        block_order = STATE.block_order or "BaselineFirst"
        if block_order == "BaselineFirst":
            return redirect(url_for("ams_experience"))
        else:
            return redirect(url_for("baseline_intro"))

    ctx = common_context(mode="AMS")
    return render_template("nasa_ams.html", **ctx)


@app.route("/ams_experience", methods=["GET", "POST"])
def ams_experience() -> object:
    if request.method == "POST":
        pid_str = request.form.get("participant_id", "")
        date_input = request.form.get("date", "")

        helped = request.form.get("ams_helped_performance", "").strip()
        trust = request.form.get("ams_trust", "").strip()
        clarity = request.form.get("ams_prompt_clarity", "").strip()
        prefer = request.form.get("ams_prefer_over_manual", "").strip()
        comments = request.form.get("ams_comments", "").strip()

        participant_id, block_order, study_date = ensure_metadata(pid_str, date_input)

        row = {
            "participant_id": str(participant_id),
            "date": study_date,
            "condition": block_order,
            "mode": "AMS",
            "survey_type": "AMS_experience",
            "ams_helped_performance": helped,
            "ams_trust": trust,
            "ams_prompt_clarity": clarity,
            "ams_prefer_over_manual": prefer,
            "ams_comments": comments,
        }
        append_survey_row(row)

        return redirect(url_for("integrity"))

    ctx = common_context(mode="AMS")
    return render_template("ams_experience.html", **ctx)


@app.route("/integrity", methods=["GET", "POST"])
def integrity() -> object:
    if STATE.participant_id is None:
        # If no participant_id is known yet, redirect to intro.
        return redirect(url_for("intro"))

    participant_id = STATE.participant_id

    if request.method == "POST":
        # Proceed to closing page
        return redirect(url_for("closing"))

    summary = compute_integrity_summary(participant_id)
    ctx = common_context()
    ctx.update(summary)
    return render_template("integrity.html", **ctx)


@app.route("/closing", methods=["GET"])
def closing() -> object:
    ctx = common_context()
    return render_template("closing.html", **ctx)


# ---------------------------------------------------------------------------
# Force terminate and retry endpoints
# ---------------------------------------------------------------------------


@app.route("/force_terminate", methods=["POST"])
def force_terminate_route() -> object:
    pid_str = request.form.get("terminate_participant_id", "").strip()
    if not pid_str and STATE.participant_id is not None:
        pid_str = str(STATE.participant_id)
    try:
        participant_id = int(pid_str)
    except (TypeError, ValueError):
        # If parsing fails, we simply do nothing and return to intro.
        return redirect(url_for("intro"))

    force_terminate_participant(participant_id)
    return redirect(url_for("intro"))


@app.route("/retry", methods=["POST"])
def retry_route() -> object:
    """
    Retry a specific cell (participant, mode, N).

    Expected form fields:
      - retry_participant_id
      - retry_condition  (Y/N or AMS/Baseline-like)
      - retry_n          (number of plates)
    """
    pid_str = request.form.get("retry_participant_id", "").strip()
    cond_str = request.form.get("retry_condition", "").strip()
    n_str = request.form.get("retry_n", "").strip()

    try:
        participant_id = int(pid_str)
        n_val = int(n_str)
    except (TypeError, ValueError):
        # If input is invalid, simply go back to integrity or intro.
        if STATE.participant_id is not None:
            return redirect(url_for("integrity"))
        return redirect(url_for("intro"))

    # Map retry_condition to mode label.
    # Interpret "Y" as AMS, "N" as Baseline, and also accept strings.
    cond_upper = cond_str.upper()
    if cond_upper in ("Y", "AMS", "WITHAMS", "A"):
        mode_label = "AMS"
    else:
        mode_label = "Baseline"

    # Delete existing data for this cell and run a new session.
    retry_delete_cell(participant_id, mode_label, n_val)
    STATE.participant_id = participant_id
    if STATE.block_order is None:
        STATE.block_order = compute_block_order(participant_id)

    _save_state_to_disk()

    run_retry_session(participant_id, mode_label, n_val)

    return ("", 204)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _clear_state_file()
    STATE.reset()
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)

