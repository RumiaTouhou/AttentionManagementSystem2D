# Multi-Plate AMS for Task Switching (2D, N>2)

This repository provides a research toolkit for studying **task switching** with an **Attention Management System (AMS)** that uses **reinforcement learning** to decide when the user should switch between tasks.

The design is based on the research paper **“Supporting Task Switching with Reinforcement Learning”** by Alexander Lingler et al. (ACM CHI 2024):  
https://dl.acm.org/doi/10.1145/3613904.3642063

In that paper, the AMS observes the state of each task and then decides when to switch the active task, while the user only controls the currently active task, so that overall task performance can improve compared to manual switching. However, the AMS framework it proposed was only evaluated in tasks where number of subtasks = 2, and the performance is limited if directly extending to tasks where N>2.

---

## Why this reimplementation exists

This repository reimplements the original idea in a new environment, and the reasons are practical and research-related.

1. **Better performance when N is large**  
   In the original Unity version, the frame rate can drop when **N ≥ 7**, which changes the difficulty and adds noise into both training and user studies.  
   In this implementation, the environment is 2D and lightweight, so that it can keep stable timing for larger N.

2. **Reduce unwanted human-factor noise from a 3D environment**  
   A 3D scene can add extra perception and control differences, for example due to camera angle, depth perception, and visual clutter.  
   In this implementation, the game is 2D, so that the visual input is simpler and more consistent.

3. **Easier maintenance and integration for future studies**  
   The original project uses an older Unity editor version (Unity 2021.3.45f1), and it cannot open cleanly in newer Unity versions. 
   In this implementation, the game is written in Python with Pygame, so that it is easier to integrate new models, new logging, and new study logic.

For the adjustment I made to the AMS model, please refer to **"How this AMS design differs from the original paper"** section of this README.

---

## What the game is

Same with the design of the original paper:
- The game contains **N plates** (2 to 12).
- Each plate has one ball, and the ball moves due to initial plate tilt, gravity, drag, and an anti-stall rule.
- At any time, the player can control **only one plate**, while the other plates continue to evolve.
- A round ends when **any** ball crosses the boundary of its plate.
- The drag increases linearly as time grows, making plates more slippery and harder to control.

This setup creates a continuous multitasking problem, where switching too slowly causes neglected plates to fail, while switching too often can reduce control quality.

Furthermore, the game difficulty was slightly decreased (survival time around 175s in single plate game) in order to collect more meaningful user behavior logs in longer gameplay episodes.

A DEMO video exist in project root folder as a demonstration of game mechanism and robustness of AMS I proposed.

---

## Game interface

![Game interface](images/GameInterface.png)

- Player have to keep switching and balancing between plates to ensure the red ball not reaching the edge of the plate.  
- Yellow arrow: plate tilt direction and magnitude
- Plate inner grids: tilt magnitude reference, 15 degrees, 30 degrees, 45 degrees
- Yellow background: the plate user is currently controlling

## AMS prompting sequence

![Prompting](images/Prompting.png)

1. User controlling a plate (in dark yellow).
2. AMS suggest a new plate switch to. The AMS play a short sound. The current controlled plate turn pink and the target plate turn light yellow. The current controlled plate display the layout coordination of the target plate it will switch to. The user can still control the current plate at this moment.
3. The target plate become the new plate user will control. The background of the new plate turn dark yellow, and the original plate it switched from turn black as it is no longer in control. 

---
## Game manipulation 

![Control](images/Control.png)

---

## Repository layout

Main files:

- `PlateGame.py`  
  The game, the physics engine, the AMS interface, the interactive UI, and the headless training environment wrapper.

- `AMSCore.py`  
  `AMSNet` (policy/value network), checkpoint save/load, and `PPOAMS` runtime wrapper.

- `AMSTrain.py`  
  PPO training loop for the AMS, with a curriculum over different N values.

- `CognitiveAgent.py`  
  A belief-based cognitive gameplay agent used during training to model partial observability, reaction time, and action slip.

- `ControlMapping.py`  
  Single-hand prompt controller that applies a pre-cue (0.3s) and then commits the switch.

- `Metrics.py`  
  Logging for episodes and switches, plus optional XAI-style measurements.

- `study_controller.py`  
  A Flask server that runs the user study pipeline and uses webpages in `templates/`.

Webpages:

- `templates/`  
  All webpages for the user study flow (intro, consent, surveys, integrity checks, and related pages).

Output folders (created at runtime):

- `AMSTrained/`  
  Saved AMS checkpoints, for example `AMS3.pt`, `AMS6.pt`, `AMS9.pt`.

- `AMSTrainLogs/<stage>/`  
  Training logs for each stage.

- `PlayLogs/`  
  User study logs (episodes, switches, surveys, layout maps, and optional decision traces).

---

## Installation

You need Python and the following packages:

- pygame
- numpy
- torch
- flask
- matplotlib (optional, only used to save training plots)

Example:

```bash
pip install pygame numpy torch flask matplotlib
```

For `torch`, installation may differ across operating systems and GPU settings, so you can follow the PyTorch install guide if needed.

---

## How to play the game

### Modes

The game has several modes. For interactive play, the most common ones are:

* `PlaygroundNoAMS`: manual play, no AMS, no logging
* `PlaygroundAMS`: play with AMS, no logging
* `HumanBaseline`: baseline condition for user study, manual switching, logging enabled
* `AMSPlay`: AMS-assisted condition for user study, logging enabled

### Start and pause

* Press **BACKSPACE** to start, and press BACKSPACE again to pause or resume.
* On a gamepad, the code also maps start/exit to trigger buttons (see the on-screen instructions).

### Control of the current plate

* Use the joystick to tilt the controlled plate.
* The game reads left and right stick axes and adds them (see `_read_joystick_axes()` in `PlateGame.py`).

### Manual switching (baseline modes)

In `HumanBaseline` and `PlaygroundNoAMS`, switching is manual:

* You can switch with **D-pad** directional navigation.
* You can also press **number keys 1..N** to switch directly.

### AMS switching (AMS modes)

In `AMSPlay` and `PlaygroundAMS`, switching is done by the AMS:

* The AMS selects a new plate.
* The system shows a **pre-cue** for **0.3 seconds**.
* After the pre-cue, the system commits the switch.

This design matches the idea that a switch is not instantaneous, and the user needs a short period to prepare.

---

## How this AMS design differs from the original paper

This implementation follows the same high-level idea, while several design decisions are different due to the essential needs to support N>2.

### 1) 2D physics and engineered observations

The original Unity implementation uses a 3D state representation.
This implementation uses a 2D environment and an engineered observation vector, where each plate is described by features such as:

* distance to boundary (normalized)
* speed and outward speed measures (normalized)
* tilt magnitude and tilt outward component (normalized)
* unattended time (normalized)
* time-to-boundary (TTB) and a TTB-derived urgency feature
* an `ever_controlled` flag

The observation is padded to `N_max = 12`, and unused slots are filled with zeros, while an action mask marks which slots are valid.

### 2) Training reward is not only a timing reward

The original SupervisorAgentV1 uses a timing-based logistic reward.
In this implementation, the training reward includes timing terms and additional shaping terms, such as hazard reduction, neglect penalties, coverage shaping, and N-scaled penalties, so that PPO training remains stable when N is large.

### 3) Safety layers are added to ensure robustness when N>2

When N increases, the probability that at least one plate becomes urgent increases, and small policy errors can end the episode quickly.
Due to this, the system adds safety layers, including:

* **Action masking** for unused plate slots (because `N_max` is fixed).
* **Pre-cue gating** so that the AMS cannot start a new switch while a pre-cue is active.
* **Minimum dwell time** after a switch, which depends on N, so that the system does not switch too rapidly at high N.
* **Emergency triage override** in interactive AMS play, where the system can override the policy choice and switch to the plate with the smallest TTB when TTB is below a threshold.
* **Deterministic tie-break sampling** in inference, where the system samples when the top two action probabilities are very close, so that deterministic argmax does not repeatedly pick one plate in symmetric states.

These layers are described in code in `PlateGame.py` and `AMSCore.py`.

### Existing findings

**Note: below are the findings when conducting user study with a less robust AMS model, not this current implementation. The "mute barrier" problems addressed below no longer exist in our current model.**


- The evaluation identified two distinct performance regimes: an "effective regime" for moderate task counts N ≤ 5 where the system improved performance, and a "failure regime" for high task counts N ≥ 6 where performance collapsed. **(Now resolved. The current AMS model can perform robustly across N = 2 to N = 9)**

- In the effective regime (up to five tasks), the AMS-assisted condition outperformed the self-managed baseline, successfully reducing the rate of performance degradation as the number of tasks increased.


- Within the effective regime, the AMS improved performance by adopting a proactive strategy that increased the pace of switching (shorter dwell times) compared to the manual baseline.


- The AMS tended to switch into target tasks earlier (preventive entry) while they were at lower risk levels, whereas human users in the baseline condition tended to switch later when the target tasks had reached higher risk levels.


- At higher task counts N ≥ 6, the system encountered a critical stability limit termed the "mute barrier," where the supervisor stopped scheduling switches entirely, leading to long periods of inactivity.
 
- This failure mode was pervasive at high task counts; specifically at N = 6, 100% of recorded AMS episodes exhibited "noticeable mute" behavior where the system ceased switching for over 5 seconds.
 
- In the failure regime, episodes ended predominantly due to "starvation/neglect," meaning the system locked onto one controlled task while allowing unattended tasks to drift into failure.
 
- Despite the increased switching frequency in the effective regime, the users' post-switch reaction times did not increase, indicating that the AMS strategy remained compatible with human cognitive limits.

- The study confirmed that training the supervisor against a simulated "Cognitive Gameplay Agent" (which modeled partial observability and motor inertia) produced a policy that was usable by humans, unlike policies trained against unconstrained agents.

### What to do next

- Validate the current revised model with real participants.
- Rerun the analysis with the logs collected with revised model to demonstrate the effectiveness of the new AMS design, and justify AMS could improve user's multitasking performance effectively when N > 2. 

---

## Training the AMS

Training uses PPO and a simple curriculum, which changes the distribution of N across episodes.

Curriculum stages:

* `AMS3`: N in {2, 3}
* `AMS6`: N in {2, 3, 4, 6}
* `AMS9`: N in {2, 3, 4, 6, 8, 9}

Example training command:

```bash
python AMSTrain.py --curricula 3,6,9 --num-envs 30 --total-env-steps-per-stage 3000000
```

Outputs:

* Checkpoints saved to `AMSTrained/AMS3.pt`, `AMSTrained/AMS6.pt`, `AMSTrained/AMS9.pt`
* Training logs saved to `AMSTrainLogs/<stage>/`, including:

  * `config.json`
  * `progress.csv`
  * `metrics_per_N.csv`
  * `curves_main.png` (only if matplotlib is installed)

---

## Logs and data storage (user study modes)

Logging is enabled in `HumanBaseline` and `AMSPlay`, and it is disabled in Playground modes.

By default, logs go to `PlayLogs/`, while you can also set `--log-dir` to write to a different folder.

### Episodes log

* File: `episodes.csv`
* One row per saved episode
* Includes: participant id, condition, N, duration, failure plate, mean metrics (health, tilt, drag), neglect time, and initial condition statistics.

### Switches log

* File: `switches.csv`
* One row per switch (excluding aborted switches)
* Includes:

  * pre-cue start time and commit time
  * dwell time and time since last switch
  * source and target plate state at pre-cue and commit
  * reaction time when available
  * optional XAI fields (HMR, OC, stabilization time, and related flags)

### Survey log

* File: `surveyinfo.csv`
* Written by the user study pipeline in `study_controller.py`
* Includes demographics, gaming background, consent, NASA-TLX entries, and AMS experience questions.

### Layout mapping sidecar

* File: `layout_map_<id>.json`
* Stores the logical-to-internal mapping used for that participant and N.
* This file helps analysis, because the UI layout uses internal indices, while logs often use logical indices.

### Optional decision trace log

* File: `decision_trace.csv`
* Enabled by passing `--decision-trace` in `AMSPlay` runs started by the study pipeline.
* Records each AMS decision, including:

  * policy action and chosen action
  * override flag
  * top probabilities and entropy
  * per-plate TTB and hazard proxy snapshots

---

## How to run and supervise the user study pipeline

### Webpages

All webpages are stored in the `templates/` folder, and `study_controller.py` serves them using Flask.

### Start the pipeline server

```bash
python study_controller.py
```

Then open:

* [http://127.0.0.1:5000](http://127.0.0.1:5000)

### What the pipeline does

The pipeline guides a complete user study flow, including:

* participant id and date
* demographics and gaming background surveys
* consent
* practice sessions
* baseline and AMS blocks (N from 2 to 9)
* NASA-TLX after each block
* AMS experience survey
* integrity checks and retry tools

The block order is counterbalanced by participant id, while odd ids use BaselineFirst and even ids use AMSFirst.

### Episode confirmation with O-gating

In `HumanBaseline` and `AMSPlay`, after Game Over the UI shows a message that asks the supervisor to press **O** to save the episode.

This behavior is intentional, and it supports data quality control:

* If the experiment supervisor thinks the episode should be saved, then the supervisor presses **O**, and the episode is written to CSV.
* If the experiment supervisor thinks the episode should not be saved, then the supervisor does not press O, and the supervisor directly restarts the episode.

Examples of when an episode should not be saved include:

* the participant asks to redo the trial,
* the participant was disturbed by an external event,
* the supervisor observes strong fatigue that affects performance during the episode.

In this way, the saved dataset can better match the intended study protocol.

### Retry and integrity tools

The pipeline includes:

* an integrity page that checks whether each (mode, N) cell has enough episodes,
* a retry tool that deletes one specific cell and relaunches it,
* a force terminate tool that deletes all data for one participant.

---

## Generalization to other tasks and future research direction

This framework is designed for multitask settings where each task has a risk estimate that can be computed or approximated.

In this repository, risk is approximated from features such as distance, speed, outward motion, tilt alignment, and time-to-boundary.

This design should generalize to other tasks where a risk estimate is available, for example when the task state can be measured and a failure boundary can be defined.

For tasks where risks are hard to estimate, robust performance will be more difficult, since the AMS cannot rely on a clear risk signal, and the reward can become sparse and delayed.
Future research can study methods that estimate risk from history, use partial observability models, and apply robust reinforcement learning methods under uncertainty and distribution shift.

