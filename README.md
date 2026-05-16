# Patch Tracker

A personal habit-modification timer that gradually transforms behavior using structured 1-hour intervals, self-ratings, and machine learning recommendations.

## How It Works

Each day is divided into **1-hour intervals**. Within each interval, a "patched" phase (the habit you want to reduce) is followed by an "unpatched" phase (the alternative you want to grow). Over 10 weeks, the ratio shifts progressively toward full "unpatched" time.

| Week | Patched | Unpatched |
|------|---------|-----------|
| 1    | 45 min  | 15 min    |
| 2    | 40 min  | 20 min    |
| 3    | 35 min  | 25 min    |
| 4    | 30 min  | 30 min    |
| 5    | 25 min  | 35 min    |
| 6    | 20 min  | 40 min    |
| 7    | 15 min  | 45 min    |
| 8    | 10 min  | 50 min    |
| 9    | 5 min   | 55 min    |
| 10   | 0 min   | 60 min    |

After each interval, you rate how you felt (**Good / Tired / Bad**). "Bad" ratings drop progress down a week. Consecutive "good" days earn blend offsets that subtly increase unpatched time. An XGBoost model trained on your rating history recommends optimal interval lengths.

## Features

- **Floating timer window** — frameless, always-on-top, semi-transparent dark theme
- **Pause / Resume** — pause mid-interval and pick up where you left off
- **Adaptive progression** — ratings and streaks automatically adjust the schedule
- **ML recommendations** — XGBoost regression model suggests optimal unpatched time based on 8 features (day of week, time of day, recent ratings, streak, etc.)
- **ntfy notifications** — optional push notifications via a local [ntfy](https://ntfy.sh/) server
- **CSV logging** — every interval is logged with date, week, rating, and actual times used
- **Relative load tracking** — shows day-over-day change in unpatched time

## Installation

Requires **Python 3**. Core timer works with only the standard library (Tkinter is built in).

### Optional ML dependencies (recommended):

```bash
pip install xgboost pandas numpy joblib
```

The app gracefully degrades if these are missing — ML recommendation buttons simply won't appear.

## Usage

Run the primary version:

```bash
python dev_dev_main/main_dev_pausebutton.py
```

1. **Setup window** — choose "Continue" to resume the last session or "New Baseline" to start fresh. Select your current week and toggle eye break reminders.
2. **Timer window** — a floating countdown timer appears. It alternates between patched (red label) and unpatched (green label) phases.
3. **Rate each interval** — when the interval ends, rate how you felt.
4. **Use Rec/Safe buttons** — "Rec" applies the ML-recommended unpatched time; "Safe" uses the default schedule.
5. **Skip / Stop** — skip ahead or end the session early. The app handles week transitions and state persistence automatically.

### ntfy Notifications

The ntfy server URL is hardcoded in `dev_dev_main/main_dev_pausebutton.py` line 36. **Edit `NTFY_URL` to point to your own ntfy server before using this feature:**

```python
NTFY_URL = "http://<your-server-ip>:8909/patch_tracker"
```

The app sends POST requests when breaks end and when the patched phase ends.

## Project Structure

```
dev_dev_main/          # Primary active version (XGBoost) — run this one
├── main_dev_pausebutton.py
├── state.json
├── log.csv
├── model.xgb
└── backup_main_old.py
Main_dev/              # PyTorch variant (incomplete)
another main/          # Older PyTorch variant
backup/                # Older XGBoost backup
```

The other directories (`Main_dev/`, `another main/`, `backup/`) contain earlier iterations preserved for reference.

## ML Model

The primary version uses **XGBoost** for regression. The model is trained on historical log data with these features:

- Day of week, normalized hour of day
- Current week in the schedule
- Base unpatched time for the selected week
- Yesterday's average unpatched time
- Blend offset from streak bonuses
- Recent rating proportion (last 5 intervals)
- Days since last "bad" rating

Training happens on startup when new data is available, with a time-based train/validation split and early stopping.

## Data Files

- **`state.json`** — persists current week, streak, and baseline offset
- **`log.csv`** — records every interval with date, week, rating, and actual times
- **`model.xgb`** — trained XGBoost model file (rebuilt on startup when new data exists)
