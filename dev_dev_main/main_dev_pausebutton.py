#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patch Timer — Compact Dark (Transparent, No Shortcuts)
- Small, always-on-top, non-resizable
- Semi-transparent window (alpha ~0.88)
- Dark palette tuned for transparency
- Timer on top, concise info, short button labels
- No keyboard shortcuts
- Optional ML features (auto-disabled if deps missing)
- Safer timing logic with clamped durations and defensive after-cancel
- ntfy server notificaitons for when break is over and when patched is over
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
import os, json, csv, datetime
from typing import Optional, Callable, List, Tuple

# === Files ===
STATE_FILE       = "state.json"
LOG_FILE         = "log.csv"
MODEL_FILE       = "model.xgb"

# === Program knobs ===
FULL_DAY_MIN        = 6
MIN_BLEND_DAYS      = 4
BAD_DELAY_DAYS      = 3
OFFSET_DELAY_DAYS   = 2
REL_LOAD_THRESHOLD  = 0.05  # 5% relative jump gate
SECONDS_PER_BLOCK   = 3600  # 1-hour intervals
WINDOW_ALPHA        = 0.88  # overall transparency (0..1)
# === Notifications (ntfy) ===
NTFY_URL = "http://192.168.0.224:8909/patch_tracker"

def notify_ntfy(msg: str):
    """Fire-and-forget POST to a local ntfy topic."""
    try:
        import urllib.request
        req = urllib.request.Request(NTFY_URL, data=msg.encode("utf-8"), method="POST")
        # Optional niceties for ntfy apps:
        req.add_header("Title", msg)
        urllib.request.urlopen(req, timeout=2)
    except Exception:
        # Stay silent on failures; never break the timer
        pass


DEFAULT_STATE = {
    "current_week": 1,
    "good_day_streak": 0,
    "last_day_blended": False,
    "baseline_offset": 0
}

WEEK_CONFIG = {
    1:  {"patched":45, "unpatched":15},
    2:  {"patched":40, "unpatched":20},
    3:  {"patched":35, "unpatched":25},
    4:  {"patched":30, "unpatched":30},
    5:  {"patched":25, "unpatched":35},
    6:  {"patched":20, "unpatched":40},
    7:  {"patched":15, "unpatched":45},
    8:  {"patched":10, "unpatched":50},
    9:  {"patched":5,  "unpatched":55},
    10: {"patched":0,  "unpatched":60},
}

# --- Colors (Dark, tuned for alpha) ---
C_BG          = "#0b0f14"
C_ELEVATED    = "#10151d"
C_FG          = "#e6edf3"
C_MUTED       = "#9aa7b4"
C_BORDER      = "#1a2330"
C_ACCENT      = "#23c2d1"
C_GOOD        = "#1ed28a"
C_WARN        = "#f5b342"

PHASE_COLORS = {
    "patched":   "#2b1d1d",
    "unpatched": "#1b2a23",
    "break":     "#1b2230",
}

# === Utility ===
def mmss(sec: int) -> str:
    m, s = divmod(max(0, int(sec)), 60)
    return f"{m:02d}:{s:02d}"

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def load_state():
    if not os.path.exists(STATE_FILE):
        return DEFAULT_STATE.copy()
    try:
        with open(STATE_FILE) as f:
            s = json.load(f)
    except Exception:
        return DEFAULT_STATE.copy()
    for k, v in DEFAULT_STATE.items():
        s.setdefault(k, v)
    return s

def save_state(s):
    with open(STATE_FILE, 'w') as f:
        json.dump(s, f, indent=2)

def append_log(date, bw, idx, wu, rating, used_unpatched, used_patched):
    new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date","baseline_week","interval_idx","week_used","rating","used_unpatched","used_patched"])
        w.writerow([date, bw, idx, wu, rating, used_unpatched, used_patched])

def get_log_entries(date_str):
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, newline='') as f:
        return [r for r in csv.DictReader(f) if r['date'] == date_str]

def _all_dates():
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, newline='') as f:
        return sorted({r['date'] for r in csv.DictReader(f)})

def time_since_last_bad(today):
    dates = _all_dates()
    if not dates:
        return 999
    earliest = datetime.date.fromisoformat(dates[0])
    d = datetime.date.fromisoformat(today)
    days = 0
    while d > earliest:
        d -= datetime.timedelta(days=1)
        days += 1
        for e in get_log_entries(d.isoformat()):
            if e['rating'] == 'bad':
                return days
    return days

def compute_streak(today):
    dates = _all_dates()
    if not dates:
        return 0
    earliest = datetime.date.fromisoformat(dates[0])
    cnt = 0
    d = datetime.date.fromisoformat(today) - datetime.timedelta(days=1)
    while d >= earliest:
        es = get_log_entries(d.isoformat())
        if not es:
            d -= datetime.timedelta(days=1)
            continue
        if any(e['rating']=='bad' for e in es) or len(es) < FULL_DAY_MIN:
            break
        cnt += 1
        if cnt >= MIN_BLEND_DAYS:
            break
        d -= datetime.timedelta(days=1)
    return cnt

# === Optional ML (safe if deps missing) ===
def _ml_available():
    try:
        import pandas  # noqa: F401
        import numpy   # noqa: F401
        import joblib  # noqa: F401
        import xgboost # noqa: F401
        return True
    except Exception:
        return False

def train_model_safe():
    if not _ml_available():
        return None
    import pandas as pd
    import numpy as np
    import joblib
    from xgboost import XGBRegressor

    if not os.path.exists(LOG_FILE):
        return None
    df = pd.read_csv(LOG_FILE, parse_dates=['date'])
    if len(df) < 30:
        return None

    df.sort_values(['interval_idx','date'], inplace=True)
    df['prev_unpatched'] = df.groupby('interval_idx')['used_unpatched'].shift(1)
    df['rel_load'] = (df['used_unpatched'] - df['prev_unpatched']) / df['prev_unpatched']
    df['rel_load'] = df['rel_load'].fillna(0)

    df['date_ord'] = df['date'].dt.date.map(lambda d: d.toordinal())
    df['good_day_streak'] = df['date'].dt.date.map(lambda d: compute_streak(d.isoformat()))
    df['days_since_bad']  = df['date'].dt.date.map(lambda d: time_since_last_bad(d.isoformat()))
    df['used_patched'] = df['used_patched'].fillna(60 - df['used_unpatched'])

    features = [
        'baseline_week','interval_idx','used_unpatched','used_patched',
        'date_ord','good_day_streak','days_since_bad','rel_load'
    ]
    X = df[features].values
    y = df['used_unpatched'].values

    unique_dates = df['date'].dt.date.drop_duplicates().sort_values()
    if len(unique_dates) > 10:
        cutoff = unique_dates.iloc[-10]
        train_mask = df['date'].dt.date < cutoff
        val_mask   = df['date'].dt.date >= cutoff
    else:
        idx = np.arange(len(df)); rng = np.random.default_rng(7); rng.shuffle(idx)
        split = int(len(df)*0.8)
        train_mask = df.index.isin(idx[:split])
        val_mask   = df.index.isin(idx[split:])

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]

    model = XGBRegressor(
        n_estimators=160,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.2
    )
    try:
        model.fit(X_train, y_train, eval_set=[(X_val,y_val)], early_stopping_rounds=12, verbose=False)
    except TypeError:
        model.fit(X_train, y_train, eval_set=[(X_val,y_val)], verbose=False)

    try:
        joblib.dump(model, MODEL_FILE)
    except Exception:
        pass
    return model

def load_model_safe():
    if not _ml_available():
        return None
    import joblib
    try:
        if os.path.exists(MODEL_FILE):
            return joblib.load(MODEL_FILE)
    except Exception:
        pass
    return train_model_safe()

# === Theme ===
def apply_dark_theme(root: tk.Tk) -> ttk.Style:
    root.configure(bg=C_BG)
    try:
        root.tk.call('tk', 'scaling', 1.0)  # compact
    except Exception:
        pass

    style = ttk.Style(root)
    try: style.theme_use('clam')
    except Exception: pass

    style.configure('.', background=C_BG, foreground=C_FG, fieldbackground=C_ELEVATED, bordercolor=C_BORDER)
    style.configure('TFrame', background=C_BG)
    style.configure('Card.TFrame', background=C_ELEVATED, borderwidth=1, relief='solid')
    style.configure('TLabel', background=C_BG, foreground=C_FG)
    style.configure('Muted.TLabel', background=C_BG, foreground=C_MUTED)
    style.configure('Phase.TFrame', background=PHASE_COLORS['patched'])
    style.configure('TButton', background=C_ELEVATED, foreground=C_FG, borderwidth=1)
    style.map('TButton', background=[('active', '#1a2330')], relief=[('pressed','sunken')])
    style.configure('TRadiobutton', background=C_BG, foreground=C_FG)
    style.configure('TCheckbutton', background=C_BG, foreground=C_FG)
    style.configure('TLabelframe', background=C_BG, foreground=C_MUTED, bordercolor=C_BORDER)
    style.configure('TLabelframe.Label', background=C_BG, foreground=C_MUTED)
    return style

# === Custom modals ===
class Modal:
    def __init__(self, parent: tk.Tk, title: str, message: str, buttons: List[Tuple[str, Optional[Callable]]], width: int = 320):
        self.parent = parent
        self.res = None
        self.top = tk.Toplevel(parent)
        self.top.transient(parent)
        self.top.grab_set()
        self.top.title(title)
        self.top.configure(bg=C_ELEVATED)
        self.top.resizable(False, False)

        # Center on parent
        self.top.update_idletasks()
        px = parent.winfo_x() + (parent.winfo_width() - width) // 2
        py = parent.winfo_y() + (parent.winfo_height() - 150) // 2
        self.top.geometry(f"{width}x150+{max(0,px)}+{max(0,py)}")

        frm = ttk.Frame(self.top, style='Card.TFrame', padding=12)
        frm.pack(expand=True, fill='both')
        ttk.Label(frm, text=title, font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(0,6))
        ttk.Label(frm, text=message, style='Muted.TLabel', wraplength=width-36, justify='left').pack(anchor='w', pady=(0,8))

        btns = ttk.Frame(frm)
        btns.pack(anchor='e')
        for i, (txt, cb) in enumerate(buttons):
            def _mk(cmd: Optional[Callable]):
                return lambda: (cmd() if cmd else None, self._close(txt))
            b = ttk.Button(btns, text=txt, command=_mk(cb))
            b.grid(row=0, column=i, padx=(0,6) if i < len(buttons)-1 else 0)

        self.top.bind('<Escape>', lambda e: self._close(None))

    def _close(self, result):
        self.res = result
        self.top.grab_release()
        self.top.destroy()

def ask_ok(parent: tk.Tk, title: str, message: str):
    m = Modal(parent, title, message, [('OK', None)])
    parent.wait_window(m.top)
    return True

def ask_choice(parent: tk.Tk, title: str, message: str, choices: List[Tuple[str,str]]):
    clicked = {'v': None}
    def setv(v): clicked['v'] = v
    m = Modal(parent, title, message, [(txt, lambda v=val: setv(v)) for (txt,val) in choices])
    parent.wait_window(m.top)
    return clicked['v']

# === Session bootstrap ===
def start_session(eye_breaks: bool, today: str):
    state = load_state()
    state['good_day_streak']  = compute_streak(today)
    state['last_day_blended'] = (
        state['good_day_streak'] >= MIN_BLEND_DAYS and
        time_since_last_bad(today) >= BAD_DELAY_DAYS
    )
    save_state(state)

    model = load_model_safe()  # may be None
    rec: Optional[float] = None
    base_unp = WEEK_CONFIG[state['current_week']]['unpatched']
    next_unp = WEEK_CONFIG.get(state['current_week']+1, WEEK_CONFIG[max(WEEK_CONFIG)])['unpatched']

    if model is not None:
        # Estimate relative load from yesterday's first entry (if any)
        yest = (datetime.date.fromisoformat(today) - datetime.timedelta(days=1)).isoformat()
        ents = get_log_entries(yest)
        rel = 0.0
        if ents:
            try:
                prev_u = float(ents[0]['used_unpatched'])
                rel = (base_unp - prev_u) / prev_u if prev_u else 0.0
            except Exception:
                rel = 0.0
        try:
            import numpy as np
            feat = np.array([[
                state['current_week'], 1,
                base_unp,
                WEEK_CONFIG[state['current_week']]['patched'],
                datetime.date.fromisoformat(today).toordinal(),
                state['good_day_streak'],
                time_since_last_bad(today),
                rel
            ]], dtype=float)
            p = float(model.predict(feat)[0])
            if rel > REL_LOAD_THRESHOLD:
                rec = float(base_unp)
            else:
                rec = float(max(base_unp, min(p, next_unp)))
        except Exception:
            rec = float(base_unp)

    TimerApp(state, today, eye_breaks, model, rec)

# === Main (setup window) ===
class MainApp:
    def __init__(self):
        state = load_state()
        today = datetime.date.today().isoformat()

        root = tk.Tk()
        root.title("Patch Timer — Setup")
        apply_dark_theme(root)
        root.attributes("-topmost", True)
        root.attributes("-alpha", WINDOW_ALPHA)
        root.resizable(False, False)

        container = ttk.Frame(root, style='Card.TFrame', padding=10)
        container.grid(sticky="nsew", padx=8, pady=8)
        root.columnconfigure(0, weight=1); root.rowconfigure(0, weight=1)

        has = os.path.exists(STATE_FILE)
        self.choice = tk.StringVar(value="continue" if has else "new")
        r1 = ttk.Radiobutton(container, text="Continue", variable=self.choice, value="continue", command=self._update)
        r2 = ttk.Radiobutton(container, text="New Baseline", variable=self.choice, value="new", command=self._update)
        r1.grid(sticky="w"); r2.grid(sticky="w", pady=(2,6))

        info = (f"W{state['current_week']}  "
                f"S{state['good_day_streak']}  "
                f"Off{state['baseline_offset']}s  "
                f"Bad{time_since_last_bad(today)}d")
        self.info_lbl = ttk.Label(container, text=info, style='Muted.TLabel')
        if has and self.choice.get()=="continue":
            self.info_lbl.grid(sticky="w", pady=(2,8))

        self.week_var = tk.IntVar(value=state['current_week'])
        wf = ttk.Labelframe(container, text="Start week", padding=6)
        for w in range(1, max(WEEK_CONFIG)+1):
            cfg = WEEK_CONFIG[w]
            txt = f"W{w}: {cfg['unpatched']} / {cfg['patched']}"
            ttk.Radiobutton(wf, text=txt, variable=self.week_var, value=w).pack(anchor="w")
        if self.choice.get()=="new":
            wf.grid(sticky="w", pady=(2,8))
        self.week_frame = wf

        self.eye_var = tk.BooleanVar()
        ttk.Checkbutton(container, text="Eye breaks (display only)", variable=self.eye_var).grid(sticky="w", pady=(2,8))

        go = ttk.Button(container, text="Start", command=lambda:self._go(root,today))
        go.grid(sticky="e")
        root.mainloop()

    def _update(self):
        if self.choice.get()=="new":
            self.week_frame.grid(sticky="w", pady=(2,8))
            self.info_lbl.grid_forget()
        else:
            self.week_frame.grid_forget()
            self.info_lbl.grid(sticky="w", pady=(2,8))

    def _go(self, root, today):
        state = load_state()
        if self.choice.get()=="new":
            state = DEFAULT_STATE.copy()
            state['current_week'] = self.week_var.get()
        save_state(state)
        root.destroy()
        start_session(self.eye_var.get(), today)

# === Timer window ===
class TimerApp:
    def __init__(self, state, today, eye_breaks, model, rec: Optional[float]):
        self.state, self.today, self.model, self.rec = state, today, model, rec
        self.idx, self.week_used = 1, state['current_week']
        self.use_rec = self.safe_use = False
        self.after_id = None
        self.paused=False
        self.remaining=0
        self.current_phase = 'patched'
        self.cur_unp = self.cur_pat = 0

        self.app = tk.Tk()
        self.app.title("Patch Timer")
        apply_dark_theme(self.app)
        self.app.attributes("-topmost", True)
        self.app.attributes("-alpha", WINDOW_ALPHA)
        self.app.resizable(False, False)
        self.app.overrideredirect(True)  # frameless window

        # --- Layout: compact vertical stack ---
        wrap = ttk.Frame(self.app, style='Card.TFrame', padding=8)
        wrap.pack(fill="both", expand=True, padx=8, pady=8)

        # Phase strip
        self.phase_strip = ttk.Frame(wrap, style='Phase.TFrame', height=4)
        self.phase_strip.pack(fill='x', side='top')

        # Timer (draggable)
        self.lbl = ttk.Label(wrap, font=("Segoe UI", 24, "bold"))
        self.lbl.pack(fill="x", pady=(4,2))
        self.lbl.configure(anchor='center')
        self.lbl.bind("<ButtonPress-1>", self._start_move)
        self.lbl.bind("<B1-Motion>", self._do_move)

        # Info line 1: rec + interval count
        info1 = ttk.Frame(wrap); info1.pack(fill='x', pady=(0,2))
        self.rec_lbl = ttk.Label(info1, text=f"Rec {self.rec:.1f}m" if isinstance(self.rec,(int,float)) else "Rec --", style='Muted.TLabel')
        self.rec_lbl.pack(side='left')
        self.int_lbl = ttk.Label(info1, text="Int 0", style='Muted.TLabel')
        self.int_lbl.pack(side='right')

        # Info line 2: current week and split
        info2 = ttk.Frame(wrap); info2.pack(fill='x', pady=(0,4))
        cfg = WEEK_CONFIG[self.state['current_week']]
        self.split_lbl = ttk.Label(info2, text=f"W{self.state['current_week']} • U{cfg['unpatched']} / P{cfg['patched']}", style='Muted.TLabel')
        self.split_lbl.pack(side='left')
        # Optional load delta shown later if available
        self.load_lbl = ttk.Label(info2, text="", style='Muted.TLabel')
        self.load_lbl.pack(side='right')

        # Buttons: short labels in two rows
        btn_row1 = ttk.Frame(wrap); btn_row1.pack(fill='x', pady=(2,2))
        btn_row2 = ttk.Frame(wrap); btn_row2.pack(fill='x')

        self.btn_use_rec = ttk.Button(btn_row1, text="Rec", command=self._apply_rec, width=6)
        self.btn_safe    = ttk.Button(btn_row1, text="Safe", command=self._apply_safe, width=6)
        self.btn_retrain = ttk.Button(btn_row1, text="Retrain", command=self._manual_retrain, width=8)
        self.btn_pause   = ttk.Button(btn_row2, text="Pause", command=self._toggle_pause, width=8)
        self.btn_skip    = ttk.Button(btn_row2, text="Skip", command=self._skip_to_unpatched, width=6)
        self.btn_stop    = ttk.Button(btn_row2, text="Stop", command=self._stop, width=6)

        for b in (self.btn_use_rec, self.btn_safe, self.btn_retrain):
            b.pack(side='left', padx=4)
        for b in (self.btn_pause, self.btn_skip, self.btn_stop):
            b.pack(side='left', padx=4)

        # Disable ML-related controls if model support unavailable
        ml_ok = _ml_available()
        if not ml_ok:
            for b in (self.btn_use_rec, self.btn_safe, self.btn_retrain):
                b.state(["disabled"])

        # Relative-load indicator (best effort)
        yest = (datetime.date.fromisoformat(today)-datetime.timedelta(days=1)).isoformat()
        ents = get_log_entries(yest)
        if ents:
            try:
                last_unp = float(ents[-1]['used_unpatched'])
                today_unp = self.rec if isinstance(self.rec,(int,float)) else WEEK_CONFIG[self.state['current_week']]['unpatched']
                diff = today_unp - last_unp
                pct = (diff/last_unp*100) if last_unp else 0.0
                self.load_lbl.config(text=f"Δ {diff:+.0f}m {pct:+.1f}%")
            except Exception:
                pass

        self._position()
        self._start_interval()
        self.app.mainloop()

    # ----- Window utilities -----
    def _start_move(self, e):
        self.sx, self.sy = e.x_root, e.y_root
        self.ox, self.oy = self.app.winfo_x(), self.app.winfo_y()

    def _do_move(self, e):
        dx, dy = e.x_root - self.sx, e.y_root - self.sy
        self.app.geometry(f"+{self.ox+dx}+{self.oy+dy}")

    def _position(self):
        self.app.update_idletasks()
        w = self.app.winfo_width()
        h = self.app.winfo_height()
        sh = self.app.winfo_screenheight()
        x = 16
        y = sh - h - 40
        self.app.geometry(f"+{x}+{y}")

    # ----- Control handlers -----
    def _toggle_pause(self):
        if not self.paused:
            if self.after_id:
                try: self.app.after_cancel(self.after_id)
                except Exception: pass
            self.paused = True
            self.btn_pause.configure(text="Resume")
            self._toast("Paused")
        else:
            self.paused = False
            self.btn_pause.configure(text="Pause")
            self._tick(self.current_phase, self.remaining)

    def _apply_rec(self):
        if not isinstance(self.rec,(int,float)):
            self._toast("No recommendation")
            return
        self.use_rec, self.safe_use = True, False
        if self.current_phase=='patched':
            if self.after_id:
                try: self.app.after_cancel(self.after_id)
                except Exception: pass
            self._recalc(); self._style('patched'); self._tick('patched',self.cur_pat)
        self._toast(f"Rec {self.rec:.1f}m")

    def _apply_safe(self):
        if not isinstance(self.rec,(int,float)):
            self._toast("No recommendation")
            return
        self.safe_use, self.use_rec = True, False
        if self.current_phase=='patched':
            if self.after_id:
                try: self.app.after_cancel(self.after_id)
                except Exception: pass
            self._recalc(); self._style('patched'); self._tick('patched',self.cur_pat)
        safe_m = max(2, float(self.rec)-2)
        self._toast(f"Safe {safe_m:.1f}m")

    def _manual_retrain(self):
        model = train_model_safe()
        if model:
            self.model = model
            self._toast("Model retrained")
        else:
            self._toast("Retrain skipped")

    def _recalc(self):
        cfg = WEEK_CONFIG[self.state['current_week']]
        base_u, base_p = cfg['unpatched']*60, cfg['patched']*60

        if self.use_rec and isinstance(self.rec,(int,float)):
            self.cur_unp = int(float(self.rec)*60)
            self.cur_pat = SECONDS_PER_BLOCK - self.cur_unp
        elif self.safe_use and isinstance(self.rec,(int,float)):
            safe_m = max(2, float(self.rec)-2)
            self.cur_unp = int(safe_m*60)
            self.cur_pat = SECONDS_PER_BLOCK - self.cur_unp
        else:
            if self.state['last_day_blended']:
                off = int(self.state['baseline_offset'])
                self.cur_unp, self.cur_pat = base_u+off, base_p-off
            else:
                self.cur_unp, self.cur_pat = base_u, base_p

        self.cur_unp = clamp(self.cur_unp, 0, SECONDS_PER_BLOCK)
        self.cur_pat = SECONDS_PER_BLOCK - self.cur_unp

        # Update split label to reflect current split for this interval
        u_m, p_m = self.cur_unp//60, self.cur_pat//60
        self.split_lbl.config(text=f"W{self.state['current_week']} • U{u_m} / P{p_m}")

    def _style(self, phase: str):
        color = PHASE_COLORS.get(phase, PHASE_COLORS['patched'])
        ttk.Style().configure('Phase.TFrame', background=color)
        self.current_phase = phase
        self.lbl.configure(foreground=(C_WARN if phase=='patched' else C_GOOD))

    # ----- Timer engine -----
    def _tick(self, phase: str, sec: int):
        self.current_phase = phase
        self.remaining = sec
        self.lbl.config(text=mmss(sec))
        if sec > 0:
            if not self.paused:
                self.after_id = self.app.after(1000, lambda: self._tick(phase, sec-1))
        else:
            if not self.paused:
                if phase=='patched':
                    notify_ntfy("Break ended")
                    ask_ok(self.app, "Switch", "Time to UNPATCH now.")
                    self._style('unpatched'); self._tick('unpatched',self.cur_unp)
                else:
                    notify_ntfy("Unpatch Started 40NF 30close")
                    self._rate_interval()

    def _get_rating(self) -> Optional[str]:
        return ask_choice(self.app, "Done", "How did that feel?",
                          [("Good","good"),("Tired","tired"),("Bad","bad")])

    def _rate_interval(self):
        if self.after_id:
            try: self.app.after_cancel(self.after_id)
            except Exception: pass

        rating = self._get_rating()
        if rating == 'bad':
            prev = max(1,self.state['current_week']-1)
            self.state['current_week']=prev; self.state['baseline_offset']=0
            save_state(self.state)
            ask_ok(self.app, "Bad Day", f"Reset to Week {prev}.")
            self.app.destroy(); return

        append_log(
            self.today,
            self.state['current_week'],
            self.idx,
            self.week_used,
            rating if rating else 'tired',
            self.cur_unp//60,
            self.cur_pat//60
        )
        self.idx+=1
        self.int_lbl.config(text=f"Int {self.idx-1}")
        self._start_interval()

    def _skip_to_unpatched(self):
        if self.after_id:
            try: self.app.after_cancel(self.after_id)
            except Exception: pass
        self._recalc()
        ask_ok(self.app, "Skip", "Skipping to UNPATCHED.")
        self._style('unpatched'); self._tick('unpatched',self.cur_unp)

    def _stop(self):
        es = get_log_entries(self.today)
        if (
            self.state['last_day_blended']
            and len(es)>=FULL_DAY_MIN
            and all(e['rating']=='good' for e in es)
            and time_since_last_bad(self.today)>=OFFSET_DELAY_DAYS
        ):
            self.state['baseline_offset']+=90
            nw = min(self.state['current_week']+1, max(WEEK_CONFIG))
            need = WEEK_CONFIG[nw]['unpatched']*60
            if self.state['baseline_offset']>=need:
                self.state['current_week']=nw
                self.state['baseline_offset']=0
                ask_ok(self.app, "Level Up!", f"Advanced to Week {nw}!")
        save_state(self.state)
        if self.after_id:
            try: self.app.after_cancel(self.after_id)
            except Exception: pass
        self.app.destroy()

    # ----- Flow -----
    def _start_interval(self):
        self._recalc(); self._style('patched'); self._tick('patched',self.cur_pat)

    # ----- UX helpers -----
    def _toast(self, msg: str, duration_ms: int = 1200):
        tw = tk.Toplevel(self.app)
        tw.overrideredirect(True)
        tw.attributes("-topmost", True)
        tw.attributes("-alpha", WINDOW_ALPHA)
        tw.configure(bg=C_ELEVATED)
        frm = ttk.Frame(tw, style='Card.TFrame', padding=(8,4,8,4))
        ttk.Label(frm, text=msg, style='Muted.TLabel').pack()
        frm.pack()

        self.app.update_idletasks()
        x = self.app.winfo_x() + self.app.winfo_width() - 220
        y = self.app.winfo_y() + self.app.winfo_height() - 70
        tw.geometry(f"+{x}+{y}")
        tw.after(duration_ms, tw.destroy)

if __name__=="__main__":
    today = datetime.date.today().isoformat()
    MainApp()
