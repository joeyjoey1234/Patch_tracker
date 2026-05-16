import tkinter as tk
from tkinter import ttk, messagebox
import os, json, csv, random, datetime

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor

# === Configuration & Persistence ===
STATE_FILE       = "state.json"
LOG_FILE         = "log.csv"
MODEL_FILE       = "model.xgb"

# Thresholds
FULL_DAY_MIN        = 6    # min intervals per day to count as a full day
MIN_BLEND_DAYS      = 4    # days in a row to enable earning offsets
BAD_DELAY_DAYS      = 3    # cooldown days after a bad before earning offsets
OFFSET_DELAY_DAYS   = 2    # cooldown days after a bad before earning offsets
REL_LOAD_THRESHOLD  = 0.05 # max 5% jump in unpatched time

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

def load_state():
    if not os.path.exists(STATE_FILE):
        return DEFAULT_STATE.copy()
    with open(STATE_FILE) as f:
        s = json.load(f)
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
            w.writerow([
                "date","baseline_week","interval_idx","week_used",
                "rating","used_unpatched","used_patched"
            ])
        w.writerow([date, bw, idx, wu, rating, used_unpatched, used_patched])

def get_log_entries(date_str):
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, newline='') as f:
        return [r for r in csv.DictReader(f) if r['date'] == date_str]

def time_since_last_bad(today):
    if not os.path.exists(LOG_FILE):
        return 999
    with open(LOG_FILE, newline='') as f:
        dates = sorted({r['date'] for r in csv.DictReader(f)})
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
    if not os.path.exists(LOG_FILE):
        return 0
    with open(LOG_FILE, newline='') as f:
        dates = sorted({r['date'] for r in csv.DictReader(f)})
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

# === Machine Learning with XGBoost ===
def train_model():
    print("Training model with XGBoost...", flush=True)
    if not os.path.exists(LOG_FILE):
        print("No logs found; skipping training.", flush=True)
        return None
    df = pd.read_csv(LOG_FILE, parse_dates=['date'])
    if len(df) < 30:
        print("Not enough data (<30 rows); skipping training.", flush=True)
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
        idx = np.arange(len(df)); np.random.shuffle(idx)
        split = int(len(df)*0.8)
        train_mask = df.index.isin(idx[:split])
        val_mask   = df.index.isin(idx[split:])

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]

    model = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8
    )
    try:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val,y_val)],
            early_stopping_rounds=10,
            verbose=False
        )
    except TypeError:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val,y_val)],
            verbose=False
        )

    joblib.dump(model, MODEL_FILE)
    print("Training complete.", flush=True)
    return model

def load_model():
    if os.path.exists(MODEL_FILE):
        try:
            return joblib.load(MODEL_FILE)
        except:
            pass
    return train_model()

# === Core App Logic ===
def start_session(eye_breaks, today):
    state = load_state()
    state['good_day_streak']  = compute_streak(today)
    state['last_day_blended'] = (
        state['good_day_streak'] >= MIN_BLEND_DAYS and
        time_since_last_bad(today) >= BAD_DELAY_DAYS
    )
    save_state(state)

    model = load_model()
    rec = None
    base_unp = WEEK_CONFIG[state['current_week']]['unpatched']
    next_unp = WEEK_CONFIG.get(state['current_week']+1, WEEK_CONFIG[max(WEEK_CONFIG)])['unpatched']

    if model:
        # compute rel_load from yesterday first interval
        yest = (datetime.date.fromisoformat(today) - datetime.timedelta(days=1)).isoformat()
        ents = get_log_entries(yest)
        rel = 0
        if ents:
            prev_u = float(ents[0]['used_unpatched'])
            rel = (base_unp - prev_u) / prev_u if prev_u else 0

        feat = np.array([[
            state['current_week'], 1,
            base_unp,
            WEEK_CONFIG[state['current_week']]['patched'],
            datetime.date.fromisoformat(today).toordinal(),
            state['good_day_streak'],
            time_since_last_bad(today),
            rel
        ]])
        p = model.predict(feat)[0]
        # gate by relative load
        if rel > REL_LOAD_THRESHOLD:
            rec = base_unp
        else:
            rec = max(base_unp, min(p, next_unp))

    TimerApp(state, today, eye_breaks, model, rec)

class MainApp:
    def __init__(self):
        state = load_state()
        today = datetime.date.today().isoformat()

        root = tk.Tk(); root.title("Patch Timer Setup")
        frm = ttk.Frame(root,padding=10); frm.grid()

        has = os.path.exists(STATE_FILE)
        self.choice = tk.StringVar(value="continue" if has else "new")
        ttk.Radiobutton(frm, text="Continue saved session",
                        variable=self.choice, value="continue",
                        command=self._update).grid(sticky="w")
        ttk.Radiobutton(frm, text="Start new baseline",
                        variable=self.choice, value="new",
                        command=self._update).grid(sticky="w")

        info = (f"Week:{state['current_week']}  "
                f"Streak:{state['good_day_streak']}  "
                f"Offset:{state['baseline_offset']}s  "
                f"SinceBad:{time_since_last_bad(today)}d")
        self.info_lbl = tk.Label(frm, text=info)
        if has and self.choice.get()=="continue":
            self.info_lbl.grid(sticky="w", pady=(5,10))

        self.week_var = tk.IntVar(value=state['current_week'])
        wf = ttk.LabelFrame(frm, text="Select starting week", padding=5)
        for w in range(1, max(WEEK_CONFIG)+1):
            cfg = WEEK_CONFIG[w]
            txt = f"Week {w}: {cfg['unpatched']}m unpatched / {cfg['patched']}m patched"
            ttk.Radiobutton(wf, text=txt, variable=self.week_var, value=w).pack(anchor="w")
        if self.choice.get()=="new":
            wf.grid(sticky="w", pady=(5,10))
        self.week_frame = wf

        self.eye_var = tk.BooleanVar()
        ttk.Checkbutton(frm,
            text="Include Eye Breaks (20m work / 3m break)",
            variable=self.eye_var).grid(sticky="w", pady=(5,10))

        ttk.Button(frm, text="GO", command=lambda:self._go(root,today)).grid()
        root.mainloop()

    def _update(self):
        if self.choice.get()=="new":
            self.week_frame.grid(sticky="w", pady=(5,10))
            self.info_lbl.grid_forget()
        else:
            self.week_frame.grid_forget()
            self.info_lbl.grid(sticky="w", pady=(5,10))

    def _go(self, root, today):
        state = load_state()
        if self.choice.get()=="new":
            state = DEFAULT_STATE.copy()
            state['current_week'] = self.week_var.get()
        save_state(state)
        root.destroy()
        start_session(self.eye_var.get(), today)

class TimerApp:
    def __init__(self, state, today, eye_breaks, model, rec):
        self.state, self.today, self.model, self.rec = state, today, model, rec
        self.idx, self.week_used = 1, state['current_week']
        self.use_rec = self.safe_use = False
        self.after_id = None; self.paused=False; self.remaining=0

        self.app = tk.Tk(); self.app.overrideredirect(True)
        self.app.attributes("-topmost", True); self.app.attributes("-alpha", 0.8)

        self.lbl = tk.Label(self.app, font=("Helvetica",18), width=10)
        self.lbl.pack(fill="both", expand=True)
        self.int_lbl = tk.Label(self.app, text="Intervals Completed: 0", font=("Helvetica",10))
        self.int_lbl.pack(fill="x")

        ctrl = ttk.Frame(self.app); ctrl.pack(fill="x")
        ttk.Label(ctrl, text=f"Rec: {self.rec:.1f}m" if self.rec else "Rec: --").pack(side="left")
        ttk.Button(ctrl, text="Use Rec", command=self._apply_rec).pack(side="left")
        ttk.Button(ctrl, text="Safe Use", command=self._apply_safe).pack(side="left")
        ttk.Button(ctrl, text="Retrain Now", command=self._manual_retrain).pack(side="left")
        ttk.Button(ctrl, text="Pause", command=self._toggle_pause).pack(side="left")

        # Relative-load indicator
        yest = (datetime.date.fromisoformat(today)-datetime.timedelta(days=1)).isoformat()
        ents = get_log_entries(yest)
        if ents:
            last_unp = float(ents[-1]['used_unpatched'])
            today_unp = self.rec or WEEK_CONFIG[self.state['current_week']]['unpatched']
            diff = today_unp - last_unp
            pct = (diff/last_unp*100) if last_unp else 0
            lbl = tk.Label(self.app,
                text=f"Load ↑ {diff:.0f} min ({pct:+.1f}%)",
                font=("Helvetica",10,"italic"))
            lbl.pack(fill="x", pady=(0,5))

        ttk.Button(self.app, text="STOP", command=self._stop).pack(fill="x")
        ttk.Button(self.app, text="Skip to Unpatched", command=self._skip_to_unpatched).pack(fill="x")

        self.lbl.bind("<ButtonPress-1>", self._start_move)
        self.lbl.bind("<B1-Motion>", self._do_move)

        self._position(); self._start_interval()
        self.app.mainloop()

    def _toggle_pause(self):
        if not self.paused:
            if self.after_id: self.app.after_cancel(self.after_id)
            self.paused = True
        else:
            self.paused = False
            self._tick(self.current_phase, self.remaining)

    def _apply_rec(self):
        if not self.rec: return
        self.use_rec, self.safe_use = True, False
        if self.current_phase=='patched':
            if self.after_id: self.app.after_cancel(self.after_id)
            self._recalc(); self._style('patched'); self._tick('patched',self.cur_pat)

    def _apply_safe(self):
        if not self.rec: return
        self.safe_use, self.use_rec = True, False
        if self.current_phase=='patched':
            if self.after_id: self.app.after_cancel(self.after_id)
            self._recalc(); self._style('patched'); self._tick('patched',self.cur_pat)

    def _manual_retrain(self):
        model = train_model()
        if model: self.model = model
        messagebox.showinfo("Retrain","Model retrained.", parent=self.app)

    def _recalc(self):
        cfg = WEEK_CONFIG[self.state['current_week']]
        base_u, base_p = cfg['unpatched']*60, cfg['patched']*60
        if self.use_rec:
            self.cur_unp = int(self.rec*60); self.cur_pat = 3600-self.cur_unp
        elif self.safe_use:
            safe_m = max(2, self.rec-2)
            self.cur_unp = int(safe_m*60); self.cur_pat = 3600-self.cur_unp
        else:
            if self.state['last_day_blended']:
                off = self.state['baseline_offset']
                self.cur_unp, self.cur_pat = base_u+off, base_p-off
            else:
                self.cur_unp, self.cur_pat = base_u, base_p

    def _start_move(self,e):
        self.sx,self.sy = e.x,e.y
    def _do_move(self,e):
        x = self.app.winfo_x()+e.x-self.sx
        y = self.app.winfo_y()+e.y-self.sy
        self.app.geometry(f"+{x}+{y}")
    def _position(self):
        self.app.update_idletasks()
        y = self.app.winfo_screenheight()-self.app.winfo_height()
        self.app.geometry(f"+0+{y}")

    def _start_interval(self):
        self._recalc(); self._style('patched'); self._tick('patched',self.cur_pat)

    def _style(self,phase):
        cols={'patched':('#8B0000','white'),'unpatched':('#90EE90','black'),'break':('#87CEFA','black')}
        bg,fg = cols[phase]
        self.lbl.config(bg=bg,fg=fg); self.app.config(bg=bg)

    def _tick(self,phase,sec):
        self.current_phase=phase; self.remaining=sec
        self.lbl.config(text=f"{sec//60:02d}:{sec%60:02d}")
        if sec>0 and not self.paused:
            self.after_id = self.app.after(1000, lambda: self._tick(phase,sec-1))
        elif not self.paused:
            if phase=='patched':
                messagebox.showinfo("Time to unpatch","Unpatch now!",parent=self.app)
                self._style('unpatched'); self._tick('unpatched',self.cur_unp)
            else:
                self._rate_interval()

    def _get_rating(self):
        dlg = tk.Toplevel(self.app); dlg.title("Interval complete")
        res={'r':None}
        ttk.Label(dlg,text="Good / Tired / Bad?").pack(padx=10,pady=5)
        fm=ttk.Frame(dlg); fm.pack(pady=(0,10))
        for txt,val in [("Good","good"),("Tired","tired"),("Bad","bad")]:
            ttk.Button(fm,text=txt,command=lambda v=val:(res.update(r=v),dlg.destroy())).pack(side="left",padx=5)
        dlg.transient(self.app); dlg.grab_set(); self.app.wait_window(dlg)
        return res['r']

    def _rate_interval(self):
        if self.after_id: self.app.after_cancel(self.after_id)
        rating = self._get_rating()
        if rating=='bad':
            prev = max(1,self.state['current_week']-1)
            self.state['current_week']=prev; self.state['baseline_offset']=0
            save_state(self.state)
            messagebox.showinfo("Bad Day",f"Reset to Week {prev}.",parent=self.app)
            self.app.destroy(); return

        append_log(
            self.today,
            self.state['current_week'],
            self.idx,
            self.week_used,
            rating,
            self.cur_unp//60,
            self.cur_pat//60
        )
        self.idx+=1
        self.int_lbl.config(text=f"Intervals Completed: {self.idx-1}")
        self._start_interval()

    def _skip_to_unpatched(self):
        if self.after_id: self.app.after_cancel(self.after_id)
        self._recalc(); messagebox.showinfo("Skipping","Now unpatching.",parent=self.app)
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
                messagebox.showinfo("Level Up!",f"Advanced to Week {nw}!",parent=self.app)
        save_state(self.state)
        if self.after_id: self.app.after_cancel(self.after_id)
        self.app.destroy()

if __name__=="__main__":
    MainApp()
