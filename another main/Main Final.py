import tkinter as tk
from tkinter import ttk, messagebox
import os, json, csv, random, datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# === Configuration & Persistence ===
STATE_FILE       = "state.json"
LOG_FILE         = "log.csv"
MODEL_FILE       = "model.pt"

# Thresholds
FULL_DAY_MIN      = 6   # min intervals to count as a full day
MIN_BLEND_DAYS    = 4   # days in a row to enable earning offsets
BAD_DELAY_DAYS    = 3   # cooldown days after a bad before earning offsets
OFFSET_DELAY_DAYS = 2   # cooldown days after a bad before earning offsets

DEFAULT_STATE = {
    "current_week": 1,
    "good_day_streak": 0,
    "last_day_blended": False,
    "baseline_offset": 0
}

WEEK_CONFIG = {
    1: {"patched":45, "unpatched":15},
    2: {"patched":40, "unpatched":20},
    3: {"patched":35, "unpatched":25},
    4: {"patched":30, "unpatched":30},
    5: {"patched":25, "unpatched":35},
    6: {"patched":20, "unpatched":40},
    7: {"patched":15, "unpatched":45},
    8: {"patched":10, "unpatched":50},
}

def load_state():
    if not os.path.exists(STATE_FILE):
        return DEFAULT_STATE.copy()
    with open(STATE_FILE,'r') as f:
        return json.load(f)

def save_state(state):
    with open(STATE_FILE,'w') as f:
        json.dump(state, f, indent=2)

def append_log(date, bw, idx, wu, rating, used_unpatched, used_patched):
    """Always writes 7 columns, including week_used."""
    new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE,'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow([
                "date","baseline_week","interval_idx","week_used",
                "rating","used_unpatched","used_patched"
            ])
        w.writerow([
            date, bw, idx, wu, rating,
            used_unpatched, used_patched
        ])

def get_log_entries(date):
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, newline='') as f:
        return [r for r in csv.DictReader(f) if r['date']==date]

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
            if e['rating']=='bad':
                return days
    return days

def compute_streak(today):
    cnt = 0
    d = datetime.date.fromisoformat(today) - datetime.timedelta(days=1)
    while True:
        es = get_log_entries(d.isoformat())
        if not es:
            d -= datetime.timedelta(days=1)
            continue
        if any(e['rating']=='bad' for e in es):
            break
        if len(es) < FULL_DAY_MIN:
            break
        cnt += 1
        if cnt >= MIN_BLEND_DAYS:
            break
        d -= datetime.timedelta(days=1)
    return cnt

class IntervalDataset(Dataset):
    def __init__(self):
        self.data = []
        if not os.path.exists(LOG_FILE):
            return
        state = load_state()
        for r in csv.DictReader(open(LOG_FILE)):
            # parse used_unpatched
            un = r.get('used_unpatched') or '0'
            un = float(un)
            # parse used_patched, fallback to 60 - un
            pat = r.get('used_patched')
            if not pat:
                pat = 60 - un
            else:
                pat = float(pat)
            feat = [
                int(r['baseline_week']),
                int(r['interval_idx']),
                un,
                pat,
                datetime.date.fromisoformat(r['date']).toordinal(),
                state['good_day_streak'],
                time_since_last_bad(r['date'])
            ]
            self.data.append((
                torch.tensor(feat, dtype=torch.float32),
                torch.tensor(un, dtype=torch.float32)
            ))
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

class RegModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7,16), nn.ReLU(),
            nn.Linear(16,8), nn.ReLU(),
            nn.Linear(8,1)
        )
    def forward(self, x): return self.net(x).squeeze()

def train_model():
    print("Training model, please wait...", flush=True)
    ds = IntervalDataset()
    if len(ds) < 30:
        print("Not enough data; skipping.", flush=True)
        return None, 1e9
    idx = list(range(len(ds))); random.shuffle(idx)
    split = int(len(ds)*0.8)
    train_ds = [ds[i] for i in idx[:split]]
    val_ds   = [ds[i] for i in idx[split:]]
    loader   = DataLoader(train_ds, batch_size=16, shuffle=True)

    model, opt = RegModel(), optim.Adam(RegModel().parameters(), lr=1e-3)
    best_mae, best_w = 1e9, None
    for _ in range(20):
        model.train()
        for x,y in loader:
            opt.zero_grad()
            p = model(x)
            (torch.abs(p-y).mean()).backward()
            opt.step()
        model.eval()
        mae = sum(torch.abs(model(x)-y).item() for x,y in val_ds)/len(val_ds)
        if mae < best_mae*0.99:
            best_mae, best_w = mae, model.state_dict()
    if best_w:
        model.load_state_dict(best_w)
    torch.save(model.state_dict(), MODEL_FILE)
    print("Training complete.", flush=True)
    return model, best_mae

def load_model():
    try:
        if not os.path.exists(MODEL_FILE):
            raise FileNotFoundError
        m = RegModel()
        m.load_state_dict(torch.load(MODEL_FILE))
        m.eval()
        return m
    except Exception:
        model,_ = train_model()
        return model

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

        info = (f"Week:{state['current_week']}   "
                f"Streak:{state['good_day_streak']}   "
                f"Offset:{state['baseline_offset']}s   "
                f"SinceBad:{time_since_last_bad(today)}d")
        self.info_lbl = tk.Label(frm,text=info)
        if has and self.choice.get()=="continue":
            self.info_lbl.grid(sticky="w", pady=(5,10))

        self.week_var = tk.IntVar(value=state['current_week'])
        wf = ttk.LabelFrame(frm, text="Select starting week", padding=5)
        for w in range(1,9):
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

def start_session(eye_breaks, today):
    state = load_state()
    state['good_day_streak'] = compute_streak(today)
    state['last_day_blended'] = (
        state['good_day_streak'] >= MIN_BLEND_DAYS and
        time_since_last_bad(today) >= BAD_DELAY_DAYS
    )
    save_state(state)

    model = load_model()
    rec = None
    if model:
        base_unp = WEEK_CONFIG[state['current_week']]['unpatched']
        next_unp = WEEK_CONFIG[min(state['current_week']+1,8)]['unpatched']
        feat = torch.tensor([
            state['current_week'],
            1,
            state['good_day_streak'],
            int(state['last_day_blended']),
            datetime.date.fromisoformat(today).toordinal(),
            base_unp,
            WEEK_CONFIG[state['current_week']]['patched']
        ], dtype=torch.float32)
        with torch.no_grad():
            p = model(feat).item()
        rec = max(base_unp, min(p, next_unp))

    TimerApp(state, today, eye_breaks, model, rec)

class TimerApp:
    def __init__(self, state, today, eye_breaks, model, rec):
        self.state = state
        self.today = today
        self.model = model
        self.rec = rec
        self.idx = 1
        self.week_used = state['current_week']
        self.use_rec = False
        self.safe_use = False
        self.after_id = None

        self.app = tk.Tk()
        self.app.overrideredirect(True)
        self.app.attributes("-topmost", True)
        self.app.attributes("-alpha", 0.8)

        self.lbl = tk.Label(self.app, font=("Helvetica", 18), width=10)
        self.lbl.pack(fill="both", expand=True)
        self.int_lbl = tk.Label(self.app, text="Intervals Completed: 0", font=("Helvetica", 10))
        self.int_lbl.pack(fill="x")

        ctrl = ttk.Frame(self.app); ctrl.pack(fill="x")
        rec_text = f"Rec: {self.rec:.1f}m" if self.rec else "Rec: --"
        ttk.Label(ctrl, text=rec_text).pack(side="left")
        ttk.Button(ctrl, text="Use Rec", command=self._apply_rec).pack(side="left")
        ttk.Button(ctrl, text="Safe Use", command=self._apply_safe).pack(side="left")
        ttk.Button(ctrl, text="Retrain Now", command=self._manual_retrain).pack(side="left")

        ttk.Button(self.app, text="STOP", command=self._stop).pack(fill="x")
        ttk.Button(self.app, text="Skip to Unpatched", command=self._skip_to_unpatched).pack(fill="x")

        self.lbl.bind("<ButtonPress-1>", self._start_move)
        self.lbl.bind("<B1-Motion>", self._do_move)

        self._position()
        self._start_interval()
        self.app.mainloop()

    def _apply_rec(self):
        if not self.rec: return
        self.use_rec = True; self.safe_use = False
        self._recalc()
        if self.current_phase=='patched':
            if self.after_id: self.app.after_cancel(self.after_id)
            self._style('patched'); self._tick('patched', self.cur_pat)

    def _apply_safe(self):
        if not self.rec: return
        self.safe_use = True; self.use_rec = False
        self._recalc()
        if self.current_phase=='patched':
            if self.after_id: self.app.after_cancel(self.after_id)
            self._style('patched'); self._tick('patched', self.cur_pat)

    def _manual_retrain(self):
        model,_ = train_model()
        self.model = model
        if model:
            base_unp = WEEK_CONFIG[self.state['current_week']]['unpatched']
            next_unp = WEEK_CONFIG[min(self.state['current_week']+1,8)]['unpatched']
            feat = torch.tensor([
                self.state['current_week'], self.idx,
                self.state['good_day_streak'],
                int(self.state['last_day_blended']),
                datetime.date.fromisoformat(self.today).toordinal(),
                base_unp,
                WEEK_CONFIG[self.state['current_week']]['patched']
            ], dtype=torch.float32)
            with torch.no_grad():
                p = model(feat).item()
            self.rec = max(base_unp, min(p, next_unp))
        # update rec label
        for child in self.app.winfo_children():
            if isinstance(child, ttk.Frame):
                child.winfo_children()[0].config(text=f"Rec: {self.rec:.1f}m")
                break
        messagebox.showinfo("Retrain","Model retrained.", parent=self.app)

    def _recalc(self):
        cfg = WEEK_CONFIG[self.state['current_week']]
        base_unp = cfg['unpatched'] * 60
        base_pat = cfg['patched'] * 60

        if self.use_rec:
            self.cur_unp = int(self.rec * 60)
            self.cur_pat = 3600 - self.cur_unp

        elif self.safe_use:
            safe_m = max(2, self.rec - 2)
            self.cur_unp = int(safe_m * 60)
            self.cur_pat = 3600 - self.cur_unp

        else:
            if self.state['last_day_blended']:
                off = self.state['baseline_offset']
                self.cur_unp = base_unp + off
                self.cur_pat = base_pat - off
            else:
                self.cur_unp = base_unp
                self.cur_pat = base_pat

    def _start_move(self, e):
        self.sx, self.sy = e.x, e.y

    def _do_move(self, e):
        x = self.app.winfo_x() + e.x - self.sx
        y = self.app.winfo_y() + e.y - self.sy
        self.app.geometry(f"+{x}+{y}")

    def _position(self):
        self.app.update_idletasks()
        y = self.app.winfo_screenheight() - self.app.winfo_height()
        self.app.geometry(f"+0+{y}")

    def _start_interval(self):
        self._recalc()
        self._style('patched')
        self._tick('patched', self.cur_pat)

    def _style(self, phase):
        cols = {
            'patched':   ('#8B0000','white'),
            'unpatched': ('#90EE90','black'),
            'break':     ('#87CEFA','black')
        }
        bg, fg = cols[phase]
        self.lbl.config(bg=bg, fg=fg)
        self.app.config(bg=bg)

    def _tick(self, phase, sec):
        self.current_phase = phase
        self.lbl.config(text=f"{sec//60:02d}:{sec%60:02d}")
        if sec > 0:
            self.after_id = self.app.after(1000, lambda: self._tick(phase, sec-1))
            return
        if phase == 'patched':
            messagebox.showinfo("Time to unpatch","Unpatch now!", parent=self.app)
            self._style('unpatched')
            self._tick('unpatched', self.cur_unp)
        else:
            self._rate_interval()

    def _get_rating(self):
        dlg = tk.Toplevel(self.app); dlg.title("Interval complete")
        res = {'r': None}
        ttk.Label(dlg, text="Good / Tired / Bad?").pack(padx=10, pady=5)
        frm = ttk.Frame(dlg); frm.pack(pady=(0,10))
        ttk.Button(frm, text="Good",
            command=lambda:(res.update(r='good'), dlg.destroy())
        ).pack(side="left", padx=5)
        ttk.Button(frm, text="Tired",
            command=lambda:(res.update(r='tired'), dlg.destroy())
        ).pack(side="left", padx=5)
        ttk.Button(frm, text="Bad",
            command=lambda:(res.update(r='bad'), dlg.destroy())
        ).pack(side="left", padx=5)
        dlg.transient(self.app); dlg.grab_set(); self.app.wait_window(dlg)
        return res['r']

    def _rate_interval(self):
        if self.after_id:
            self.app.after_cancel(self.after_id)
        rating = self._get_rating()
        if rating == 'bad':
            prev = max(1, self.state['current_week'] - 1)
            self.state['current_week']    = prev
            self.state['baseline_offset'] = 0
            save_state(self.state)
            messagebox.showinfo("Bad Day",
                f"Reset to Week {prev}.", parent=self.app)
            self.app.destroy()
            return

        append_log(
            self.today,
            self.state['current_week'],
            self.idx,
            self.week_used,
            rating,
            self.cur_unp//60,
            self.cur_pat//60
        )

        self.idx += 1
        self.int_lbl.config(text=f"Intervals Completed: {self.idx-1}")
        self._start_interval()

    def _skip_to_unpatched(self):
        if self.after_id:
            self.app.after_cancel(self.after_id)
        # recompute to include offset
        self._recalc()
        messagebox.showinfo("Skipping","Now unpatching.", parent=self.app)
        self._style('unpatched')
        self._tick('unpatched', self.cur_unp)

    def _stop(self):
        es = get_log_entries(self.today)
        if (
            self.state['last_day_blended']
            and len(es) >= FULL_DAY_MIN
            and all(e['rating']=='good' for e in es)
            and time_since_last_bad(self.today) >= OFFSET_DELAY_DAYS
        ):
            self.state['baseline_offset'] += 60

        save_state(self.state)
        if self.after_id:
            self.app.after_cancel(self.after_id)
        self.app.destroy()

if __name__ == "__main__":
    MainApp()
