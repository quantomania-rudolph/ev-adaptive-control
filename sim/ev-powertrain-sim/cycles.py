"""Drive-cycle ingestion: turn ANY messy speed-versus-time file into a clean, checked speed request for the powertrain.

Design stance (learning note):
    Real data is never clean. The tests in tests/test_cycles.py enumerate 25+ ways it goes wrong, and each one is handled in
    exactly one of three ways:
      FIX    the loader repairs it and writes a line in `Cycle.report` (so the data team can see what was changed),
      ERROR  the loader refuses with a CycleError that names the problem and the remedy (never a silent wrong answer),
      WARN   the data is usable but suspicious (units guessed, speed too high for the motor); recorded in the report.
    Nothing is repaired silently. `Cycle.report` is itself data: it is saved next to every run.

Pipeline:  read (encoding, delimiter, header lines)  ->  pick columns (+ units)  ->  numeric coercion  ->  time repair
           ->  speed repair (NaN, negatives, spikes)  ->  uniform resample  ->  optional grade  ->  Cycle
Then:      VehicleEq maps vehicle speed to motor speed and the vehicle's inertia and road load to the motor shaft, and
           fit_vehicle_to_motor() sizes the equivalent mass so the cycle is physically drivable by the Tier 1 motor.
"""
from __future__ import annotations

import csv
import io
import pathlib
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

G = 9.80665
MPH, KMH = 0.44704, 1.0 / 3.6


class CycleError(ValueError):
    """Raised when a file cannot be turned into a trustworthy cycle. `.code` is a stable machine-readable id."""

    def __init__(self, code: str, msg: str):
        super().__init__(f"[{code}] {msg}")
        self.code = code


@dataclass
class Cycle:
    t: np.ndarray                       # s, starts at 0, uniform step dt
    v: np.ndarray                       # m/s, >= 0
    dt: float
    grade: np.ndarray | None = None     # rad, same length as t
    measured: dict = field(default_factory=dict)      # optional aligned real-world channels (for comparison only)
    report: list = field(default_factory=list)        # [{code, level, msg, n}]
    source: str = ""

    @property
    def duration(self) -> float:
        return float(self.t[-1])

    def accel(self) -> np.ndarray:
        return np.gradient(self.v, self.t)

    def note(self, code, level, msg, n=0):
        self.report.append(dict(code=code, level=level, msg=msg, n=int(n)))

    def codes(self):
        return [r["code"] for r in self.report]

    def report_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.report, columns=["code", "level", "msg", "n"])


# --------------------------------------------------------------------------------------------------------- reading
_TIME_NAMES = ["time", "t", "timestamp", "test time", "time_s", "seconds", "secs", "elapsed", "datetime"]
_SPEED_NAMES = ["speed", "vehicle_speed", "velocity", "v", "target speed", "speed_kmh", "speed_mph", "speed_ms",
                "vehicle speed", "veh_spd", "vel", "gps_speed"]


def _unit_from_text(text: str):
    s = text.lower()
    # lookarounds, not \b: underscore counts as a word character, so "speed_mph" would not match \bmph\b
    if re.search(r"(?<![a-z])mph(?![a-z])|mi/h|miles", s): return "mph"
    if re.search(r"km/h|(?<![a-z])kmh(?![a-z])|(?<![a-z])kph(?![a-z])|kilomet", s): return "kmh"
    if re.search(r"m/s|(?<![a-z])mps(?![a-z])|m\.s-1|ms-1", s): return "ms"
    return None


def _read_text(src):
    if isinstance(src, (bytes, bytearray)):
        raw = bytes(src)
    else:
        p = pathlib.Path(src)
        if not p.exists():
            raise CycleError("file_missing", f"no such file: {p}")
        if p.is_dir():
            raise CycleError("file_is_directory", f"{p} is a directory, expected a file")
        raw = p.read_bytes()
    if not raw.strip():
        raise CycleError("file_empty", "file is empty")
    # utf-16 only when the byte-order mark says so: arbitrary bytes "decode" as utf-16 into garbage (found by test 06)
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16"), "utf-16"
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return raw.decode(enc), enc
        except UnicodeError:
            continue
    raise CycleError("file_encoding", "could not decode file as utf-8, utf-16 or latin-1")


_NUM_CELL = re.compile(r"^[-+]?\d[\d.,eE+-]*\s*[A-Za-z/%\u00b0]{0,6}$")     # a number, optionally followed by a short unit


def _numeric_row(cells) -> int:
    return sum(1 for c in cells if c.strip() and _NUM_CELL.match(c.strip()))


def _split(line, delim):
    return re.split(r"\s+", line.strip()) if delim == "ws" else next(csv.reader([line], delimiter=delim))


def _sniff(sample: str):
    for d in (",", ";", "\t", "|"):
        counts = [len(l.split(d)) for l in sample.splitlines()[:50] if l.strip()]
        if counts and min(counts) > 1 and max(counts) == min(counts):
            return d
    best = max((",", ";", "\t", "|"), key=lambda d: sample.count(d))
    return best if sample.count(best) else "ws"


def read_table(src, report: list):
    """Text -> DataFrame of raw strings, skipping title lines, tolerating odd delimiters and decimal commas."""
    txt, enc = _read_text(src)
    if enc != "utf-8-sig":
        report.append(dict(code="encoding_fallback", level="fix", msg=f"decoded as {enc}", n=0))
    lines = [l for l in txt.splitlines() if l.strip() and not l.lstrip().startswith(("#", "//"))]
    if not lines:
        raise CycleError("file_empty", "file has no data lines (only comments or blanks)")
    delim = _sniff("\n".join(lines[:200]))
    rows = [_split(l, delim) for l in lines]
    first_num = next((i for i, r in enumerate(rows) if len(r) >= 2 and _numeric_row(r) >= 2 and _numeric_row(r) >= len(r) - 1), None)
    if first_num is None:
        raise CycleError("no_numeric_rows", "no row with two or more numeric columns was found")
    header_lines = lines[:first_num]
    if first_num >= 2:
        report.append(dict(code="title_lines_skipped", level="fix", msg=f"{first_num} non-numeric lines before the data", n=first_num))
    width = len(rows[first_num])
    # header = nearest preceding row with NO numeric cell (a data row with junk like "n/a" must not be mistaken for it)
    h = next((i for i in range(first_num - 1, -1, -1) if len(rows[i]) == width and _numeric_row(rows[i]) == 0), None)
    header = rows[h] if h is not None else None
    start = h + 1 if h is not None else first_num
    data = [r for r in rows[start:] if len(r) == width]
    dropped = len(rows) - start - len(data)
    if dropped:
        report.append(dict(code="ragged_rows_dropped", level="fix", msg=f"{dropped} rows with the wrong number of fields dropped", n=dropped))
    df = pd.DataFrame(data, columns=[h.strip() for h in header] if header else [f"col{i}" for i in range(width)])
    df.attrs["header_text"] = " ".join(header_lines)
    return df


# -------------------------------------------------------------------------------------------- column selection, units
def _pick(cols, names, what):
    low = {c.lower().strip(): c for c in cols}
    for n in names:
        if n in low:
            return low[n]
    for c in cols:
        if any(n in c.lower() for n in names if len(n) > 3):
            return c
    return None


def _to_float(series: pd.Series) -> pd.Series:
    if series.dtype.kind in "fiub":
        return series.astype(float)
    s = series.astype(str).str.strip().str.replace(r"[^\d,.\-+eE]", "", regex=True)
    # decimal comma ("12,5") but not a thousands comma ("1,234.5")
    comma_dec = s.str.contains(r"^-?\d+,\d{1,}$", regex=True) & ~s.str.contains(r"\.")
    s = s.where(~comma_dec, s.str.replace(",", ".", regex=False)).str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce")


def _time_to_seconds(t: np.ndarray, unit, name: str, rep: list):
    """Return seconds. Epoch and millisecond stamps are common in CAN logs (the TUM Tesla data is epoch ms)."""
    finite = t[np.isfinite(t)]
    if finite.size < 2:
        raise CycleError("too_few_samples", "fewer than two usable time stamps")
    if unit is None:
        n = name.lower()
        if re.search(r"\bms\b|milli|_ms", n): unit = "ms"
        elif re.search(r"\bus\b|micro", n): unit = "us"
        elif finite.max() > 1e14: unit = "us"
        elif finite.max() > 1e11: unit = "ms"
        elif finite.max() > 1e9: unit = "s"           # epoch seconds
        else:
            med = float(np.median(np.diff(np.sort(finite)))) if finite.size > 2 else 1.0     # sorted: input may be out of order
            unit = "ms" if med >= 20 else "s"
        rep.append(dict(code="time_unit_inferred", level="warn", msg=f"time unit guessed as {unit}; pass time_unit= to be sure", n=0))
    scale = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "min": 60.0}.get(unit)
    if scale is None:
        raise CycleError("bad_time_unit", f"unknown time_unit {unit!r}; use s, ms, us or min")
    return t * scale


# ------------------------------------------------------------------------------------------------------------ loader
def load_cycle(src, *, time_col=None, speed_col=None, speed_unit=None, time_unit=None, dt=0.1, max_gap_s=5.0,
               a_max=6.0, v_max=80.0, min_duration_s=10.0, smooth_s=0.0, grade_cols=None, keep=None,
               on_gap="longest", strict=False, source=None) -> Cycle:
    """Load and sanitise a speed trace.

    src: path, bytes, DataFrame, or (t, v) arrays.   a_max: m/s^2 above which a sample is a spike (6 g-ish is impossible
    for a road car; tyres limit to ~1 g).   v_max: m/s above which units are assumed wrong (80 m/s = 288 km/h).
    on_gap: 'longest' keeps the longest continuous stretch when a gap exceeds max_gap_s; 'error' refuses.
    keep: dict {name: column} of extra real channels to align to the cycle grid (used to compare against measurements).
    strict: any WARN becomes an error.
    """
    rep: list = []
    name_t = name_v = None
    if isinstance(src, tuple) and len(src) == 2:
        df = pd.DataFrame({"time": np.asarray(src[0], float), "speed": np.asarray(src[1], float)})
        label = source or "arrays"
    elif isinstance(src, pd.DataFrame):
        df, label = src.copy(), source or "DataFrame"
    else:
        df, label = read_table(src, rep), source or str(src)[:120]
    if df.shape[0] == 0:
        raise CycleError("file_empty", "table has no rows")
    header_text = df.attrs.get("header_text", "")

    name_t = time_col or _pick(df.columns, _TIME_NAMES, "time")
    name_v = speed_col or _pick(df.columns, _SPEED_NAMES, "speed")
    if time_col and time_col not in df.columns:
        raise CycleError("column_missing", f"time column {time_col!r} not in {list(df.columns)[:12]}")
    if speed_col and speed_col not in df.columns:
        raise CycleError("column_missing", f"speed column {speed_col!r} not in {list(df.columns)[:12]}")
    if (name_t is None or name_v is None) and df.shape[1] == 2:
        name_t, name_v = df.columns
        rep.append(dict(code="two_column_fallback", level="info", msg=f"two columns only: used {name_t!r} as time and {name_v!r} as speed", n=0))
    if name_t is None or name_v is None:
        raise CycleError("column_ambiguous", f"cannot tell which columns are time/speed; got {list(df.columns)[:15]}; pass time_col= and speed_col=")
    if name_t == name_v:
        raise CycleError("column_ambiguous", "time and speed resolved to the same column")

    t = _to_float(df[name_t]).to_numpy(dtype=float, copy=True)      # copy: pandas may hand back a read-only view
    v = _to_float(df[name_v]).to_numpy(dtype=float, copy=True)
    n0 = len(t)
    bad_nonfinite = ~np.isfinite(t) | ~np.isfinite(v)
    if np.isinf(t).any() or np.isinf(v).any():
        rep.append(dict(code="inf_values", level="fix", msg="infinite values treated as missing", n=int((np.isinf(t) | np.isinf(v)).sum())))
        t[np.isinf(t)] = np.nan; v[np.isinf(v)] = np.nan
    if np.isnan(t).all():
        raise CycleError("time_all_missing", "time column has no numeric values")
    if np.isnan(v).all():
        raise CycleError("speed_all_missing", "speed column has no numeric values")

    # ---- units
    unit = speed_unit or _unit_from_text(name_v) or _unit_from_text(header_text)
    if unit is None:
        vmax_raw = np.nanmax(np.abs(v))
        unit = "kmh" if vmax_raw > 60 else "ms"
        rep.append(dict(code="speed_unit_inferred", level="warn", msg=f"speed unit guessed as {unit} from max {vmax_raw:.1f}; pass speed_unit=", n=0))
    f = {"ms": 1.0, "kmh": KMH, "mph": MPH, "kph": KMH}.get(unit)
    if f is None:
        raise CycleError("bad_speed_unit", f"unknown speed_unit {unit!r}; use ms, kmh or mph")
    v = v * f
    if time_unit is None and re.search(r"\bsec|\bseconds\b|\(s\)|\[s\]", (name_t + " " + header_text).lower()):
        time_unit = "s"                                   # the header says seconds: no guessing needed
    ts = _time_to_seconds(t, time_unit, name_t, rep)

    extra = {}
    for k, col in (keep or {}).items():
        if col in df.columns:
            extra[k] = _to_float(df[col]).to_numpy(dtype=float, copy=True)

    # ---- drop rows without time; time repair
    ok = np.isfinite(ts)
    if (~ok).any():
        rep.append(dict(code="rows_without_time", level="fix", msg="rows with no timestamp dropped", n=int((~ok).sum())))
    ts, v = ts[ok], v[ok]
    extra = {k: a[ok] for k, a in extra.items()}
    if ts.size < 2:
        raise CycleError("too_few_samples", "fewer than two samples remain after removing rows without time")
    if np.any(np.diff(ts) < 0):
        order = np.argsort(ts, kind="stable")
        rep.append(dict(code="time_not_monotonic", level="fix", msg="timestamps were out of order; sorted", n=int((np.diff(ts) < 0).sum())))
        ts, v = ts[order], v[order]
        extra = {k: a[order] for k, a in extra.items()}
    dup = np.diff(ts) == 0
    if dup.any():
        keep_idx = np.r_[~dup, True]            # keep the LAST sample of a repeated stamp
        rep.append(dict(code="duplicate_timestamps", level="fix", msg="duplicate timestamps collapsed (last value kept)", n=int(dup.sum())))
        ts, v = ts[keep_idx], v[keep_idx]
        extra = {k: a[keep_idx] for k, a in extra.items()}
    ts = ts - ts[0]

    # ---- gaps
    gaps = np.where(np.diff(ts) > max_gap_s)[0]
    if gaps.size:
        if on_gap == "error":
            raise CycleError("time_gap", f"{gaps.size} gap(s) longer than {max_gap_s}s; largest {np.diff(ts).max():.1f}s")
        edges = np.r_[0, gaps + 1, ts.size]
        seg = np.argmax(np.diff(edges))
        lo, hi = edges[seg], edges[seg + 1]
        rep.append(dict(code="time_gap_split", level="fix", msg=f"{gaps.size} gap(s) > {max_gap_s}s; kept the longest continuous stretch ({ts[hi-1]-ts[lo]:.0f}s)", n=int(gaps.size)))
        ts, v = ts[lo:hi], v[lo:hi]
        extra = {k: a[lo:hi] for k, a in extra.items()}
        ts = ts - ts[0]
    if ts[-1] < min_duration_s:
        raise CycleError("too_short", f"cycle is {ts[-1]:.1f}s long; minimum is {min_duration_s}s")

    # ---- speed repair
    if np.isnan(v).any():
        n_nan = int(np.isnan(v).sum())
        if n_nan > 0.5 * v.size:
            raise CycleError("speed_mostly_missing", f"{n_nan}/{v.size} speed samples are missing (>50%); choose another speed column")
        good = np.isfinite(v)
        v = np.interp(ts, ts[good], v[good])
        rep.append(dict(code="speed_nan_interpolated", level="fix", msg="missing speed samples linearly interpolated", n=n_nan))
    if (v < 0).any():
        neg = int((v < 0).sum())
        if neg > 0.2 * v.size:
            raise CycleError("reverse_driving", f"{neg} negative speeds (>20%): reversing is not supported; check sign convention")
        v = np.clip(v, 0, None)
        rep.append(dict(code="negative_speed_clipped", level="fix", msg="small negative speeds set to 0", n=neg))
    if np.nanmax(v) > v_max:
        raise CycleError("speed_implausible", f"max speed {np.nanmax(v):.0f} m/s exceeds {v_max} m/s: units are probably wrong (mph vs km/h vs m/s)")
    # spike removal: a sample whose implied acceleration to BOTH neighbours is impossible is replaced
    a = np.abs(np.diff(v) / np.maximum(np.diff(ts), 1e-6))
    spikes = np.zeros(v.size, bool)
    spikes[1:-1] = (a[:-1] > a_max) & (a[1:] > a_max)
    if spikes.any():
        v[spikes] = np.interp(ts[spikes], ts[~spikes], v[~spikes])
        rep.append(dict(code="spikes_removed", level="fix", msg=f"isolated samples implying > {a_max} m/s^2 replaced by interpolation", n=int(spikes.sum())))

    # ---- uniform grid
    step = np.diff(ts)
    med = float(np.median(step))
    if med > 2.0:
        rep.append(dict(code="sparse_sampling", level="warn", msg=f"median sample spacing {med:.2f}s is coarse; interpolating to the {dt}s grid", n=0))
    if np.std(step) > 0.2 * med:
        rep.append(dict(code="irregular_sampling", level="fix", msg="irregular sample spacing; resampled to a uniform grid", n=0))
    tg = np.arange(0.0, ts[-1] + 1e-9, dt)
    vg = np.interp(tg, ts, v)
    eg = {k: np.interp(tg, ts[np.isfinite(a)], a[np.isfinite(a)]) if np.isfinite(a).sum() > 2 else np.full_like(tg, np.nan) for k, a in extra.items()}
    if smooth_s > 0:
        w = max(1, int(round(smooth_s / dt)))
        vg = np.convolve(np.pad(vg, (w // 2, w - 1 - w // 2), mode="edge"), np.ones(w) / w, mode="valid")
        rep.append(dict(code="smoothed", level="fix", msg=f"moving average over {smooth_s}s applied", n=0))
    if np.ptp(vg) < 1e-6:
        rep.append(dict(code="constant_speed", level="warn", msg="speed never changes: nothing dynamic to test", n=0))
    if np.nanmax(vg) < 0.5:
        rep.append(dict(code="standstill", level="warn", msg="vehicle never moves (max < 0.5 m/s)", n=0))

    cyc = Cycle(t=tg, v=vg, dt=dt, measured=eg, report=rep, source=label)
    if grade_cols:
        _add_grade(cyc, df, grade_cols, ts_raw=None)
    if strict and any(r["level"] == "warn" for r in rep):
        raise CycleError("strict_warning", "; ".join(r["msg"] for r in rep if r["level"] == "warn"))
    return cyc


def _add_grade(cyc: Cycle, grade, cols, ts_raw=None):
    """Grade (rad) from an elevation trace measured on the cycle grid. Smoothed over ~10 s and capped at +-15 % slope (cap = 0.15 in the code; an earlier docstring said 25 %)."""
    elev = cyc.measured.get("elevation")
    if elev is None or not np.isfinite(elev).any():
        cyc.note("grade_unavailable", "warn", "no usable elevation channel; grade set to zero")
        return
    elev = pd.Series(elev).interpolate(limit_direction="both").to_numpy()
    med = pd.Series(elev).rolling(int(5.0 / cyc.dt) | 1, center=True, min_periods=1).median().to_numpy()   # kills GPS jumps
    w = max(3, int(30.0 / cyc.dt))
    es = np.convolve(np.pad(med, (w // 2, w - 1 - w // 2), mode="edge"), np.ones(w) / w, mode="valid")
    if es.size <= w + 1:
        cyc.note("grade_unavailable", "warn", f"cycle ({es.size * cyc.dt:.0f} s) is not longer than the 30 s slope window; grade set to zero")
        return
    dist = np.cumsum(cyc.v) * cyc.dt
    # slope over a 30 s window of distance, not sample to sample, so standstill does not divide by ~0
    h = w
    d_el = es[h:] - es[:-h]
    d_ds = dist[h:] - dist[:-h]
    slope_mid = np.where(d_ds > 10.0, d_el / np.maximum(d_ds, 1e-3), 0.0)
    slope = np.interp(np.arange(es.size), np.arange(slope_mid.size) + h / 2.0, slope_mid)
    cap = 0.15
    n_cap = int((np.abs(slope) > cap).sum())
    cyc.grade = np.arctan(np.clip(slope, -cap, cap))
    cyc.note("grade_from_elevation", "info" if n_cap < 0.02 * slope.size else "warn",
             f"grade = 30 s elevation change / distance, median-filtered, capped at +-{cap:.0%}; {n_cap} samples hit the cap", n_cap)


# ------------------------------------------------------------------------------------------------- vehicle mapping
@dataclass
class VehicleEq:
    """Vehicle reflected to the motor shaft. k = motor speed / vehicle speed [rad/s per m/s] = gear ratio / wheel radius."""
    k: float
    mass: float              # kg, equivalent (see fit_vehicle_to_motor)
    crr: float = 0.010
    cda: float = 0.62        # m^2 (Cd*A, Tesla Model 3 order of magnitude; an assumption)
    rho: float = 1.2
    drivetrain_b: float = 0.02

    def plant_kwargs(self):
        """Arguments for GEMPlant so the shaft sees the vehicle: inertia m/k^2, rolling m g Crr/k, drag 0.5 rho CdA/k^3."""
        return dict(j_load=self.mass / self.k ** 2, a=self.mass * G * self.crr / self.k, b=self.drivetrain_b,
                    c=0.5 * self.rho * self.cda / self.k ** 3)

    def grade_torque(self, grade):
        return self.mass * G * np.sin(grade) / self.k

    def required_torque(self, cyc: Cycle):
        """Shaft torque the cycle demands, from the same equations the plant integrates."""
        w = self.k * cyc.v
        pk = self.plant_kwargs()
        T = (pk["j_load"] * np.gradient(w, cyc.t) + pk["a"] * (w > 0.1) + pk["b"] * w + pk["c"] * w ** 2)
        if cyc.grade is not None:
            T = T + self.grade_torque(cyc.grade)
        return T


def fit_vehicle_to_motor(cyc: Cycle, *, omega_target=190.0, T_cap=74.0, quantile=0.99, headroom=0.85, mass_max=2500.0,
                         mass_min=20.0, allow_road_scale=False, **veh):
    """Choose k so peak vehicle speed maps to omega_target, then the largest equivalent mass whose torque demand stays
    under headroom*T_cap for `quantile` of the cycle. Returns (VehicleEq, info). The result is an EQUIVALENT mass, not the
    real one: the Tier 1 motor is smaller than a car motor and the real speed range needs field weakening (Tier 2)."""
    vpk = float(np.nanmax(cyc.v))
    if vpk < 0.5:
        raise CycleError("cannot_scale", "cycle never moves, cannot choose a gear ratio")
    k = omega_target / vpk
    lo, hi = mass_min, mass_max
    def tq(m):
        return float(np.quantile(np.abs(VehicleEq(k=k, mass=m, **veh).required_torque(cyc)), quantile))
    road_scale = 1.0
    if tq(lo) > headroom * T_cap:
        if not allow_road_scale:
            raise CycleError("cycle_infeasible", f"even a {lo} kg equivalent vehicle needs {tq(lo):.0f} N.m (> {headroom*T_cap:.0f}); "
                             "aerodynamic and rolling load at this speed exceed the Tier 1 motor. Use a slower cycle, or pass "
                             "allow_road_scale=True to shrink the road load (a toy scaling, recorded in the result).")
        base_cda, base_crr = veh.pop("cda", 0.62), veh.pop("crr", 0.010)
        a_, b_ = 0.02, 1.0
        for _ in range(40):
            m_ = 0.5 * (a_ + b_)
            def tq_s(m, s_=m_): return float(np.quantile(np.abs(VehicleEq(k=k, mass=m, cda=base_cda * s_, crr=base_crr * s_, **veh).required_torque(cyc)), quantile))
            a_, b_ = (m_, b_) if tq_s(lo) <= headroom * T_cap else (a_, m_)
        road_scale = a_
        if road_scale <= 0.021:
            raise CycleError("cycle_infeasible", "even with road load scaled to 2 % the cycle exceeds the motor")
        veh["cda"], veh["crr"] = base_cda * road_scale, base_crr * road_scale
    if tq(hi) <= headroom * T_cap:
        lo = hi
    else:
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if tq(mid) <= headroom * T_cap else (lo, mid)
    veq = VehicleEq(k=k, mass=lo, **veh)
    Tr = veq.required_torque(cyc)
    info = dict(k=k, mass_eq=lo, T_peak=float(np.abs(Tr).max()), T_q=float(np.quantile(np.abs(Tr), quantile)),
                frac_over_cap=float((np.abs(Tr) > T_cap).mean()), omega_peak=float(k * vpk), road_scale=road_scale)
    return veq, info


# ------------------------------------------------------------------------------------------- dataset-specific adapter
def load_tum_tesla(path, *, dt=0.1, **kw) -> Cycle:
    """TUM 'Electric powertrain data of a Tesla Model 3' track_N.csv (CC BY 4.0, doi:10.14459/2024mp1735471).

    Facts measured on the files we downloaded (not assumed): time is epoch milliseconds, ~10 Hz; `speed` is GPS speed in
    m/s (about 95-100 % coverage), `vehicle_speed` is whole km/h with gaps; columns differ between tracks, so every
    optional channel is looked up by name and skipped if absent."""
    df = pd.read_csv(path, low_memory=False)
    cols = {c: c for c in ("hv_battery_current", "hv_battery_voltage", "rear_motor_torque", "engine_rpm", "elevation",
                           "hv_soc", "accelerator_pedal") if c in df.columns}
    if "time" not in df.columns:
        raise CycleError("column_missing", "no 'time' column; this does not look like a TUM track file")
    gps = _to_float(df["speed"]) if "speed" in df.columns else None
    vs = _to_float(df["vehicle_speed"]) if "vehicle_speed" in df.columns else None
    if gps is not None and gps.notna().mean() >= 0.8:
        col, unit = "speed", "ms"
    elif vs is not None and vs.notna().mean() >= 0.5:
        col, unit = "vehicle_speed", "kmh"
    else:
        raise CycleError("speed_mostly_missing", "neither 'speed' nor 'vehicle_speed' has enough samples")
    cyc = load_cycle(df, time_col="time", speed_col=col, speed_unit=unit, time_unit="ms", dt=dt, keep=cols,
                     source=f"TUM Tesla Model 3 {pathlib.Path(path).name}", **kw)
    cyc.note("speed_column", "info", f"used '{col}' ({unit})", 0)
    missing = [c for c in ("hv_battery_current", "hv_battery_voltage", "rear_motor_torque", "engine_rpm", "elevation") if c not in cols]
    if missing:
        cyc.note("optional_channels_missing", "info", "not in this track: " + ", ".join(missing), len(missing))
    if "elevation" in cyc.measured:
        _add_grade(cyc, df, cols)
    return cyc


# ------------------------------------------------------------------------------------------- synthetic cycle classes
def synthetic_cycle(kind: str, dt: float = 0.1, duration: float = 20.0, v_peak: float = 25.0) -> Cycle:
    """Standard test shapes so the mismatch can be compared across CLASSES of speed request.

    ramp   gentle trapezoid (feasible: the data never asks for more than the motor can do)
    sine   speed oscillates around a mean (feasible if slow, tests the speed loop's tracking)
    step   instant speed jumps (INFEASIBLE BY DESIGN: a step needs infinite acceleration, so any mismatch
           it causes is attributable to the DATA, which is exactly what the attribution flag must show)
    """
    t = np.arange(0.0, duration + 1e-9, dt)
    if kind == "ramp":
        v = np.interp(t, [0, 0.4 * duration, 0.8 * duration, duration], [0, v_peak, v_peak, 0.6 * v_peak])
    elif kind == "sine":
        v = 0.55 * v_peak + 0.4 * v_peak * np.sin(2 * np.pi * t / (duration / 3.0) - np.pi / 2)
    elif kind == "step":
        edges = np.linspace(0, duration, 6)
        levels = np.array([0.0, 0.4, 1.0, 0.6, 0.85]) * v_peak
        v = levels[np.clip(np.searchsorted(edges, t, side="right") - 1, 0, len(levels) - 1)]
        v = v.astype(float)
    else:
        raise CycleError("bad_kind", f"unknown synthetic cycle '{kind}' (ramp, sine, step)")
    c = Cycle(t=t, v=np.maximum(v, 0.0), dt=dt, source=f"synthetic:{kind}")
    c.note("synthetic", "info", f"synthetic '{kind}' cycle, {duration:.0f} s, peak {v_peak:.1f} m/s", 0)
    return c


# ------------------------------------------------------------------------------------------- data-vs-controller attribution
def data_demand(cyc: Cycle, veq: VehicleEq, *, J_rotor=0.0388, p=3, psi=0.066, Lq=1.2e-3, i_lim=250.0, u_max=173.2):
    """What the DATA asks of the motor, independent of any controller.

    The controller receives omega_ref by LINEAR INTERPOLATION between cycle samples, so the acceleration it really sees on
    each interval is (w[k+1]-w[k])/dt. A central difference (np.gradient) would halve a one-sample jump and under-report it
    (found by the pipeline red-team: a 1 m/s jump in 0.1 s gave ratio 0.54 here before, against a true 1.05). So the demand is
    computed per SEGMENT and each sample takes the larger of its two neighbouring segments.

    T_req    shaft torque the request needs (rotor inertia included, from the same equations the plant integrates)
    T_avail  most torque the motor can hold at that speed with the NOMINAL voltage ceiling and the current limit
             (steady-state voltage ellipse with i_d = 0, resistance ignored)
    ratio    |T_req| / T_avail. Above 1 the request cannot be met by ANY controller: mismatch there is the data's fault.
    *_seg    the same three, one value per interval (length n-1); use these to look a value up at an arbitrary time.
    """
    w = veq.k * cyc.v
    pk = veq.plant_kwargs()
    dt = np.diff(cyc.t)
    acc = np.diff(w) / dt
    w_mid = 0.5 * (w[1:] + w[:-1])
    T_seg = (pk["j_load"] + J_rotor) * acc + pk["a"] * (w_mid > 0.1) + pk["b"] * w_mid + pk["c"] * w_mid ** 2
    if cyc.grade is not None:
        T_seg = T_seg + veq.grade_torque(0.5 * (cyc.grade[1:] + cyc.grade[:-1]))

    def avail(speed):
        w_el = np.maximum(np.abs(p * speed), 1.0)
        room = np.maximum((u_max / w_el) ** 2 - psi ** 2, 0.0)
        return 1.5 * p * psi * np.minimum(i_lim, np.sqrt(room) / Lq)

    A_seg = avail(np.maximum(np.abs(w[1:]), np.abs(w[:-1])))            # worst (fastest) speed in the interval: least torque available
    R_seg = np.abs(T_seg) / np.maximum(A_seg, 1e-9)
    # per-sample view: the larger demand of the two segments that touch the sample
    left = np.r_[R_seg[0], R_seg]
    right = np.r_[R_seg, R_seg[-1]]
    pick_right = right > left
    T_pad_l, T_pad_r = np.r_[T_seg[0], T_seg], np.r_[T_seg, T_seg[-1]]
    T_req = np.where(pick_right, T_pad_r, T_pad_l)
    ratio = np.maximum(left, right)
    return dict(T_req=T_req, T_avail=avail(w), ratio=ratio, T_req_seg=T_seg, T_avail_seg=A_seg, ratio_seg=R_seg)
