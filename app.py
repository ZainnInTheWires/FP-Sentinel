import html
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px # pyright: ignore[reportMissingImports]
import plotly.graph_objects as go # pyright: ignore[reportMissingImports]
import streamlit as st # type: ignore
from scipy.signal import spectrogram
from sklearn.ensemble import IsolationForest, RandomForestClassifier # type: ignore
from sklearn.inspection import permutation_importance # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score # type: ignore
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split # type: ignore
from sklearn.pipeline import make_pipeline # pyright: ignore[reportMissingModuleSource]
from sklearn.preprocessing import StandardScaler # pyright: ignore[reportMissingModuleSource]

st.set_page_config(page_title="FP-Sentinel | Predictive Maintenance", page_icon="🔬", layout="wide")

# ═════════════════════════════ Constants ═════════════════════════════
LABELS = ["Normal", "Imbalance-like", "High vibration"]
FEATURE_NAMES = [
    "Intensity RMS", "Intensity STD", "Peak deviation", "Crest factor",
    "Band RMS (5–150 Hz)", "Dominant frequency (Hz)",
    "Spectral centroid (Hz)", "High-band energy ratio",
    "Displacement RMS (ground truth)",
]
GROUND_TRUTH_FEATURE = FEATURE_NAMES[-1]          # NOT measurable by a real FPI sensor
SENSOR_FEATURES = FEATURE_NAMES[:-1]              # what the model and health score may use
SEVERITY_IDX = FEATURE_NAMES.index("Band RMS (5–150 Hz)")

OPTICS_AIRY = "Airy (Fabry–Pérot)"
OPTICS_TWO = "Two-beam approx. (low finesse)"

WARN_HEALTH, CRIT_HEALTH, EOL_HEALTH, HYST = 70, 40, 20, 5   # health-index thresholds (%)
COLORS = {"Normal": "#2ca02c", "Warning": "#e6a700", "Critical": "#d62728"}
ICONS = {"Normal": "🟢", "Warning": "🟡", "Critical": "🔴"}
ACTIONS = {
    "Normal": "Continue routine monitoring.",
    "Warning": "Schedule an inspection at the next planned stop.",
    "Critical": "Plan maintenance now and consider reducing load.",
}
NO_TREND = 1e6   # sentinel: "no wear trend detected"

FS = 1000
DURATION = 2.0
N = int(FS * DURATION)
T = np.arange(N) / FS

MACHINES = [   # simulated plant assets (deterministic presets)
    dict(name="Pump P-101", life=240, shape="Linear", day=40, seed=11),
    dict(name="Compressor C-204", life=200, shape="Linear", day=130, seed=12),
    dict(name="Fan F-310", life=160, shape="Accelerating", day=135, seed=13),
    dict(name="Motor M-415", life=120, shape="Accelerating", day=112, seed=14),
]

# Compatibility: newer Streamlit uses width="stretch", older uses use_container_width=True
import inspect
_STRETCH = ({"width": "stretch"} if "width" in inspect.signature(st.plotly_chart).parameters
            else {"use_container_width": True})


def show(fig):
    st.plotly_chart(fig, **_STRETCH)


# ═════════════════════════════ Simulation core ═════════════════════════════
def simulate_displacement(label, rng, amp_scale=1.0):
    """Simulated machine displacement (ground truth)."""
    phase = rng.uniform(0, 2 * np.pi)
    if label == "Normal":
        x = 0.12*np.sin(2*np.pi*28*T + phase) + 0.035*np.sin(2*np.pi*63*T) + 0.018*rng.normal(size=N)
    elif label == "Imbalance-like":
        f = rng.uniform(22, 38)
        x = 0.42*np.sin(2*np.pi*f*T + phase) + 0.09*np.sin(2*np.pi*2*f*T + 0.3) + 0.025*rng.normal(size=N)
    else:
        f1, f2 = rng.uniform(20, 35), rng.uniform(75, 130)
        x = (0.65*np.sin(2*np.pi*f1*T + phase) + 0.25*np.sin(2*np.pi*f2*T)
             + 0.11*np.sin(2*np.pi*5*T) + 0.04*rng.normal(size=N))
    return amp_scale * x


def simulate_wear_displacement(wear, rng):
    """Illustrative wear model: wear 0 = healthy, 1 = severely degraded (smooth blend of the classes)."""
    s = float(np.clip(wear, 0, 1))
    phase = rng.uniform(0, 2 * np.pi)
    return ((0.12 + 0.53*s) * np.sin(2*np.pi*28*T + phase)
            + 0.035*(1 - s) * np.sin(2*np.pi*63*T)
            + 0.25 * s**1.5 * np.sin(2*np.pi*100*T)
            + 0.11*s * np.sin(2*np.pi*5*T)
            + (0.018 + 0.022*s) * rng.normal(size=N))


def fp_signal(x, noise_level, drift_level, rng, optics=OPTICS_AIRY, reflectivity=0.30):
    """Photodetector intensity from displacement x (a.u.). The sensor sees only this signal."""
    delta = 0.7 + 4.0 * x                                    # round-trip phase (rad), arbitrary scaling
    if optics == OPTICS_AIRY:
        coef_f = 4 * reflectivity / (1 - reflectivity) ** 2  # coefficient of finesse
        intensity = 1.0 / (1.0 + coef_f * np.sin(delta / 2) ** 2)
    else:
        intensity = 0.5 + 0.42 * np.cos(delta)
    noise_term = noise_level * rng.normal(size=len(x))
    baseline = drift_level * np.linspace(-0.5, 0.5, len(x))
    return np.clip(intensity + noise_term + baseline, 0, 1)


def features_from_signal(x, y):
    """9 features: 8 from the intensity signal + displacement RMS (ground truth, kept only for ablation)."""
    y0 = y - np.mean(y)
    rms = float(np.sqrt(np.mean(y0**2)))
    std = float(np.std(y0))
    peak = float(np.max(np.abs(y0)))
    crest = peak / (rms + 1e-9)
    mag = np.abs(np.fft.rfft(y0)) / len(y0)
    freqs = np.fft.rfftfreq(len(y0), d=1 / FS)
    dom = float(freqs[1:][np.argmax(mag[1:])])
    centroid = float(np.sum(freqs * mag) / (np.sum(mag) + 1e-12))
    band = (freqs >= 5) & (freqs <= 150)
    band_rms = float(np.sqrt(2 * np.sum(mag[band] ** 2)))
    high = float(np.sum(mag[freqs >= 60] ** 2) / (np.sum(mag**2) + 1e-12))
    xrms = float(np.sqrt(np.mean((x - np.mean(x)) ** 2)))
    return np.array([rms, std, peak, crest, band_rms, dom, centroid, high, xrms], dtype=float)


@st.cache_data(show_spinner=False)
def build_dataset(samples_per_class, noise_level, drift_level, optics, reflectivity, seed):
    """noise_level may be a float or a (low, high) tuple for mixed-noise training sets."""
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for label in LABELS:
        for _ in range(samples_per_class):
            amp = rng.uniform(0.5, 2.0)
            nl = rng.uniform(*noise_level) if isinstance(noise_level, tuple) else noise_level
            x = simulate_displacement(label, rng, amp)
            y = fp_signal(x, nl, drift_level, rng, optics, reflectivity)
            rows.append(features_from_signal(x, y))
            labels.append(label)
    return pd.DataFrame(rows, columns=FEATURE_NAMES), np.array(labels)


@st.cache_data(show_spinner=False)
def calibrate_health(noise, drift, optics, refl, n=40, seed=99):
    """Commissioning step: sensor severity of a healthy vs a failed reference machine (amplitude 1.0)."""
    rng = np.random.default_rng(seed)
    out = {}
    for label in ("Normal", "High vibration"):
        vals = []
        for _ in range(n):
            x = simulate_displacement(label, rng, 1.0)
            y = fp_signal(x, noise, drift, rng, optics, refl)
            vals.append(features_from_signal(x, y)[SEVERITY_IDX])
        out[label] = float(np.median(vals))
    return out["Normal"], out["High vibration"]


def health_index(severity, base, fail):
    span = max(fail - base, 1e-6)
    return float(100 * (1 - np.clip((severity - base) / span, 0, 1)))


def level_from_health(h):
    return "Normal" if h >= WARN_HEALTH else ("Warning" if h >= CRIT_HEALTH else "Critical")


# ═════════════════════════════ Models ═════════════════════════════
def _rf():
    return RandomForestClassifier(n_estimators=180, max_depth=10, class_weight="balanced",
                                  random_state=21, n_jobs=-1)


def _quantile_baseline(train_vals, test_vals):
    q1, q2 = np.quantile(train_vals, [0.34, 0.67])
    return np.where(test_vals < q1, "Normal", np.where(test_vals < q2, "Imbalance-like", "High vibration"))


@st.cache_resource(show_spinner=False)
def train_models(noise, drift, optics, refl):
    X, labels = build_dataset(180, noise, drift, optics, refl, 2026)
    X_tr, X_te, y_tr, y_te = train_test_split(X, labels, test_size=0.25, random_state=21, stratify=labels)
    S_tr, S_te = X_tr[SENSOR_FEATURES], X_te[SENSOR_FEATURES]

    rf = _rf().fit(S_tr, y_tr)                                    # sensor-only (deployable)
    pred = rf.predict(S_te)
    rf_full = _rf().fit(X_tr, y_tr)                               # + displacement (oracle, ablation only)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")).fit(S_tr, y_tr)

    cv = cross_val_score(_rf(), X[SENSOR_FEATURES], labels,
                         cv=StratifiedKFold(5, shuffle=True, random_state=21), scoring="accuracy")
    perm = permutation_importance(rf, S_te, y_te, n_repeats=8, random_state=21, n_jobs=-1)
    perm_df = (pd.DataFrame({"Feature": SENSOR_FEATURES, "Importance": perm.importances_mean,
                             "Std": perm.importances_std}).sort_values("Importance"))

    # Anomaly detector: trained on healthy data only, alarm threshold = 5th percentile of healthy scores
    normal_tr = S_tr[y_tr == "Normal"]
    iso = IsolationForest(n_estimators=200, random_state=21).fit(normal_tr)
    iso_thr = float(np.percentile(iso.decision_function(normal_tr), 5))
    flags = iso.decision_function(S_te) < iso_thr
    iso_far = float(np.mean(flags[y_te == "Normal"]))
    iso_det = {lab: float(np.mean(flags[y_te == lab])) for lab in LABELS[1:]}

    return dict(
        rf=rf, rf_full=rf_full, lr=lr, iso=iso, iso_thr=iso_thr,
        acc=accuracy_score(y_te, pred), f1=f1_score(y_te, pred, average="macro"),
        cv_mean=float(cv.mean()), cv_std=float(cv.std()),
        lr_acc=accuracy_score(y_te, lr.predict(S_te)),
        full_acc=accuracy_score(y_te, rf_full.predict(X_te)),
        base_int_acc=accuracy_score(y_te, _quantile_baseline(X_tr["Intensity RMS"], X_te["Intensity RMS"])),
        oracle_thr_acc=accuracy_score(y_te, _quantile_baseline(X_tr[GROUND_TRUTH_FEATURE], X_te[GROUND_TRUTH_FEATURE])),
        cm=confusion_matrix(y_te, pred, labels=LABELS), perm=perm_df,
        iso_far=iso_far, iso_det=iso_det, X=X, labels=labels,
    )


@st.cache_resource(show_spinner=False)
def train_mixed(drift, optics, refl):
    X, labels = build_dataset(180, (0.0, 0.40), drift, optics, refl, 4242)
    return _rf().fit(X[SENSOR_FEATURES], labels)


@st.cache_data(show_spinner=False)
def robustness_sweep(train_noise, drift, optics, refl):
    single = train_models(train_noise, drift, optics, refl)["rf"]
    mixed = train_mixed(drift, optics, refl)
    rows = []
    for nl in np.round(np.arange(0.0, 0.61, 0.1), 2):
        X, labels = build_dataset(70, float(nl), drift, optics, refl, 777)   # unseen seed
        S = X[SENSOR_FEATURES]
        rows.append({"Test noise": float(nl),
                     "Trained at fixed noise": accuracy_score(labels, single.predict(S)),
                     "Trained on mixed noise (0–0.40)": accuracy_score(labels, mixed.predict(S))})
    return pd.DataFrame(rows)


# ═════════════════════════════ Wear-out, RUL, alarms ═════════════════════════════
def wear_profile(u, shape):
    u = float(np.clip(u, 0, 1))
    return u if shape == "Linear" else (np.exp(3 * u) - 1) / (np.exp(3) - 1)


@st.cache_data(show_spinner=False)
def simulate_history(life_days, shape, noise, drift, optics, refl, seed):
    """One simulated 2 s sensor snapshot per day, from day 0 to the (hidden) end of life."""
    rng = np.random.default_rng(seed)
    models = train_models(noise, drift, optics, refl)
    base, fail = calibrate_health(noise, drift, optics, refl)
    feats, wear = [], []
    for d in range(life_days + 1):
        s = wear_profile(d / life_days, shape)
        x = simulate_wear_displacement(s, rng)
        y = fp_signal(x, noise, drift, rng, optics, refl)
        feats.append(features_from_signal(x, y))
        wear.append(s)
    F = pd.DataFrame(feats, columns=FEATURE_NAMES)
    S = F[SENSOR_FEATURES]
    scores = models["iso"].decision_function(S)
    return pd.DataFrame({
        "day": np.arange(life_days + 1),
        "true_wear": wear,
        "band_rms": F["Band RMS (5–150 Hz)"].values,
        "health": [health_index(v, base, fail) for v in F["Band RMS (5–150 Hz)"]],
        "pattern": models["rf"].predict(S),
        "anomaly_score": scores,
        "anomaly_flag": scores < models["iso_thr"],
    })


def estimate_rul(days, health, method="Auto", window=30, eol=EOL_HEALTH, n_draws=400, seed=0):
    """Trend extrapolation of the health index to the end-of-life threshold, with an uncertainty band
    from the covariance of the fitted coefficients. Linear: h(t)=a+bt. Exponential: ln(100-h)=a+bt."""
    d = np.asarray(days, float)[-window:]
    h = np.asarray(health, float)[-window:]
    if len(d) < 8:
        return None
    fits = {}
    for m in ("Linear", "Exponential"):
        y = h if m == "Linear" else np.log(np.clip(100 - h, 1.0, None))
        coef, cov = np.polyfit(d, y, 1, cov=True)
        fitted = np.polyval(coef, d) if m == "Linear" else 100 - np.exp(np.polyval(coef, d))
        fits[m] = (coef, cov, float(np.sqrt(np.mean((fitted - h) ** 2))))
    chosen = min(fits, key=lambda k: fits[k][2]) if method == "Auto" else method
    coef, cov, rmse = fits[chosen]

    def rul_from(c):
        b, a = c
        if chosen == "Linear":
            if b >= -1e-9:
                return NO_TREND
            t_fail = (eol - a) / b
        else:
            if b <= 1e-9:
                return NO_TREND
            t_fail = (np.log(100 - eol) - a) / b
        return float(min(max(t_fail - d[-1], 0.0), NO_TREND))

    point = rul_from(coef)
    draws = np.random.default_rng(seed).multivariate_normal(coef, cov, size=n_draws)
    vals = np.array([rul_from(c) for c in draws])
    lo, hi = np.percentile(vals, [10, 90])
    return dict(model=chosen, point=point, lo=float(lo), hi=float(hi), coef=coef,
                rmse=rmse, last_day=float(d[-1]))


def fmt_rul(est):
    if est is None:
        return "Need ≥ 8 days of data"
    if est["point"] >= NO_TREND / 10:
        return "No wear trend"
    return f"{est['point']:.0f} d"


def fmt_rul_band(est):
    if est is None or est["point"] >= NO_TREND / 10:
        return "—"
    hi = "∞" if est["hi"] >= NO_TREND / 10 else f"{est['hi']:.0f}"
    return f"{est['lo']:.0f}–{hi} d"


def run_alarm_logic(health_values, warn=WARN_HEALTH, crit=CRIT_HEALTH, hyst=HYST):
    """Three-level alarm state machine with hysteresis (prevents alarm chatter near thresholds)."""
    state, log = "Normal", []
    for day, h in enumerate(health_values):
        new = state
        if state == "Normal":
            new = "Critical" if h < crit else ("Warning" if h < warn else "Normal")
        elif state == "Warning":
            if h < crit:
                new = "Critical"
            elif h >= warn + hyst:
                new = "Normal"
        else:
            if h >= crit + hyst:
                new = "Normal" if h >= warn + hyst else "Warning"
        if new != state:
            log.append({"Day": day, "From": state, "To": new, "Health (%)": round(float(h), 1), "Acknowledged": False})
            state = new
    cols = ["Day", "From", "To", "Health (%)", "Acknowledged"]
    return pd.DataFrame(log, columns=cols), state


def first_sustained(flags, k=3):
    run = 0
    for i, f in enumerate(flags):
        run = run + 1 if f else 0
        if run >= k:
            return i - k + 1
    return None


# ═════════════════════════════ Report / export helpers ═════════════════════════════
def build_report_html(snapshot, plant_df, alarm_log, deg_summary):
    def kv(d):
        return "".join(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>" for k, v in d.items())
    alarm_html = alarm_log.drop(columns=["Acknowledged"], errors="ignore").to_html(index=False) if len(alarm_log) else "<p>No alarm transitions.</p>"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>FP-Sentinel maintenance report</title>
<style>body{{font-family:Arial,sans-serif;max-width:900px;margin:2rem auto;color:#222}}
table{{border-collapse:collapse;margin:.5rem 0 1.5rem}}th,td{{border:1px solid #ccc;padding:6px 10px;text-align:left}}
th{{background:#f3f3f3}}.note{{background:#fff7e0;border:1px solid #e6c36a;padding:10px}}</style></head><body>
<h1>FP-Sentinel maintenance report</h1>
<p class="note"><b>Simulation only.</b> All values come from synthetic signals; no physical Fabry–Pérot sensor or machine is connected.</p>
<h2>Live sandbox snapshot</h2><table>{kv(snapshot)}</table>
<h2>Wear-out simulation summary</h2><table>{kv(deg_summary)}</table>
<h2>Alarm log (wear-out simulation)</h2>{alarm_html}
<h2>Plant overview (simulated assets)</h2>{plant_df.to_html(index=False)}
</body></html>"""


# ═════════════════════════════ Header & sidebar ═════════════════════════════
st.title("🔬 FP-Sentinel")
st.subheader("Fabry–Pérot Interferometer Simulation + AI Predictive Maintenance")
st.caption("Simulation demonstrator • Synthetic data only • No physical FPI hardware is connected")

with st.sidebar:
    st.header("Live simulation controls")
    condition = st.selectbox("Machine condition to simulate", LABELS, index=1)
    noise = round(float(st.slider("Noise level (relative)", 0.0, 0.60, 0.15, 0.01)), 2)
    drift = round(float(st.slider("Baseline drift", 0.0, 0.30, 0.05, 0.01)), 2)
    amplitude_scale = float(st.slider("Vibration amplitude scale", 0.5, 2.0, 1.0, 0.05))
    optics = st.radio("Optics model", [OPTICS_AIRY, OPTICS_TWO],
                      help="Airy is the full Fabry–Pérot transmission. The two-beam model is the low-finesse "
                           "cosine approximation.")
    refl = round(float(st.slider("Mirror reflectivity R (Airy only)", 0.05, 0.90, 0.30, 0.05,
                                 disabled=(optics != OPTICS_AIRY))), 2)
    seed_live = st.number_input("Live signal seed", min_value=0, max_value=99999, value=42, step=1,
                                help="Affects only the live signal. Training data uses fixed seeds.")
    st.divider()
    st.caption("Simulation parameters, not calibrated physical sensor settings. The model and the health "
               "score see only the intensity signal, never the simulated displacement.")

with st.spinner("Training models (cached after the first run)..."):
    models = train_models(noise, drift, optics, refl)
    base_sev, fail_sev = calibrate_health(noise, drift, optics, refl)

# ═════════════════════════════ Live snapshot (computed once, used by several tabs) ═════════════════════════════
rng_live = np.random.default_rng(int(seed_live))
x_live = simulate_displacement(condition, rng_live, amplitude_scale)
y_live = fp_signal(x_live, noise, drift, rng_live, optics, refl)
feat = features_from_signal(x_live, y_live)
feat_sensor = pd.DataFrame([feat[:-1]], columns=SENSOR_FEATURES)

severity = float(feat[SEVERITY_IDX])
health = health_index(severity, base_sev, fail_sev)
level = level_from_health(health)
probs = models["rf"].predict_proba(feat_sensor)[0]
pred_live = str(models["rf"].classes_[int(np.argmax(probs))])
confidence = float(np.max(probs))
anom_score = float(models["iso"].decision_function(feat_sensor)[0])
anom_flag = anom_score < models["iso_thr"]

tab_live, tab_deg, tab_plant, tab_model, tab_rob, tab_export = st.tabs(
    ["📡 Live monitor", "📉 Wear-out & RUL", "🏭 Plant overview", "🧠 Model evaluation",
     "🌪️ Robustness", "📤 Export & notes"])

# ───────────────────────── Tab 1: live monitor ─────────────────────────
with tab_live:
    st.markdown("### Live system health")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Health index", f"{health:.1f}%", f"{ICONS[level]} {level}", delta_color="off")
    c2.metric("Vibration severity (sensor-derived)", f"{severity:.3f} a.u.",
              help="RMS of the 5–150 Hz band of the intensity signal. Calibrated against a simulated healthy "
                   "and a failed reference machine at the current noise/drift.")
    c3.metric("Detected pattern", pred_live, f"{confidence:.0%} confidence", delta_color="off")
    c4.metric("Anomaly detector", "ANOMALY" if anom_flag else "Within envelope", f"score {anom_score:+.3f}", delta_color="off")

    if level == "Critical":
        st.error("⚠️ **CRITICAL:** Vibration is very high. " + ACTIONS[level])
    elif level == "Warning":
        st.warning("⚠️ **MAINTENANCE ADVISORY:** Elevated vibration. " + ACTIONS[level])
    else:
        st.success("✅ **STABLE:** Vibration is within safe limits. " + ACTIONS[level])
    st.caption("A single snapshot cannot give remaining life. RUL needs a trend: see the **Wear-out & RUL** tab.")

    colA, colB = st.columns([1, 2])
    with colA:
        if confidence < 0.60:
            st.warning(f"**Uncertain: inspect signal**\n\nTop class: {pred_live}\n\nConfidence: {confidence:.1%}")
        else:
            st.info(f"**Detected: {pred_live}**\n\nConfidence: {confidence:.1%}")
        if anom_flag:
            st.error("Anomaly detector: signal is outside the healthy envelope.")
        else:
            st.success("Anomaly detector: signal looks like healthy operation.")
    with colB:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=confidence * 100, title={"text": "Classifier confidence (%)"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#1f77b4"},
                   "steps": [{"range": [0, 60], "color": "#ffcccc"}, {"range": [60, 100], "color": "#ccffcc"}],
                   "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 60}}))
        fig_g.update_layout(height=230, margin=dict(l=20, r=20, t=40, b=10))
        show(fig_g)

    left, right = st.columns(2)
    with left:
        fig_x = px.line(x=T, y=x_live, labels={"x": "Time (s)", "y": "Displacement (a.u.)"},
                        title="Simulated displacement (ground truth, NOT seen by the model)")
        fig_x.update_traces(line_color="#2ca02c", line_width=1)
        fig_x.update_layout(height=330, margin=dict(l=20, r=20, t=50, b=20))
        show(fig_x)
    with right:
        fig_y = px.line(x=T, y=y_live, labels={"x": "Time (s)", "y": "Intensity (a.u.)"},
                        title="Simulated Fabry–Pérot photodetector signal (the only model input)")
        fig_y.update_traces(line_color="#1f77b4", line_width=1)
        fig_y.update_layout(height=330, margin=dict(l=20, r=20, t=50, b=20), yaxis_range=[-0.05, 1.05])
        show(fig_y)

    y0 = y_live - np.mean(y_live)
    spec = np.abs(np.fft.rfft(y0)) / len(y0)
    freqs = np.fft.rfftfreq(len(y0), d=1 / FS)
    left2, right2 = st.columns(2)
    with left2:
        fig_fft = px.line(x=freqs, y=spec, labels={"x": "Frequency (Hz)", "y": "Magnitude"},
                          title="FFT magnitude spectrum")
        fig_fft.update_traces(line_color="#d62728", line_width=1)
        fig_fft.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20), xaxis_range=[0, 250])
        show(fig_fft)
    with right2:
        f_sp, t_sp, S_sp = spectrogram(y0, fs=FS, nperseg=256, noverlap=192, scaling="spectrum")
        keep = f_sp <= 250
        fig_sp = go.Figure(go.Heatmap(z=10 * np.log10(S_sp[keep] + 1e-12), x=t_sp, y=f_sp[keep],
                                      colorscale="Viridis", colorbar=dict(title="dB")))
        fig_sp.update_layout(title="Spectrogram (time–frequency)", height=320, margin=dict(l=20, r=20, t=50, b=20),
                             xaxis_title="Time (s)", yaxis_title="Frequency (Hz)")
        show(fig_sp)

# ───────────────────────── Tab 2: wear-out & RUL ─────────────────────────
with tab_deg:
    st.markdown("### Wear-out simulation, trend-based RUL and alarm logic")
    st.caption("One simulated 2-second sensor snapshot per day. The health index comes from the sensor signal only; "
               "the true wear profile is hidden from the estimator and used only to validate it.")
    d1, d2, d3, d4 = st.columns(4)
    life_days = d1.slider("True life (days, hidden)", 60, 365, 180, 5)
    shape = d2.selectbox("Wear profile", ["Linear", "Accelerating"], index=1)
    observed = d3.slider("Days observed so far", 10, life_days, int(life_days * 0.75), 1)
    window = d4.slider("Trend window (days)", 10, 60, 30, 5)
    d5, d6, d7 = st.columns(3)
    method = d5.selectbox("Trend model", ["Auto", "Linear", "Exponential"],
                          help="Auto picks whichever fit has the lower error on the window.")
    wear_seed = int(d6.number_input("Wear seed", 0, 9999, 7, 1))
    reveal = d7.checkbox("Reveal ground-truth future and end-of-life day", value=False)

    hist = simulate_history(life_days, shape, noise, drift, optics, refl, wear_seed)
    obs = hist.iloc[: observed + 1]
    health_s = obs["health"].rolling(5, min_periods=1).mean()
    est = estimate_rul(obs["day"].values, health_s.values, method, window)

    full_s = hist["health"].rolling(5, min_periods=1).mean().values
    eol_hits = np.where(full_s <= EOL_HEALTH)[0]
    actual_eol = int(eol_hits[0]) if len(eol_hits) else None

    alarm_log, alarm_state = run_alarm_logic(obs["health"].values)
    first_anom = first_sustained(obs["anomaly_flag"].values, 3)
    warn_rows = alarm_log[alarm_log["To"].isin(["Warning", "Critical"])]
    first_warn = int(warn_rows["Day"].iloc[0]) if len(warn_rows) else None

    pred_fail = None
    if est is not None and est["point"] < NO_TREND / 10:
        pred_fail = est["last_day"] + est["point"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Health at day {observed}", f"{obs['health'].iloc[-1]:.1f}%",
              f"{ICONS[alarm_state]} {alarm_state}", delta_color="off")
    m2.metric("Estimated RUL", fmt_rul(est), f"10–90%: {fmt_rul_band(est)}", delta_color="off")
    m3.metric("Trend model used", est["model"] if est else "—", f"fit RMSE {est['rmse']:.1f} pts" if est else None, delta_color="off")
    if reveal and actual_eol is not None and pred_fail is not None:
        m4.metric("Ground-truth end of life", f"day {actual_eol}", f"estimate error {pred_fail - actual_eol:+.0f} d", delta_color="off")
    else:
        m4.metric("Ground-truth end of life", "hidden" if not reveal else "not reached", None)

    fig_h = go.Figure()
    if reveal:
        fut = hist.iloc[observed:]
        fig_h.add_trace(go.Scatter(x=fut["day"], y=fut["health"].rolling(5, min_periods=1).mean(), mode="lines",
                                   name="Simulated future (hidden from estimator)", line=dict(color="#9aa0a6", dash="dot")))
    fig_h.add_trace(go.Scatter(x=obs["day"], y=obs["health"], mode="markers", name="Daily health",
                               marker=dict(size=4, color="#9ecae1")))
    fig_h.add_trace(go.Scatter(x=obs["day"], y=health_s, mode="lines", name="Smoothed (5-day mean)",
                               line=dict(color="#1f77b4", width=2)))
    if est is not None and pred_fail is not None:
        tt = np.linspace(est["last_day"] - min(window, observed) + 1, min(pred_fail, life_days * 1.4), 80)
        b, a = est["coef"]
        fit_y = (a + b * tt) if est["model"] == "Linear" else 100 - np.exp(a + b * tt)
        fig_h.add_trace(go.Scatter(x=tt, y=np.clip(fit_y, 0, 100), mode="lines", name=f"{est['model']} extrapolation",
                                   line=dict(color="#ff7f0e", dash="dash")))
        lo_x = est["last_day"] + est["lo"]
        hi_x = est["last_day"] + (est["hi"] if est["hi"] < NO_TREND / 10 else life_days * 1.4)
        fig_h.add_vrect(x0=lo_x, x1=min(hi_x, life_days * 1.4), fillcolor="#ff7f0e", opacity=0.12, line_width=0,
                        annotation_text="predicted failure (10–90%)", annotation_position="top left")
    if reveal and actual_eol is not None:
        fig_h.add_vline(x=actual_eol, line_color="#d62728", annotation_text="actual end of life")
    for val, nm, col in [(WARN_HEALTH, "Warning", COLORS["Warning"]), (CRIT_HEALTH, "Critical", COLORS["Critical"]),
                         (EOL_HEALTH, "End of life", "#7f0000")]:
        fig_h.add_hline(y=val, line_dash="dot", line_color=col, annotation_text=nm, annotation_position="bottom right")
    fig_h.update_layout(title="Health index over time", xaxis_title="Day", yaxis_title="Health index (%)",
                        yaxis_range=[0, 105], height=430, margin=dict(l=20, r=20, t=50, b=20),
                        legend=dict(orientation="h", y=-0.2))
    show(fig_h)

    k1, k2, k3 = st.columns(3)
    k1.metric("First sustained anomaly flag", f"day {first_anom}" if first_anom is not None else "none yet")
    k2.metric("First Warning/Critical alarm", f"day {first_warn}" if first_warn is not None else "none yet")
    if reveal and actual_eol is not None and first_warn is not None:
        k3.metric("Warning lead time before end of life", f"{actual_eol - first_warn} d")
    else:
        k3.metric("Warning lead time before end of life", "reveal ground truth to see")

    st.markdown("#### Alarm log (hysteresis ±5 pts, thresholds: Warning < 70, Critical < 40)")
    if len(alarm_log):
        st.data_editor(alarm_log, hide_index=True, disabled=["Day", "From", "To", "Health (%)"],
                       column_config={"Acknowledged": st.column_config.CheckboxColumn("Acknowledged")})
    else:
        st.info("No alarm transitions yet in the observed period.")
    st.caption("Alarm hysteresis stops the alarm from chattering when the health index hovers near a threshold. "
               "RUL uncertainty comes from the covariance of the fitted trend; a short or noisy window gives a wide band.")

    deg_summary = {
        "Wear profile / true life (simulated)": f"{shape}, {life_days} days",
        "Days observed": observed,
        "Health at last observation": f"{obs['health'].iloc[-1]:.1f}%",
        "Alarm state": alarm_state,
        "Estimated RUL": f"{fmt_rul(est)} (10–90%: {fmt_rul_band(est)})",
        "Trend model": est["model"] if est else "n/a",
    }

# ───────────────────────── Tab 3: plant overview ─────────────────────────
with tab_plant:
    st.markdown("### Simulated plant overview")
    st.caption("Four simulated assets at different stages of wear, monitored with the same sensor-only pipeline. "
               "Each asset uses the current noise, drift and optics settings.")
    rows = []
    for m in MACHINES:
        h = simulate_history(m["life"], m["shape"], noise, drift, optics, refl, m["seed"]).iloc[: m["day"] + 1]
        hs = h["health"].rolling(5, min_periods=1).mean()
        e = estimate_rul(h["day"].values, hs.values, "Auto", 30)
        _, st_state = run_alarm_logic(h["health"].values)
        rows.append({
            "Asset": m["name"], "Sim. day": m["day"], "Health (%)": round(float(h["health"].iloc[-1]), 1),
            "Alarm": st_state, "Pattern": str(h["pattern"].iloc[-1]),
            "Anomaly": "Yes" if bool(h["anomaly_flag"].iloc[-3:].all()) else "No",
            "Est. RUL": fmt_rul(e), "RUL 10–90%": fmt_rul_band(e),
        })
    plant_df = pd.DataFrame(rows)
    fig_p = px.bar(plant_df, x="Asset", y="Health (%)", color="Alarm", color_discrete_map=COLORS,
                   category_orders={"Alarm": ["Normal", "Warning", "Critical"]}, range_y=[0, 100],
                   title="Asset health index")
    fig_p.add_hline(y=WARN_HEALTH, line_dash="dot", line_color=COLORS["Warning"])
    fig_p.add_hline(y=CRIT_HEALTH, line_dash="dot", line_color=COLORS["Critical"])
    fig_p.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
    show(fig_p)
    st.dataframe(plant_df, hide_index=True)
    st.caption("Trend model is chosen automatically per asset. Accelerating wear is where a straight-line fit would "
               "be optimistic, which is why the exponential option exists.")

# ───────────────────────── Tab 4: model evaluation ─────────────────────────
with tab_model:
    st.markdown("### Model evaluation (synthetic test set)")
    st.write("All scores describe this simulation only. The deployable model uses sensor-derived features only.")
    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric("Random Forest accuracy", f"{models['acc']*100:.1f}%")
    e2.metric("RF macro F1", f"{models['f1']:.3f}")
    e3.metric("5-fold CV accuracy", f"{models['cv_mean']*100:.1f}%", f"± {models['cv_std']*100:.1f}", delta_color="off")
    e4.metric("Logistic regression", f"{models['lr_acc']*100:.1f}%")
    e5.metric("Intensity-RMS threshold", f"{models['base_int_acc']*100:.1f}%")

    st.markdown("#### Ablation: how much does non-measurable information help?")
    abl = pd.DataFrame({
        "Model": ["RF, sensor features only (deployable)", "RF, sensor + displacement (oracle, not measurable)",
                  "Threshold on displacement RMS (oracle)"],
        "Accuracy (%)": [models["acc"] * 100, models["full_acc"] * 100, models["oracle_thr_acc"] * 100]})
    fig_a = px.bar(abl, x="Accuracy (%)", y="Model", orientation="h", range_x=[0, 100], text_auto=".1f")
    fig_a.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20), yaxis_title=None)
    show(fig_a)
    gap = (models["full_acc"] - models["acc"]) * 100
    st.caption(f"Oracle gain over the sensor-only model: {gap:+.1f} points. A real sensor cannot measure displacement "
               "directly, so only the first bar is a fair claim about a deployable system.")

    cA, cB = st.columns(2)
    with cA:
        fig_cm = px.imshow(models["cm"], text_auto=True, x=LABELS, y=LABELS, color_continuous_scale="Blues",
                           title="Confusion matrix (sensor-only RF)",
                           labels=dict(x="Predicted", y="True", color="Count"))
        fig_cm.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
        show(fig_cm)
    with cB:
        pm = models["perm"]
        fig_pi = go.Figure(go.Bar(x=pm["Importance"], y=pm["Feature"], orientation="h",
                                  error_x=dict(type="data", array=pm["Std"]), marker_color="#1f77b4"))
        fig_pi.update_layout(title="Permutation importance (accuracy drop, test set)", height=380,
                             margin=dict(l=20, r=20, t=50, b=20))
        show(fig_pi)

    st.markdown("#### Anomaly detector (Isolation Forest, trained on healthy data only)")
    a1, a2, a3 = st.columns(3)
    a1.metric("False-alarm rate on healthy test data", f"{models['iso_far']*100:.1f}%")
    a2.metric("Detection rate: imbalance-like", f"{models['iso_det']['Imbalance-like']*100:.1f}%")
    a3.metric("Detection rate: high vibration", f"{models['iso_det']['High vibration']*100:.1f}%")
    st.caption("Healthy samples span amplitude scales 0.5–2.0, so mild faults can overlap with strong-but-healthy "
               "operation. That overlap is realistic and is why anomaly flags are combined with alarm logic.")

# ───────────────────────── Tab 5: robustness ─────────────────────────
with tab_rob:
    st.markdown("### Noise robustness")
    st.write(f"The deployed classifier is trained at noise = **{noise:.2f}**. What happens when the sensor noise "
             "changes after deployment? Test sets use a different random seed from training.")
    if st.toggle("Run robustness sweep (a few seconds, then cached)", value=False):
        with st.spinner("Running sweep..."):
            sweep = robustness_sweep(noise, drift, optics, refl)
        fig_r = go.Figure()
        for col, colr in [("Trained at fixed noise", "#d62728"), ("Trained on mixed noise (0–0.40)", "#2ca02c")]:
            fig_r.add_trace(go.Scatter(x=sweep["Test noise"], y=sweep[col] * 100, mode="lines+markers", name=col,
                                       line=dict(color=colr)))
        fig_r.add_vline(x=noise, line_dash="dot", annotation_text="training noise (fixed model)")
        fig_r.update_layout(title="Accuracy vs test-time noise", xaxis_title="Test noise level",
                            yaxis_title="Accuracy (%)", yaxis_range=[0, 102], height=420,
                            margin=dict(l=20, r=20, t=50, b=20), legend=dict(orientation="h", y=-0.2))
        show(fig_r)
        st.caption("Training on a range of noise levels (noise augmentation) is the standard remedy for sensor noise "
                   "that is not fixed in the field.")
    else:
        st.info("Switch the toggle on to run the sweep.")

# ───────────────────────── Tab 6: export & notes ─────────────────────────
with tab_export:
    st.markdown("### Industrial data export (simulated)")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def tags(asset, health_v, sev_v, state, pattern, conf, anom, rul_v):
        p = f"ns=2;s=FPSentinel.{asset}."
        return {
            p + "HealthIndex": {"value": round(health_v, 1), "unit": "%", "quality": "SIMULATED"},
            p + "SeverityBandRMS": {"value": round(sev_v, 4), "unit": "a.u.", "quality": "SIMULATED"},
            p + "AlarmLevel": {"value": {"Normal": 0, "Warning": 1, "Critical": 2}[state], "text": state, "quality": "SIMULATED"},
            p + "PredictedPattern": {"value": pattern, "quality": "SIMULATED"},
            p + "ModelConfidence": {"value": None if conf is None else round(conf, 3), "quality": "SIMULATED"},
            p + "AnomalyFlag": {"value": bool(anom), "quality": "SIMULATED"},
            p + "RUL_Days": {"value": rul_v, "unit": "d", "quality": "SIMULATED"},
        }

    payload = {"timestamp": now, "mqtt_topic": "plant/line1/condition",
               "note": "Illustrative tag layout only. No OPC UA server or MQTT broker is running.", "tags": {}}
    payload["tags"].update(tags("LiveSandbox", health, severity, level, pred_live, confidence, anom_flag, None))
    for r in rows:
        r_est = None
        payload["tags"].update(tags(r["Asset"].replace(" ", "").replace("-", ""), r["Health (%)"], 0.0,
                                    r["Alarm"], r["Pattern"], None, r["Anomaly"] == "Yes",
                                    None if r["Est. RUL"] in ("No wear trend", "Need ≥ 8 days of data") else float(r["Est. RUL"].split()[0])))
    with st.expander("Preview tag payload (JSON)"):
        st.json(payload)

    snapshot = {
        "Generated (UTC)": now, "Simulated condition": condition, "Noise / drift / amplitude": f"{noise} / {drift} / {amplitude_scale}",
        "Optics model": optics + (f", R = {refl}" if optics == OPTICS_AIRY else ""),
        "Health index": f"{health:.1f}% ({level})", "Sensor-derived severity": f"{severity:.3f} a.u.",
        "Detected pattern": f"{pred_live} ({confidence:.0%})", "Anomaly detector": "ANOMALY" if anom_flag else "within envelope",
        "Recommended action": ACTIONS[level],
    }
    report_html = build_report_html(snapshot, plant_df, alarm_log, deg_summary)

    results = pd.DataFrame({
        "metric": ["RF accuracy (sensor-only)", "RF macro F1", "RF 5-fold CV mean", "RF 5-fold CV std", "Logistic regression accuracy",
                   "Intensity-RMS threshold accuracy", "RF + displacement accuracy (oracle)", "Displacement-RMS threshold accuracy (oracle)",
                   "Anomaly false-alarm rate", "Anomaly detection: imbalance-like", "Anomaly detection: high vibration"],
        "value": [models["acc"], models["f1"], models["cv_mean"], models["cv_std"], models["lr_acc"], models["base_int_acc"],
                  models["full_acc"], models["oracle_thr_acc"], models["iso_far"],
                  models["iso_det"]["Imbalance-like"], models["iso_det"]["High vibration"]]})
    dataset_df = models["X"].copy()
    dataset_df["label"] = models["labels"]

    b1, b2, b3, b4 = st.columns(4)
    b1.download_button("Maintenance report (HTML)", report_html.encode("utf-8"), "fp_sentinel_report.html", "text/html")
    b2.download_button("Tag payload (JSON)", json.dumps(payload, indent=2).encode("utf-8"), "fp_sentinel_tags.json", "application/json")
    b3.download_button("Synthetic dataset (CSV)", dataset_df.to_csv(index=False).encode("utf-8"), "fp_sentinel_synthetic_dataset.csv", "text/csv")
    b4.download_button("Model metrics (CSV)", results.to_csv(index=False).encode("utf-8"), "fp_sentinel_metrics.csv", "text/csv")

    with st.expander("Model assumptions and limitations", expanded=True):
        st.markdown("""
- Synthetic proof of concept, not a calibrated Fabry–Pérot sensor. No real machine data was used.
- The classifier, anomaly detector and health index use **only intensity-derived features**. Simulated displacement is ground truth and is used only for plotting and the ablation study.
- The Airy model is the standard Fabry–Pérot transmission with arbitrary phase scaling; the two-beam option is a low-finesse approximation.
- The health index is calibrated against simulated healthy and failed reference machines at the current noise/drift, which stands in for a commissioning baseline.
- The wear model blends the simulated vibration patterns smoothly. Real degradation is less regular, and RUL from a short trend window can be wrong. The uncertainty band shows only fitting uncertainty, not model-form error.
- Classes are simulated vibration patterns, not verified diagnoses of specific mechanical faults.
- The tag payload and report are illustrative. Nothing is connected to a PLC, SCADA system, OPC UA server or MQTT broker.
""")

st.markdown("---")
st.caption("FP-Sentinel v3.0 • Predictive Maintenance Simulator • Simulation only")
