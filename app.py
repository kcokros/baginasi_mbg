# app_mbg_projection.py
# MBG Projection — Streamlit app with "ipynb parity" mode

import io
import math
import unicodedata
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import curve_fit

# =========================
# Helpers: normalization
# =========================
def _dedup_columns(cols: List[str]) -> List[str]:
    """Ensure column labels are unique by suffixing duplicates with .1, .2, ..."""
    seen, out = {}, []
    for c in cols:
        if c in seen:
            seen[c] += 1
            out.append(f"{c}.{seen[c]}")
        else:
            seen[c] = 0
            out.append(c)
    return out

def _ascii_clean(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return s.encode("ascii", "ignore").decode("ascii")

def _normalize_header(h: str) -> str:
    h = _ascii_clean(str(h).lower().strip())
    for ch in ["\n", "\t", "(", ")", "[", "]", "{", "}", ",", ";", ":", "/", "\\", "-", "_"]:
        h = h.replace(ch, " ")
    return " ".join(h.split())

def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize headers, map to canonical names, remove dup labels,
    drop preexisting 't', parse month, derive missing fields.
    """
    df = df.copy()
    df.columns = _dedup_columns([str(c) for c in df.columns])

    norm = {c: _normalize_header(c) for c in df.columns}
    rename: Dict[str, str] = {}
    for orig, n in norm.items():
        if n in {"month", "bulan", "periode", "periode bulan"}:
            rename[orig] = "month"
        elif ("penyalur" in n or "yayasan" in n or "sppg" in n) and ("rata" not in n):
            rename[orig] = "penyalur"
        elif (("rata" in n and "penyalur" in n) or ("avg" in n and "penyalur" in n)):
            rename[orig] = "avg_per_penyalur"
        elif ("realisasi" in n and "anggaran" in n) or n == "realisasi":
            rename[orig] = "realisasi"
        elif ("penerima akhir" in n) or ("penerima" in n and "akhir" in n) or ("end beneficiaries" in n):
            rename[orig] = "penerima"
    df = df.rename(columns=rename)

    # Drop dup names (keep first) and any pre-existing 't'
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    t_like = [c for c in df.columns if c == "t" or _normalize_header(c) == "t"]
    if t_like:
        df = df.drop(columns=t_like, errors="ignore")

    # Parse month if present (accepts a variety of formats)
    if "month" in df.columns:
        df["month_dt"] = pd.to_datetime(df["month"].astype(str), errors="coerce")
    else:
        date_like = [c for c in df.columns if any(k in _normalize_header(c) for k in ["date", "tanggal", "month", "bulan", "periode"])]
        if date_like:
            pick = date_like[0]
            df["month_dt"] = pd.to_datetime(df[pick].astype(str), errors="coerce")
            if "month" not in df.columns:
                df = df.rename(columns={pick: "month"})
        else:
            df["month_dt"] = pd.NaT

    if df["month_dt"].notna().any():
        df = df.sort_values("month_dt").reset_index(drop=True)

    # Fresh t = 1..N
    df["t"] = np.arange(1, len(df) + 1)

    # Derive missing fields if possible
    if "realisasi" not in df.columns and {"penyalur", "avg_per_penyalur"} <= set(df.columns):
        df["realisasi"] = df["penyalur"].astype(float) * df["avg_per_penyalur"].astype(float)
    if "avg_per_penyalur" not in df.columns and {"realisasi", "penyalur"} <= set(df.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            df["avg_per_penyalur"] = (
                df["realisasi"].astype(float) / df["penyalur"].astype(float)
            ).replace([np.inf, -np.inf], np.nan)

    keep = ["month", "month_dt", "t", "penyalur", "avg_per_penyalur", "realisasi", "penerima"]
    base = [c for c in keep if c in df.columns]
    extras = [c for c in df.columns if c not in base]
    return df[base + extras]

# =========================
# Model functions
# =========================
def logistic(t: np.ndarray, K: float, r: float, t0: float) -> np.ndarray:
    return K / (1 + np.exp(-r * (t - t0)))

def fit_logistic_penyalur(t: np.ndarray, y: np.ndarray, mode: str = "unbounded") -> Tuple[float, float, float]:
    """
    mode: 'unbounded' (match ipynb) | 'bounded' (robust default)
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if mode == "unbounded":
        p0 = [float(np.max(y)) * 2.0, 0.5, float(np.median(t))]
        params, _ = curve_fit(
            lambda tt, K, r, t0: K / (1 + np.exp(-r * (tt - t0))),
            t, y, p0=p0, maxfev=10000
        )
        return tuple(map(float, params))

    # bounded fallback (optional)
    y_max = max(float(np.nanmax(y)), 1.0)
    bounds = ([y_max, 0.001, 0.0], [10000.0, 5.0, 36.0])
    p0 = [y_max * 1.8, 0.5, float(np.median(t))]
    try:
        params, _ = curve_fit(
            lambda tt, K, r, t0: K / (1 + np.exp(-r * (tt - t0))),
            t, y, p0=p0, bounds=bounds, maxfev=20000
        )
        return tuple(map(float, params))
    except Exception:
        return (y_max * 1.2, 0.3, float(np.median(t)))

def apply_budget_cap(series: np.ndarray, cap: float, policy: str = "clip") -> Tuple[np.ndarray, np.ndarray, Optional[int]]:
    """
    policy:
      - 'clip': first breach month is clipped to hit cap; later months -> 0
      - 'zero': months at/after breach -> 0
    Returns: (adjusted, cumulative, cap_hit_month_index [1..12] or None)
    """
    series = np.array(series, dtype=float)
    cum = np.cumsum(series)

    if cap <= 0:
        zeros = np.zeros_like(series)
        return zeros, zeros, 1

    if policy == "zero":
        adj = np.where(cum <= cap, series, 0.0)
        cum_adj = np.cumsum(adj)
        first_exceed = np.argmax(cum > cap) + 1 if np.any(cum > cap) else None
        return adj, cum_adj, first_exceed

    # default: clip
    adj = series.copy()
    if np.any(cum > cap):
        idx = int(np.argmax(cum > cap))  # 0-based
        before = float(cum[idx - 1]) if idx > 0 else 0.0
        adj[idx] = max(cap - before, 0.0)
        if idx + 1 < len(adj):
            adj[idx + 1 :] = 0.0
        cum_adj = np.cumsum(adj)
        return adj, cum_adj, idx + 1
    else:
        return adj, cum, None

def linear_trend_forecast(y_hist: np.ndarray, horizon: int) -> np.ndarray:
    """Simple linear trend projection for averages."""
    y_hist = np.array(y_hist, dtype=float)
    t = np.arange(1, len(y_hist) + 1, dtype=float)
    if len(y_hist) < 2:
        return np.full(horizon, y_hist[-1] if len(y_hist) else 0.0)
    coef = np.polyfit(t, y_hist, 1)  # slope, intercept
    slope, intercept = coef[0], coef[1]
    t_future = np.arange(len(y_hist) + 1, len(y_hist) + horizon + 1, dtype=float)
    y_future = intercept + slope * t_future
    return np.maximum(y_future, 0.0)

# =========================
# Projection core
# =========================
def build_projection(
    df_base: pd.DataFrame,
    year: int,
    sep_penyalur: Optional[float],
    sep_avg_per_penyalur: Optional[float],
    sep_realisasi_override: Optional[float],
    pagu_cap: float,
    use_sep_for_refit: bool,
    cap_policy: str,
    scen_params: Dict[str, Tuple[float, float, float]],
    regime_pukul_rata: bool,
    avg_rule: str,          # "last_observed" | "linear_trend"
    parity_mode: bool,      # if True: ipynb parity (unbounded, flat avg, zero cap)
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Optional[int]], Dict[str, Tuple[float, float, float]]]:

    df = _canonicalize_columns(df_base)

    # filter to target year if month_dt known
    if "month_dt" in df.columns and df["month_dt"].notna().any():
        df = df[df["month_dt"].dt.year == year].copy().reset_index(drop=True)

    df["t"] = np.arange(1, len(df) + 1)

    # calendar 1..12
    months = pd.date_range(f"{year}-01-01", f"{year}-12-01", freq="MS")
    res_base = pd.DataFrame({"month_dt": months, "month": months.strftime("%Y-%m"), "t": np.arange(1, 13)})

    # merge actuals by month_dt if available
    if "month_dt" in df.columns and df["month_dt"].notna().all():
        right = df[["month_dt"] + [c for c in ["penyalur", "avg_per_penyalur", "realisasi"] if c in df.columns]].copy()
        right = right.loc[:, ~right.columns.duplicated(keep="first")].drop_duplicates(subset=["month_dt"], keep="first")
        res = res_base.merge(right, on="month_dt", how="left")
    else:
        right = df[["t"] + [c for c in ["penyalur", "avg_per_penyalur", "realisasi"] if c in df.columns]].copy()
        right = right.loc[:, ~right.columns.duplicated(keep="first")].drop_duplicates(subset=["t"], keep="first")
        res = res_base.merge(right, on="t", how="left")

    # Inject September inputs (t=9; idx=8)
    sep_idx = 8
    if sep_penyalur is not None:
        res.loc[sep_idx, "penyalur"] = float(sep_penyalur)
    if sep_avg_per_penyalur is not None:
        res.loc[sep_idx, "avg_per_penyalur"] = float(sep_avg_per_penyalur)
    if sep_realisasi_override is not None:
        res.loc[sep_idx, "realisasi"] = float(sep_realisasi_override)

    # Compute Sep realisasi if parts present and not overridden
    if pd.notna(res.loc[sep_idx, "penyalur"]) and pd.notna(res.loc[sep_idx, "avg_per_penyalur"]) and pd.isna(res.loc[sep_idx, "realisasi"]):
        res.loc[sep_idx, "realisasi"] = float(res.loc[sep_idx, "penyalur"]) * float(res.loc[sep_idx, "avg_per_penyalur"])

    # If refit is ON and Sep penyalur missing but (Sep realisasi & Sep avg) exist, derive Sep penyalur
    if use_sep_for_refit and (("penyalur" not in res.columns) or pd.isna(res.loc[sep_idx, "penyalur"])):
        if ("realisasi" in res.columns and "avg_per_penyalur" in res.columns and
            pd.notna(res.loc[sep_idx, "realisasi"]) and pd.notna(res.loc[sep_idx, "avg_per_penyalur"]) and float(res.loc[sep_idx, "avg_per_penyalur"]) > 0):
            res.loc[sep_idx, "penyalur"] = float(res.loc[sep_idx, "realisasi"]) / float(res.loc[sep_idx, "avg_per_penyalur"])

    # Fit logistic on known penyalur up to Sep if toggled, else up to Aug
    known_mask = res["penyalur"].notna() & (res["t"] <= (9 if use_sep_for_refit else 8))
    t_fit = res.loc[known_mask, "t"].to_numpy(dtype=float)
    y_fit = res.loc[known_mask, "penyalur"].to_numpy(dtype=float)

    fit_mode = "unbounded" if parity_mode else "bounded"  # ipynb uses unbounded
    if len(y_fit) >= 3 and np.nanmax(y_fit) > 0:
        Kb, rb, t0b = fit_logistic_penyalur(t_fit, y_fit, mode=fit_mode)
    else:
        y_max = float(np.nanmax(y_fit)) if len(y_fit) else 100.0
        Kb, rb, t0b = y_max * 1.2, 0.3, 6.0
    fitted_params: Dict[str, Tuple[float, float, float]] = {"baseline": (Kb, rb, t0b)}

    t_all = res["t"].to_numpy(dtype=float)

    # Build average series for projection
    avg = res["avg_per_penyalur"].astype(float) if "avg_per_penyalur" in res.columns else pd.Series([np.nan]*len(res))
    if avg.notna().any():
        last_idx = int(avg.last_valid_index())
        hist = avg.iloc[: last_idx + 1].dropna().to_numpy(dtype=float)
        # Parity mode: flat Sep–Dec = rounded last observed
        if parity_mode:
            last_avg_flat = float(int(round(hist[-1] if len(hist) else 0.0)))
            avg = avg.fillna(method="ffill")
            avg = avg.fillna(last_avg_flat)
            avg.loc[res["t"] >= 9] = last_avg_flat
        else:
            if avg_rule == "linear_trend" and len(hist) >= 2:
                horizon = len(res) - (last_idx + 1)
                fut = linear_trend_forecast(hist, horizon)
                avg = avg.copy()
                if horizon > 0:
                    avg.iloc[last_idx + 1 :] = fut
            else:
                avg = avg.fillna(method="ffill")
                if avg.isna().any():
                    avg = avg.fillna(hist[-1] if len(hist) else 0.0)
            if regime_pukul_rata:
                sep_level = float(avg.iloc[sep_idx]) if pd.notna(avg.iloc[sep_idx]) else float(avg.dropna().iloc[-1]) if avg.notna().any() else 0.0
                avg.loc[res["t"] >= 9] = sep_level
    else:
        avg = pd.Series([0.0] * len(res), index=res.index, dtype=float)

    # Projection per scenario
    results: Dict[str, pd.DataFrame] = {}
    cap_hits: Dict[str, Optional[int]] = {}

    for scen_name, (mK, mr, mt0) in scen_params.items():
        K, r, t0 = Kb * float(mK), rb * float(mr), t0b * float(mt0)
        fitted_params[scen_name] = (K, r, t0)

        # penyalur (float for calculation; we'll display rounded later)
        penyalur_pred_float = logistic(t_all, K, r, t0)
        penyalur_float = penyalur_pred_float.copy()
        if "penyalur" in res.columns:
            known_p = res["penyalur"].notna()
            penyalur_float[known_p.to_numpy()] = res.loc[known_p, "penyalur"].to_numpy(dtype=float)

        # monthly realisasi raw (UNROUNDED penyalur × avg)
        realisasi = penyalur_float * avg.to_numpy(dtype=float)

        # keep actual realisasi where provided
        if "realisasi" in res.columns:
            known_r = res["realisasi"].notna()
            realisasi[known_r.to_numpy()] = res.loc[known_r, "realisasi"].to_numpy(dtype=float)

        # budget cap
        policy = "zero" if parity_mode else ("clip" if cap_policy == "clip" else "zero")
        realisasi_adj, cum_adj, hit = apply_budget_cap(realisasi, float(pagu_cap), policy=policy)

        out = pd.DataFrame({
            "month": res["month"],
            "t": res["t"],
            # Display penyalur rounded (ipynb table showed rounded, but calculation used float)
            "penyalur": np.round(penyalur_float, 0),
            # Show avg as-is (Sep–Dec will be integer in parity mode)
            "avg_per_penyalur": avg,
            "realisasi": np.round(realisasi_adj, 0),
            "cumulative_realisasi": np.round(cum_adj, 0),
            "pagu": float(pagu_cap),
            "sisa_pagu": np.round(float(pagu_cap) - cum_adj, 0),
        })
        results[scen_name] = out
        cap_hits[scen_name] = hit

    return results, cap_hits, fitted_params

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="MBG Projection", layout="wide")
st.title("📈 Proyeksi Realisasi MBG 2025")

with st.sidebar:
    st.header("⚙️ Pengaturan")
    uploaded = st.file_uploader("Upload data agregat MBG bulanan (Excel)", type=["xlsx", "xls"])

    default_paths = [
        "data agregat MBG bulanan.xlsx",
        "/mnt/data/data agregat MBG bulanan.xlsx",  # if running in a sandbox
    ]

    df_loaded = None
    if uploaded is not None:
        try:
            df_loaded = pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"Gagal membaca file yang diupload: {e}")
    else:
        for p in default_paths:
            try:
                df_loaded = pd.read_excel(p)
                st.caption(f"Menggunakan file default: {p}")
                break
            except Exception:
                continue

    if df_loaded is None:
        st.warning("Belum ada file dan tidak ditemukan default. Upload Excel untuk lanjut.")
        st.stop()

    pagu_cap = st.number_input("Total Pagu (Rp)", min_value=0, value=51_525_000_000_000, step=1_000_000_000)

    st.divider()
    st.subheader("Input September 2025 (opsional)")
    in_sep_penyalur_val = st.number_input("Penyalur (SPPG) Sep-2025", min_value=0, value=0, step=1)
    in_sep_avg_val = st.number_input("Rata-rata realisasi per penyalur Sep-2025 (Rp)", min_value=0, value=0, step=1_000_000)
    in_sep_real_override_val = st.number_input("Realisasi total Sep-2025 (override, Rp)", min_value=0, value=0, step=10_000_000)

    use_sep_refit = st.checkbox("Gunakan data September untuk memperbarui proyeksi (refit)", value=True)

    st.divider()
    st.subheader("Mode")
    parity_mode = st.toggle("🔒 ipynb parity (unbounded fit, Sep–Dec flat avg, zero-after-cap)", value=True)

    st.divider()
    with st.expander("Advanced (non-parity)", expanded=False):
        cap_policy_label = st.selectbox("Kebijakan saat mendekati pagu (non-parity)", options=["clip", "zero"], index=0)
        avg_rule = st.selectbox("Aturan proyeksi rata-rata per penyalur (non-parity)", ["last_observed", "linear_trend"])
        regime_flat = st.checkbox("Regime 'pukul rata' (Sep–Des rata) — non-parity", value=False)
        colB, colC = st.columns(2)
        with colB:
            K_mod = st.slider("Moderate: multiplier K", 0.5, 3.0, 1.5, 0.1)
            r_mod = st.slider("Moderate: multiplier r", 0.5, 3.0, 1.5, 0.1)
            t0_mod = st.slider("Moderate: multiplier t0", 0.5, 1.5, 1.25, 0.05)
        with colC:
            K_opt = st.slider("Optimistis: multiplier K", 0.5, 4.0, 2.0, 0.1)
            r_opt = st.slider("Optimistis: multiplier r", 0.5, 4.0, 2.0, 0.1)
            t0_opt = st.slider("Optimistis: multiplier t0", 0.5, 1.5, 1.30, 0.05)

# Diagnostics
df_canon = _canonicalize_columns(df_loaded)
with st.expander("🔍 Diagnostik Kolom (klik untuk melihat)", expanded=False):
    st.write(list(df_canon.columns))
    st.dataframe(df_canon.head())

# Map 0 to None for optional inputs
sep_penyalur = None if in_sep_penyalur_val == 0 else float(in_sep_penyalur_val)
sep_avg = None if in_sep_avg_val == 0 else float(in_sep_avg_val)
sep_real = None if in_sep_real_override_val == 0 else float(in_sep_real_override_val)

# Scenario multipliers (ipynb used multipliers only on penyalur params; keep same interface)
scen_multipliers = {
    "conservative": (1.0, 1.0, 1.0),
    "moderate": (K_mod if not parity_mode else 1.5, r_mod if not parity_mode else 1.5, t0_mod if not parity_mode else 1.25),
    "optimistic": (K_opt if not parity_mode else 2.0, r_opt if not parity_mode else 2.0, t0_opt if not parity_mode else 1.30),
}

# Compute projections
with st.spinner("Menghitung proyeksi..."):
    results, cap_hits, params = build_projection(
        df_base=df_loaded,
        year=2025,
        sep_penyalur=sep_penyalur,
        sep_avg_per_penyalur=sep_avg,
        sep_realisasi_override=sep_real,
        pagu_cap=float(pagu_cap),
        use_sep_for_refit=use_sep_refit,
        cap_policy=cap_policy_label if not parity_mode else "zero",
        scen_params=scen_multipliers,
        regime_pukul_rata=(False if parity_mode else regime_flat),
        avg_rule=("last_observed" if parity_mode else avg_rule),
        parity_mode=parity_mode,
    )

# KPIs
col1, col2, col3 = st.columns(3)
for scen, df_out in results.items():
    total = float(df_out["realisasi"].sum())
    sisa = float(df_out["sisa_pagu"].iloc[-1])
    hit = cap_hits.get(scen)
    label = f"**{scen.title()}**"
    with (col1 if scen == "conservative" else col2 if scen == "moderate" else col3):
        st.metric(f"{label} • Total Realisasi (Rp)", f"{total:,.0f}")
        st.metric("Sisa Pagu (Rp)", f"{sisa:,.0f}")
        st.metric("Bulan Pagu Habis", f"{hit}" if hit else "—")

st.markdown("---")

# Tabs: table, cumulative, penyalur/avg
tab1, tab2, tab3 = st.tabs(["📊 Tabel", "📈 Kumulatif vs Pagu", "📉 Penyalur & Rata-rata"])

with tab1:
    scen_choice = st.selectbox("Pilih skenario untuk ditampilkan:", options=list(results.keys()), index=0)
    st.dataframe(results[scen_choice])

    # Download Excel with all scenarios
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for scen, df_out in results.items():
            # Make avg column round(2) for readability; keep raw numbers for calc elsewhere
            df_save = df_out.copy()
            df_save["average_spending_per_penyalur"] = df_save.pop("avg_per_penyalur")
            df_save.to_excel(writer, sheet_name=scen, index=False)
    st.download_button(
        "⬇️ Download Proyeksi (Excel)",
        data=buf.getvalue(),
        file_name="MBG_projection_with_scenarios_streamlit.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

with tab2:
    chart_df = None
    for scen, df_out in results.items():
        tmp = df_out[["month", "cumulative_realisasi"]].copy()
        tmp = tmp.rename(columns={"cumulative_realisasi": f"{scen}_cum"})
        chart_df = tmp if chart_df is None else chart_df.merge(tmp, on="month", how="outer")
    chart_df = chart_df.sort_values("month")
    st.line_chart(chart_df.set_index("month"))

with tab3:
    chart_df2 = None
    for scen, df_out in results.items():
        tmp = df_out[["month", "penyalur", "avg_per_penyalur"]].copy()
        tmp = tmp.rename(columns={"penyalur": f"{scen}_penyalur", "avg_per_penyalur": f"{scen}_avg"})
        chart_df2 = tmp if chart_df2 is None else chart_df2.merge(tmp, on="month", how="outer")
    chart_df2 = chart_df2.sort_values("month")
    st.line_chart(chart_df2.set_index("month"))

st.caption(
    "Parity ON: unbounded logistic fit, Sep–Dec flat average (rounded last observed), zero-after-cap — matches ipynb. "
    "Parity OFF: use bounded fit + chosen average rule/trend, optional 'pukul rata', and your chosen cap policy."
)
