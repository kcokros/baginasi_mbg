# app_mbg_projection.py
# Streamlit app for MBG projections (manager-first UI) with robust column handling.

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
    seen = {}
    out = []
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
    """Normalize headers, map to canonical names, remove dup labels, drop preexisting t, parse month."""
    df = df.copy()
    # 1) make labels unique first (so rename doesn’t collide silently)
    df.columns = _dedup_columns([str(c) for c in df.columns])

    # 2) normalized view for mapping
    norm = {c: _normalize_header(c) for c in df.columns}

    # 3) map to canonical names
    rename: Dict[str, str] = {}
    for orig, n in norm.items():
        if n in {"month", "bulan", "periode", "periode bulan"}:
            rename[orig] = "month"
        elif ("penyalur" in n or "yayasan" in n or "sppg" in n) and ("rata" not in n):
            # e.g., "estimasi jumlah penyalur (yayasan/SPPG)"
            rename[orig] = "penyalur"
        elif (("rata" in n and "penyalur" in n) or ("avg" in n and "penyalur" in n)):
            rename[orig] = "avg_per_penyalur"
        elif ("realisasi" in n and "anggaran" in n) or n == "realisasi":
            rename[orig] = "realisasi"
        elif ("penerima akhir" in n) or ("penerima" in n and "akhir" in n) or ("end beneficiaries" in n):
            rename[orig] = "penerima"

    df = df.rename(columns=rename)

    # 4) drop duplicate names (keep first)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # 5) drop any existing 't' columns (including t.1 etc.)
    t_like = [c for c in df.columns if c == "t" or _normalize_header(c) == "t"]
    if t_like:
        df = df.drop(columns=t_like, errors="ignore")

    # 6) parse month
    if "month" in df.columns:
        df["month_dt"] = pd.to_datetime(df["month"].astype(str), errors="coerce")
    else:
        # try auto-detect a date-like column
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

    # 7) create fresh t = 1..N
    df["t"] = np.arange(1, len(df) + 1)

    # 8) derive missing fields if possible
    if "realisasi" not in df.columns and {"penyalur", "avg_per_penyalur"} <= set(df.columns):
        df["realisasi"] = df["penyalur"].astype(float) * df["avg_per_penyalur"].astype(float)
    if "avg_per_penyalur" not in df.columns and {"realisasi", "penyalur"} <= set(df.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            df["avg_per_penyalur"] = (df["realisasi"].astype(float) / df["penyalur"].astype(float)).replace([np.inf, -np.inf], np.nan)

    # keep canonical + extras (for transparency)
    keep = ["month", "month_dt", "t", "penyalur", "avg_per_penyalur", "realisasi", "penerima"]
    base = [c for c in keep if c in df.columns]
    extras = [c for c in df.columns if c not in base]
    return df[base + extras]

# =========================
# Model functions
# =========================
def logistic(t: np.ndarray, K: float, r: float, t0: float) -> np.ndarray:
    return K / (1 + np.exp(-r * (t - t0)))

def fit_logistic_penyalur(t: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Fit K, r, t0 with sane bounds; fallback to gentle growth if fit fails."""
    y = np.array(y, dtype=float)
    t = np.array(t, dtype=float)
    y_max = max(float(np.nanmax(y)), 1.0)
    K0, r0, t00 = y_max * 1.8, 0.5, float(np.median(t))
    bounds = ([y_max, 0.001, 0.0], [10000.0, 5.0, 36.0])
    try:
        params, _ = curve_fit(logistic, t, y, p0=[K0, r0, t00], bounds=bounds, maxfev=20000)
        K, r, t0 = [float(p) for p in params]
    except Exception:
        K, r, t0 = y_max * 1.2, 0.3, t00
    return K, r, t0

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
    regime_pukul_rata: bool = False,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Optional[int]], Dict[str, Tuple[float, float, float]]]:
    """
    Returns:
      - results per scenario (DataFrame with month, t, penyalur, avg_per_penyalur, realisasi, cumulative_realisasi, sisa_pagu, pagu)
      - cap_hit_month per scenario (1..12 or None)
      - fitted params per scenario (and baseline)
    """
    df = _canonicalize_columns(df_base)

    # filter to target year if month_dt is known; else assume it's the target year data
    if "month_dt" in df.columns and df["month_dt"].notna().any():
        df = df[df["month_dt"].dt.year == year].copy().reset_index(drop=True)

    # rebuild t after any filtering
    df["t"] = np.arange(1, len(df) + 1)

    # calendar for the year
    months = pd.date_range(f"{year}-01-01", f"{year}-12-01", freq="MS")
    res_base = pd.DataFrame({"month_dt": months, "month": months.strftime("%Y-%m"), "t": np.arange(1, 13)})

    # merge known actuals by month_dt if available; otherwise by t
    if "month_dt" in df.columns and df["month_dt"].notna().all():
        right = df[["month_dt"] + [c for c in ["penyalur", "avg_per_penyalur", "realisasi"] if c in df.columns]].copy()
        right = right.loc[:, ~right.columns.duplicated(keep="first")].drop_duplicates(subset=["month_dt"], keep="first")
        res = res_base.merge(right, on="month_dt", how="left")
    else:
        right = df[["t"] + [c for c in ["penyalur", "avg_per_penyalur", "realisasi"] if c in df.columns]].copy()
        right = right.loc[:, ~right.columns.duplicated(keep="first")].drop_duplicates(subset=["t"], keep="first")
        res = res_base.merge(right, on="t", how="left")

    # inject September inputs (t=9; idx=8)
    sep_idx = 8
    if sep_penyalur is not None:
        res.loc[sep_idx, "penyalur"] = float(sep_penyalur)
    if sep_avg_per_penyalur is not None:
        res.loc[sep_idx, "avg_per_penyalur"] = float(sep_avg_per_penyalur)
    if sep_realisasi_override is not None:
        res.loc[sep_idx, "realisasi"] = float(sep_realisasi_override)

    # compute Sep realisasi if not overridden and both parts are present
    if pd.notna(res.loc[sep_idx, "penyalur"]) and pd.notna(res.loc[sep_idx, "avg_per_penyalur"]) and pd.isna(res.loc[sep_idx, "realisasi"]):
        res.loc[sep_idx, "realisasi"] = float(res.loc[sep_idx, "penyalur"]) * float(res.loc[sep_idx, "avg_per_penyalur"])

    # Fit logistic on known penyalur up to Sep if toggled, else up to last known prior to Sep
    known_mask = res["penyalur"].notna() & (res["t"] <= (9 if use_sep_for_refit else min(8, res["t"].max())))
    t_fit = res.loc[known_mask, "t"].to_numpy(dtype=float)
    y_fit = res.loc[known_mask, "penyalur"].to_numpy(dtype=float)
    if len(y_fit) >= 3 and np.nanmax(y_fit) > 0:
        Kb, rb, t0b = fit_logistic_penyalur(t_fit, y_fit)
    else:
        y_max = float(np.nanmax(y_fit)) if len(y_fit) else 100.0
        Kb, rb, t0b = y_max * 1.2, 0.3, 6.0  # gentle fallback
    fitted_params: Dict[str, Tuple[float, float, float]] = {"baseline": (Kb, rb, t0b)}

    t_all = res["t"].to_numpy(dtype=float)
    # baseline average per penyalur
    if "avg_per_penyalur" in res.columns and res["avg_per_penyalur"].notna().any():
        last_avg = float(res["avg_per_penyalur"].dropna().iloc[-1])
    else:
        last_avg = np.nan

    results: Dict[str, pd.DataFrame] = {}
    cap_hits: Dict[str, Optional[int]] = {}

    for scen_name, (mK, mr, mt0) in scen_params.items():
        K, r, t0 = Kb * float(mK), rb * float(mr), t0b * float(mt0)
        fitted_params[scen_name] = (K, r, t0)

        # model penyalur then keep actuals where available
        penyalur_pred = logistic(t_all, K, r, t0)
        penyalur = penyalur_pred.copy()
        if "penyalur" in res.columns:
            known_p = res["penyalur"].notna()
            penyalur[known_p.to_numpy()] = res.loc[known_p, "penyalur"].to_numpy(dtype=float)

        # avg per penyalur: forward fill last observed; optional Sep–Dec flatten (regime pukul rata)
        if "avg_per_penyalur" in res.columns:
            avg = res["avg_per_penyalur"].astype(float).copy()
            if avg.notna().any():
                avg = avg.fillna(method="ffill")
                if avg.isna().any() and not math.isnan(last_avg):
                    avg = avg.fillna(last_avg)
            else:
                avg = pd.Series([last_avg] * len(res), index=res.index, dtype=float)
        else:
            avg = pd.Series([last_avg] * len(res), index=res.index, dtype=float)

        if regime_pukul_rata:
            # set months >= Sep to September level if present; else use last_avg
            sep_level = float(avg.iloc[sep_idx]) if pd.notna(avg.iloc[sep_idx]) else (last_avg if not math.isnan(last_avg) else 0.0)
            avg.loc[res["t"] >= 9] = sep_level

        # monthly realisasi raw
        realisasi = penyalur * avg.to_numpy(dtype=float)

        # keep actual realisasi where provided
        if "realisasi" in res.columns:
            known_r = res["realisasi"].notna()
            realisasi[known_r.to_numpy()] = res.loc[known_r, "realisasi"].to_numpy(dtype=float)

        # apply budget cap
        policy = "clip" if cap_policy == "clip" else "zero"
        realisasi_adj, cum_adj, hit = apply_budget_cap(realisasi, float(pagu_cap), policy=policy)

        out = pd.DataFrame({
            "month": res["month"],
            "t": res["t"],
            "penyalur": np.round(penyalur, 0),
            "avg_per_penyalur": np.round(avg, 0),
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

    # Try defaults if not uploaded
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
        st.warning("Belum ada file yang diupload dan tidak ditemukan file default. Upload Excel untuk melanjutkan.")
        st.stop()

    pagu_cap = st.number_input("Total Pagu (Rp)", min_value=0, value=51_525_000_000_000, step=1_000_000_000)
    cap_policy_label = st.selectbox("Kebijakan saat mendekati pagu", options=["clip (potong di bulan batas)", "zero (hentikan setelah pagu habis)"])
    cap_policy = "clip" if cap_policy_label.startswith("clip") else "zero"

    st.divider()
    st.subheader("Input September 2025 (opsional)")
    # Use 0 as "not provided" to avoid Streamlit None issues.
    in_sep_penyalur_val = st.number_input("Penyalur (SPPG) Sep-2025", min_value=0, value=0, step=1)
    in_sep_avg_val = st.number_input("Rata-rata realisasi per penyalur Sep-2025 (Rp)", min_value=0, value=0, step=1_000_000)
    in_sep_real_override_val = st.number_input("Realisasi total Sep-2025 (override, Rp)", min_value=0, value=0, step=10_000_000)

    use_sep_refit = st.checkbox("Gunakan data September untuk memperbarui proyeksi (refit)", value=True)
    regime_flat = st.checkbox("Regime 'pukul rata' (Sep–Des rata)", value=False)

    st.divider()
    with st.expander("Advanced (Analis)", expanded=False):
        st.caption("Atur multipliers skenario untuk parameter logistik (K, r, t0).")
        colB, colC = st.columns(2)
        with colB:
            K_mod = st.slider("Moderate: multiplier K", 0.5, 3.0, 1.5, 0.1)
            r_mod = st.slider("Moderate: multiplier r", 0.5, 3.0, 1.5, 0.1)
            t0_mod = st.slider("Moderate: multiplier t0", 0.5, 1.5, 1.25, 0.05)
        with colC:
            K_opt = st.slider("Optimistis: multiplier K", 0.5, 4.0, 2.0, 0.1)
            r_opt = st.slider("Optimistis: multiplier r", 0.5, 4.0, 2.0, 0.1)
            t0_opt = st.slider("Optimistis: multiplier t0", 0.5, 1.5, 1.30, 0.05)

# Diagnostics (helpful when columns differ)
df_canon = _canonicalize_columns(df_loaded)
with st.expander("🔍 Diagnostik Kolom (klik untuk melihat)", expanded=False):
    st.write("Kolom setelah normalisasi & deduplikasi:")
    st.write(list(df_canon.columns))
    st.dataframe(df_canon.head())

# Map 0 values to None (meaning: not provided)
sep_penyalur = None if in_sep_penyalur_val == 0 else float(in_sep_penyalur_val)
sep_avg = None if in_sep_avg_val == 0 else float(in_sep_avg_val)
sep_real = None if in_sep_real_override_val == 0 else float(in_sep_real_override_val)

# Scenario multipliers
scen_multipliers = {
    "conservative": (1.0, 1.0, 1.0),
    "moderate": (K_mod, r_mod, t0_mod),
    "optimistic": (K_opt, r_opt, t0_opt),
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
        cap_policy=cap_policy,
        scen_params=scen_multipliers,
        regime_pukul_rata=regime_flat,
    )

# KPI row
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

    # Download Excel for all scenarios
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for scen, df_out in results.items():
            df_out.to_excel(writer, sheet_name=scen, index=False)
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
    "Tip: Jika kolom di Excel Anda memiliki nama berbeda, aplikasi akan mencoba memetakan otomatis. "
    "Pastikan minimal ada kolom bulan (month) dan salah satu dari: "
    "1) penyalur & avg_per_penyalur (maka realisasi dihitung), atau 2) realisasi langsung."
)
