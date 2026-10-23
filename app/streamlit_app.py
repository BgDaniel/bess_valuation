"""Streamlit front end of bess_valuation: configure, start and inspect valuation runs.

Start from the repository root:

    poetry install --with app
    poetry run streamlit run app/streamlit_app.py

Page "Läufe": parameter form, "Berechnen" starts a background job (bess_valuation.jobs),
the table shows all runs with progress, remaining time and key figures and links to
the result page. Page "Ergebnis": key figures, parameters, figures, downloads.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from bess_valuation import BatteryConfig, EngineSettings
from bess_valuation.jobs import DEFAULT_JOBS_ROOT, list_jobs, read_status, submit_job

JOBS_ROOT = DEFAULT_JOBS_ROOT
DEFAULT_CONFIG = Path("configs/bess_1mw_2mwh.yaml")
GENERATOR_CONFIG = "configs/prices_synthetic.yaml"
STATE_LABELS = {"queued": "● wartet", "running": "● läuft", "done": "● fertig", "failed": "● Fehler"}
FIGURE_TITLES = {
    "01_value_fan": "Wertverlauf",
    "02_value_decomposition": "Wertzerlegung",
    "04_soc": "Ladezustand",
    "05_dispatch_profile": "Fahrprofil vs. Preise",
    "09_continuation_value": "Fortsetzungswert",
    "12_vertex_library": "Vertex-Library",
}

CSS = """
<style>
  .block-container {padding-top: 2.2rem; max-width: 1400px;}
  h1, h2, h3 {letter-spacing: -0.01em;}
  .app-subtitle {color: #9aa4b2; margin-top: -0.6rem; margin-bottom: 1.4rem;}
  div[data-testid="stMetric"] {
      background: #121a27; border: 1px solid #1e2633; border-radius: 8px;
      padding: 0.8rem 1rem;
  }
  div[data-testid="stMetricLabel"] p {color: #9aa4b2;}
  div[data-testid="stForm"] {border-color: #1e2633;}
  section[data-testid="stSidebar"] {border-right: 1px solid #1e2633;}
</style>
"""


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None or seconds != seconds:
        return "–"
    seconds = int(round(seconds))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _default_config() -> BatteryConfig:
    if DEFAULT_CONFIG.exists():
        return BatteryConfig.from_yaml(DEFAULT_CONFIG)
    return BatteryConfig(name="bess", e_max=2.0, p_charge_max=1.0, p_discharge_max=1.0, soc_init=1.0)


# ------------------------------------------------------------------ page: runs


def _run_form() -> None:
    cfg = _default_config()
    defaults = EngineSettings()
    with st.form("new_run", border=True):
        st.subheader("Neuer Lauf")
        col_b, col_p, col_n = st.columns(3, gap="large")
        with col_b:
            st.markdown("**Batterie**")
            name = st.text_input("Name", cfg.name)
            c1, c2 = st.columns(2)
            e_min = c1.number_input("e_min [MWh]", value=float(cfg.e_min), min_value=0.0, step=0.5)
            e_max = c2.number_input("e_max [MWh]", value=float(cfg.e_max), min_value=0.1, step=0.5)
            p_ch = c1.number_input("Ladeleistung [MW]", value=float(cfg.p_charge_max), min_value=0.01, step=0.5)
            p_dis = c2.number_input("Entladeleistung [MW]", value=float(cfg.p_discharge_max), min_value=0.01,
                                    step=0.5)
            eta_c = c1.number_input("η Laden", value=float(cfg.eta_charge), min_value=0.5, max_value=1.0,
                                    step=0.01, format="%.3f")
            eta_d = c2.number_input("η Entladen", value=float(cfg.eta_discharge), min_value=0.5, max_value=1.0,
                                    step=0.01, format="%.3f")
            soc_init = c1.number_input("Start-SoC [MWh]", value=float(cfg.soc_init), min_value=0.0, step=0.25)
            terminal = c2.number_input("Endwert [€/MWh]", value=float(cfg.terminal_soc_value), step=10.0)
            dt = st.selectbox("Zeitauflösung", [0.25, 1.0], index=0 if cfg.dt == 0.25 else 1,
                              format_func=lambda v: "15 min" if v == 0.25 else "1 h")
            simultaneous = st.checkbox("Gleichzeitig laden und entladen erlauben",
                                       value=cfg.allow_simultaneous_charge_discharge)
        with col_p:
            st.markdown("**Preise und Simulation**")
            source = st.radio("Preisquelle", ["synthetisch", "Szenario-Ordner"], horizontal=True)
            c1, c2 = st.columns(2)
            n_paths = c1.number_input("Pfade", value=100, min_value=4, step=50)
            reg_share = c2.slider("Anteil Regression", 0.1, 0.9, 0.5, 0.05)
            n_days = c1.number_input("Tage", value=365, min_value=2, max_value=3 * 366, step=1)
            seed = c2.number_input("Seed", value=42, min_value=0, step=1)
            start = st.date_input("Startdatum", value=date(2027, 1, 1))
            folder = st.text_input("Szenario-Ordner (nur bei Quelle Ordner)", "data/scenarios/")
        with col_n:
            st.markdown("**Numerik**")
            c1, c2 = st.columns(2)
            grid = c1.number_input("SoC-Gitterpunkte", value=int(cfg.soc_grid_size), min_value=2, max_value=41,
                                   step=1)
            degree = c2.number_input("Grad der Basis", value=defaults.degree, min_value=1, max_value=4, step=1)
            features = st.selectbox("Regressions-Features", ["auto", "state", "prices"],
                                    format_func={"auto": "automatisch", "state": "Modellzustand (L, A, B)",
                                                 "prices": "Tagesmittel und -spanne"}.get)
            projection = st.checkbox("Konkavprojektion", value=defaults.concave_projection)
            control_variate = st.checkbox("Control Variate", value=defaults.control_variate)
            value_cache = st.checkbox("Tageswerte cachen", value=defaults.value_cache_dir is not None)
            library = st.checkbox("Vertex-Library speichern", value=defaults.library_dir is not None)
        submitted = st.form_submit_button("Berechnen", type="primary", width="stretch")

    if not submitted:
        return
    battery = dataclasses.replace(
        cfg, name=name.strip() or "bess", e_min=e_min, e_max=e_max, p_charge_max=p_ch,
        p_discharge_max=p_dis, eta_charge=eta_c, eta_discharge=eta_d, soc_init=soc_init,
        terminal_soc_value=terminal, dt=dt, soc_grid_size=int(grid),
        allow_simultaneous_charge_discharge=simultaneous, max_cycles_per_day_avg=None,
    )
    try:
        BatteryConfig.from_dict(battery.to_dict())
    except ValueError as exc:
        st.error(f"Ungültige Batterieparameter: {exc}")
        return
    if source == "synthetisch":
        prices = {"source": "synthetic", "generator_config": GENERATOR_CONFIG, "n_paths": int(n_paths),
                  "n_days": int(n_days), "start": start.isoformat(), "seed": int(seed)}
        n_total = int(n_paths)
    else:
        prices = {"source": "folder", "folder": folder}
        n_total = None
    settings = {"degree": int(degree), "features": features, "concave_projection": projection,
                "control_variate": control_variate,
                "value_cache_dir": "value_cache" if value_cache else None,
                "library_dir": "library_cache" if library else None}
    if n_total:
        settings["n_regression_paths"] = max(1, min(n_total - 1, round(reg_share * n_total)))
    job_dir = submit_job({"battery": battery.to_dict(), "prices": prices, "settings": settings,
                          "results_root": "results"}, JOBS_ROOT)
    st.toast(f"Lauf {job_dir.name} gestartet")


def _jobs_frame(jobs: list[dict]) -> pd.DataFrame:
    rows = []
    for job in jobs:
        spec_path = Path(job["job_dir"]) / "job.json"
        spec = json.loads(spec_path.read_text()) if spec_path.exists() else {}
        summ = job.get("summary") or {}
        prices = spec.get("prices", {})
        rows.append({
            "Name": job.get("name", ""),
            "Status": STATE_LABELS.get(job["state"], job["state"]),
            "Fortschritt": 100.0 * float(job.get("progress") or 0.0),
            "Schritt": job.get("step") if job["state"] != "failed" else (job.get("error") or "")[:80],
            "Laufzeit": _fmt_duration(job.get("elapsed_s")),
            "Restzeit": _fmt_duration(job.get("eta_s")) if job["state"] == "running" else "–",
            "Pfade": prices.get("n_paths"),
            "Tage": prices.get("n_days"),
            "LSMC [k€]": summ.get("value_oos_eur", float("nan")) / 1e3 if summ else None,
            "Intrinsisch [k€]": summ.get("value_intrinsic_eur", float("nan")) / 1e3 if summ else None,
            "Perfect Foresight [k€]": summ.get("value_perfect_foresight_eur", float("nan")) / 1e3 if summ else None,
            "Gestartet": (job.get("started") or job.get("created") or "").replace("T", " "),
            "Ergebnis": f"/result?run={job['id']}" if job["state"] == "done" else None,
        })
    return pd.DataFrame(rows)


@st.fragment(run_every="2s")
def _runs_table() -> None:
    jobs = list_jobs(JOBS_ROOT)
    st.subheader("Läufe")
    if not jobs:
        st.caption("Noch keine Läufe. Parameter oben einstellen und auf „Berechnen“ klicken.")
        return
    st.dataframe(
        _jobs_frame(jobs), hide_index=True, width="stretch",
        column_config={
            "Fortschritt": st.column_config.ProgressColumn("Fortschritt", min_value=0, max_value=100,
                                                           format="%.0f %%"),
            "LSMC [k€]": st.column_config.NumberColumn(format="%.1f"),
            "Intrinsisch [k€]": st.column_config.NumberColumn(format="%.1f"),
            "Perfect Foresight [k€]": st.column_config.NumberColumn(format="%.1f"),
            "Ergebnis": st.column_config.LinkColumn("Ergebnis", display_text="Ergebnis öffnen"),
        },
    )


def runs_page() -> None:
    """Parameter form and table of all runs."""
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("BESS Valuation")
    st.markdown('<div class="app-subtitle">Bewertung und optimaler Einsatz einer Batterie am '
                'Day-Ahead-Markt (LSMC)</div>', unsafe_allow_html=True)
    _run_form()
    _runs_table()


# ------------------------------------------------------------------ page: result


def result_page() -> None:
    """Key figures, parameters, figures and downloads of one finished run."""
    st.markdown(CSS, unsafe_allow_html=True)
    done = [j for j in list_jobs(JOBS_ROOT) if j["state"] == "done"]
    run_id = st.query_params.get("run")
    if not done:
        st.title("Ergebnis")
        st.caption("Noch kein abgeschlossener Lauf.")
        return
    ids = [j["id"] for j in done]
    index = ids.index(run_id) if run_id in ids else 0
    chosen = st.selectbox("Lauf", ids, index=index,
                          format_func=lambda i: f"{next(j['name'] for j in done if j['id'] == i)} – {i}")
    st.query_params["run"] = chosen
    job = read_status(JOBS_ROOT / chosen)
    run_dir = Path(job["run_dir"])
    summary = json.loads((run_dir / "summary.json").read_text())
    spec = json.loads((Path(job["job_dir"]) / "job.json").read_text())

    st.title(summary["name"])
    st.markdown(f'<div class="app-subtitle">{summary["n_paths_regression"]} Regressions- + '
                f'{summary["n_paths_oos"]} Out-of-sample-Pfade · {summary["n_days"]} Tage ab '
                f'{summary["start"]} · {summary["soc_grid_size"]} SoC-Gitterpunkte · Ordner '
                f'<code>{run_dir}</code></div>', unsafe_allow_html=True)
    m = st.columns(5)
    m[0].metric("LSMC out-of-sample", f"{summary['value_oos_eur'] / 1e3:,.1f} k€",
                help=f"± {summary['value_oos_stderr_eur'] / 1e3:.2f} k€ Standardfehler")
    m[1].metric("Intrinsisch", f"{summary['value_intrinsic_eur'] / 1e3:,.1f} k€")
    m[2].metric("Extrinsischer Anteil", f"{100 * summary['extrinsic_share']:.1f} %")
    m[3].metric("Perfect Foresight", f"{summary['value_perfect_foresight_eur'] / 1e3:,.1f} k€")
    m[4].metric("Zyklen pro Tag", f"{summary['cycles_per_day_mean']:.2f}")

    fig_dir = run_dir / "figures_dark"
    if not fig_dir.exists():
        fig_dir = run_dir / "figures"
    figures = sorted(fig_dir.glob("*.png"))
    if figures:
        tabs = st.tabs([FIGURE_TITLES.get(f.stem, f.stem) for f in figures])
        for tab, fig in zip(tabs, figures):
            tab.image(str(fig), width="stretch")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Kennzahlen")
        rows = [(k, v) for k, v in summary.items() if not k.startswith("runtime_")]
        st.dataframe(pd.DataFrame(rows, columns=["Größe", "Wert"]).astype({"Wert": str}),
                     hide_index=True, width="stretch", height=420)
    with right:
        st.subheader("Laufzeiten")
        times = [(k[len("runtime_"):-2], v) for k, v in summary.items() if k.startswith("runtime_")]
        st.dataframe(pd.DataFrame(times, columns=["Schritt", "Sekunden"]), hide_index=True,
                     width="stretch",
                     column_config={"Sekunden": st.column_config.NumberColumn(format="%.1f")})
        st.subheader("Parameter")
        st.json(spec, expanded=False)
        c1, c2 = st.columns(2)
        c1.download_button("summary.csv", (run_dir / "summary.csv").read_bytes(), "summary.csv",
                           width="stretch")
        c2.download_button("paths.parquet", (run_dir / "paths.parquet").read_bytes(), "paths.parquet",
                           width="stretch")


def main() -> None:
    """Page setup and navigation."""
    st.set_page_config(page_title="BESS Valuation", page_icon=":material/battery_charging_full:",
                       layout="wide")
    nav = st.navigation([
        st.Page(runs_page, title="Läufe", icon=":material/table_rows:", url_path="runs", default=True),
        st.Page(result_page, title="Ergebnis", icon=":material/insights:", url_path="result"),
    ])
    nav.run()


if __name__ == "__main__":  # streamlit runs the script as __main__
    main()
