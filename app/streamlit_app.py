"""Streamlit front end of bess_valuation: master data, valuation runs, results.

Start from the repository root:

    poetry install --with app
    poetry run streamlit run app/streamlit_app.py

Pages
* Läufe: choose a stored battery and simulation set plus the numerical settings,
  "Berechnen" starts a background job (bess_valuation.jobs); the table shows all runs
  with progress, remaining time, key values and a link to the result.
* Ergebnis: key figures, figures (dark theme), runtimes, parameters, downloads.
* Batterien / Simulationen: tables of the stored entries (bess_valuation.catalog) with
  forms to add, edit and delete them.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from bess_valuation import BatteryConfig, EngineSettings
from bess_valuation.catalog import (DEFAULT_ROOT, SimulationSpec, delete_entry, list_batteries,
                                    list_simulations, save_battery, save_simulation, seed_defaults)
from bess_valuation.jobs import DEFAULT_JOBS_ROOT, list_jobs, read_status, submit_job

JOBS_ROOT = DEFAULT_JOBS_ROOT
CATALOG = DEFAULT_ROOT
STATE_LABELS = {"queued": "◌ wartet", "running": "● läuft", "done": "✓ fertig", "failed": "✕ Fehler"}
SOLVER_LABELS = {"dp": "DP über SoC (schnell)", "library": "Vertex-Library + Simplex",
                 "cold": "Simplex, Kaltstart", "cluster": "Simplex, Cluster-Warmstart"}
FEATURE_LABELS = {"auto": "automatisch", "state": "Modellzustand (L, A, B)", "prices": "Tagesmittel und -spanne"}
FIGURE_TITLES = {
    "01_value_fan": "Wertverlauf",
    "02_value_decomposition": "Wertzerlegung",
    "03_daily_cashflow": "Tägliche Cashflows",
    "04_soc": "Ladezustand",
    "05_dispatch_profile": "Fahrprofil",
    "09_continuation_value": "Fortsetzungswert",
    "12_vertex_library": "Vertex-Library",
    "13_regression": "Regression (R²)",
}

# Only what the theme config cannot do; selectors kept few (they may change with Streamlit).
CSS = """
<style>
  .block-container {padding-top: 2.4rem; max-width: 1440px;}
  .page-kicker {color: #6da7ec; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.08em;
                text-transform: uppercase; margin-bottom: -0.4rem;}
  .page-subtitle {color: #9aa4b2; margin: -0.5rem 0 1.6rem 0;}
  .facts {color: #9aa4b2; font-size: 0.86rem; line-height: 1.55;}
  .facts b {color: #e6e9ef; font-weight: 500;}
  div[data-testid="stMetric"] {background: #121a27; border: 1px solid #1e2633;
                               border-radius: 10px; padding: 0.9rem 1.1rem;}
  div[data-testid="stMetricLabel"] p {color: #9aa4b2; font-size: 0.8rem;}
</style>
"""


def _header(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(f'<div class="page-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None or seconds != seconds:
        return "–"
    seconds = int(round(seconds))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _fmt_time(stamp: str | None) -> str:
    if not stamp:
        return ""
    return datetime.fromisoformat(stamp).strftime("%d.%m. %H:%M")


def _battery_facts(c: BatteryConfig) -> str:
    return (f'<div class="facts"><b>{c.e_max - c.e_min:g} MWh</b> nutzbar · <b>{c.p_charge_max:g}/'
            f'{c.p_discharge_max:g} MW</b> Laden/Entladen · η <b>{c.eta_charge:g}/{c.eta_discharge:g}</b><br>'
            f'{"15 min" if c.dt == 0.25 else "1 h"} · Start-SoC <b>{c.soc_init:g} MWh</b> · '
            f'<b>{c.soc_grid_size}</b> SoC-Gitterpunkte</div>')


def _simulation_facts(s: SimulationSpec) -> str:
    if s.source == "folder":
        return f'<div class="facts">Szenario-Ordner <b>{s.folder}</b> · {"15 min" if s.dt == 0.25 else "1 h"}</div>'
    n_reg = s.n_regression_paths()
    return (f'<div class="facts"><b>{s.n_paths}</b> Pfade (<b>{n_reg}</b> Regression, '
            f'<b>{s.n_paths - n_reg}</b> out-of-sample) · <b>{s.n_days}</b> Tage ab <b>{s.start}</b><br>'
            f'Seed <b>{s.seed}</b> · {"15 min" if s.dt == 0.25 else "1 h"}</div>')


# ------------------------------------------------------------------ page: runs


def _new_run() -> None:
    batteries = list_batteries(CATALOG)
    simulations = list_simulations(CATALOG)
    defaults = EngineSettings()
    with st.container(border=True):
        st.subheader("Neuer Lauf")
        if not batteries or not simulations:
            st.info("Lege zuerst eine Batterie und eine Simulation an.")
            return
        col_b, col_s, col_n = st.columns([1, 1, 1.2], gap="large")
        with col_b:
            names = [b.name for b in batteries]
            battery = batteries[names.index(st.selectbox("Batterie", names))]
            st.markdown(_battery_facts(battery), unsafe_allow_html=True)
        with col_s:
            snames = [s.name for s in simulations]
            sim = simulations[snames.index(st.selectbox("Simulation", snames))]
            st.markdown(_simulation_facts(sim), unsafe_allow_html=True)
        with col_n:
            c1, c2 = st.columns(2)
            solver = c1.selectbox("Tageslöser", list(SOLVER_LABELS), format_func=SOLVER_LABELS.get)
            degree = c2.number_input("Grad der Basis", value=defaults.degree, min_value=1, max_value=4)
            features = c1.selectbox("Regressions-Features", list(FEATURE_LABELS), format_func=FEATURE_LABELS.get)
            grid = c2.number_input("SoC-Gitterpunkte", value=int(battery.soc_grid_size), min_value=2,
                                   max_value=41, help="überschreibt den Wert der Batterie für diesen Lauf")
            projection = c1.toggle("Konkavprojektion", value=defaults.concave_projection)
            control_variate = c2.toggle("Control Variate", value=defaults.control_variate)
            cache = c1.toggle("Tageswerte cachen", value=True)
        if st.button("Berechnen", type="primary", icon=":material/play_arrow:"):
            if battery.dt != sim.dt:
                st.error(f"Zeitauflösung passt nicht: Batterie {battery.dt} h, Simulation {sim.dt} h.")
                return
            settings = {"degree": int(degree), "features": features, "concave_projection": projection,
                        "control_variate": control_variate, "daily_solver": solver,
                        "value_cache_dir": "value_cache" if cache else None,
                        "library_dir": "library_cache" if solver == "library" else None}
            if sim.source == "synthetic":
                settings["n_regression_paths"] = sim.n_regression_paths()
            cfg = dataclasses.replace(battery, soc_grid_size=int(grid))
            job_dir = submit_job({"battery": cfg.to_dict(), "prices": sim.prices_spec(), "settings": settings,
                                  "results_root": "results", "battery_name": battery.name,
                                  "simulation_name": sim.name}, JOBS_ROOT)
            st.toast(f"Lauf {job_dir.name} gestartet", icon=":material/rocket_launch:")


def _jobs_frame(jobs: list[dict]) -> pd.DataFrame:
    rows = []
    for job in jobs:
        spec_path = Path(job["job_dir"]) / "job.json"
        spec = json.loads(spec_path.read_text()) if spec_path.exists() else {}
        summ = job.get("summary") or {}

        def k(key: str) -> str:
            return f"{summ[key] / 1e3:,.1f}" if key in summ else "–"

        rows.append({
            "Status": STATE_LABELS.get(job["state"], job["state"]),
            "Batterie": spec.get("battery_name", job.get("name", "")),
            "Simulation": spec.get("simulation_name", ""),
            "Fortschritt": 100.0 * float(job.get("progress") or 0.0),
            "Schritt": job.get("step") if job["state"] != "failed" else (job.get("error") or "")[:80],
            "Laufzeit": _fmt_duration(job.get("elapsed_s")),
            "Restzeit": _fmt_duration(job.get("eta_s")) if job["state"] == "running" else "–",
            "LSMC [k€]": k("value_oos_eur"),
            "Intrinsisch [k€]": k("value_intrinsic_eur"),
            "Perfect Foresight [k€]": k("value_perfect_foresight_eur"),
            "Löser": spec.get("settings", {}).get("daily_solver", ""),
            "Gestartet": _fmt_time(job.get("started") or job.get("created")),
            "Ergebnis": f"/result?run={job['id']}" if job["state"] == "done" else None,
        })
    return pd.DataFrame(rows)


@st.fragment(run_every="2s")
def _runs_table() -> None:
    jobs = list_jobs(JOBS_ROOT)
    st.subheader("Läufe")
    if not jobs:
        st.caption("Noch keine Läufe – Batterie und Simulation wählen und auf „Berechnen“ klicken.")
        return
    st.dataframe(
        _jobs_frame(jobs), hide_index=True, width="stretch",
        column_config={
            "Fortschritt": st.column_config.ProgressColumn("Fortschritt", min_value=0, max_value=100,
                                                           format="%.0f %%"),
            "Ergebnis": st.column_config.LinkColumn("Ergebnis", display_text="öffnen →"),
        },
    )


def runs_page() -> None:
    """Combine a battery and a simulation set into a run; table of all runs."""
    seed_defaults(CATALOG)
    _header("Bewertung", "Läufe", "Batterie und Simulation wählen, Numerik einstellen, Bewertung starten. "
            "Läufe rechnen im Hintergrund weiter, auch wenn der Browser geschlossen wird.")
    _new_run()
    _runs_table()


# ------------------------------------------------------------------ page: batteries


def batteries_page() -> None:
    """Table of stored battery configurations; add, edit, delete."""
    seed_defaults(CATALOG)
    _header("Stammdaten", "Batterien", "Gespeicherte Batteriekonfigurationen. Neue anlegen oder "
            "bestehende bearbeiten; der Name ist der Schlüssel.")
    batteries = list_batteries(CATALOG)
    st.dataframe(pd.DataFrame([{
        "Name": c.name, "Kapazität [MWh]": c.e_max - c.e_min, "e_min [MWh]": c.e_min, "e_max [MWh]": c.e_max,
        "Laden [MW]": c.p_charge_max, "Entladen [MW]": c.p_discharge_max, "η Laden": c.eta_charge,
        "η Entladen": c.eta_discharge, "Auflösung": "15 min" if c.dt == 0.25 else "1 h",
        "Start-SoC [MWh]": c.soc_init, "SoC-Gitter": c.soc_grid_size, "Endwert [€/MWh]": c.terminal_soc_value,
        "Gleichzeitig laden/entladen": c.allow_simultaneous_charge_discharge,
    } for c in batteries]), hide_index=True, width="stretch")

    names = ["➕ Neue Batterie"] + [c.name for c in batteries]
    choice = st.selectbox("Bearbeiten", names)
    cfg = batteries[names.index(choice) - 1] if choice != names[0] else BatteryConfig(
        name="neue_batterie", e_max=2.0, p_charge_max=1.0, p_discharge_max=1.0, soc_init=1.0, soc_grid_size=9)
    with st.form(f"battery_{choice}", border=True):
        c1, c2, c3, c4 = st.columns(4)
        name = c1.text_input("Name", cfg.name)
        e_min = c2.number_input("e_min [MWh]", value=float(cfg.e_min), min_value=0.0, step=0.5)
        e_max = c3.number_input("e_max [MWh]", value=float(cfg.e_max), min_value=0.1, step=0.5)
        soc_init = c4.number_input("Start-SoC [MWh]", value=float(cfg.soc_init), min_value=0.0, step=0.25)
        p_ch = c1.number_input("Ladeleistung [MW]", value=float(cfg.p_charge_max), min_value=0.01, step=0.5)
        p_dis = c2.number_input("Entladeleistung [MW]", value=float(cfg.p_discharge_max), min_value=0.01, step=0.5)
        eta_c = c3.number_input("η Laden", value=float(cfg.eta_charge), min_value=0.5, max_value=1.0,
                                step=0.01, format="%.3f")
        eta_d = c4.number_input("η Entladen", value=float(cfg.eta_discharge), min_value=0.5, max_value=1.0,
                                step=0.01, format="%.3f")
        dt = c1.selectbox("Zeitauflösung", [0.25, 1.0], index=0 if cfg.dt == 0.25 else 1,
                          format_func=lambda v: "15 min" if v == 0.25 else "1 h")
        grid = c2.number_input("SoC-Gitterpunkte", value=int(cfg.soc_grid_size), min_value=2, max_value=41)
        terminal = c3.number_input("Endwert [€/MWh]", value=float(cfg.terminal_soc_value), step=10.0)
        simultaneous = c4.checkbox("Gleichzeitig laden/entladen", value=cfg.allow_simultaneous_charge_discharge)
        saved = st.form_submit_button("Speichern", type="primary", icon=":material/save:")
    if saved:
        try:
            new = dataclasses.replace(cfg, name=name.strip(), e_min=e_min, e_max=e_max, soc_init=soc_init,
                                      p_charge_max=p_ch, p_discharge_max=p_dis, eta_charge=eta_c,
                                      eta_discharge=eta_d, dt=dt, soc_grid_size=int(grid),
                                      terminal_soc_value=terminal,
                                      allow_simultaneous_charge_discharge=simultaneous,
                                      max_cycles_per_day_avg=None)
            save_battery(new, CATALOG)
            if choice != names[0] and new.name != cfg.name:
                delete_entry("batteries", cfg.name, CATALOG)
            st.toast(f"Batterie {new.name} gespeichert", icon=":material/check:")
            st.rerun()
        except ValueError as exc:
            st.error(f"Ungültige Parameter: {exc}")
    if choice != names[0] and st.button("Löschen", icon=":material/delete:"):
        delete_entry("batteries", cfg.name, CATALOG)
        st.rerun()


# ------------------------------------------------------------------ page: simulations


def simulations_page() -> None:
    """Table of stored simulation sets; add, edit, delete."""
    seed_defaults(CATALOG)
    _header("Stammdaten", "Simulationen", "Gespeicherte Preisszenarien. Synthetische Sets sind über den "
            "Seed reproduzierbar: gleicher Seed, gleiche Pfade (und Treffer im Tageswerte-Cache).")
    sims = list_simulations(CATALOG)
    st.dataframe(pd.DataFrame([{
        "Name": s.name, "Quelle": "synthetisch" if s.source == "synthetic" else "Ordner",
        "Pfade": s.n_paths if s.source == "synthetic" else None,
        "davon Regression": s.n_regression_paths() if s.source == "synthetic" else None,
        "Tage": s.n_days if s.source == "synthetic" else None, "Start": s.start if s.source == "synthetic" else "",
        "Seed": s.seed if s.source == "synthetic" else None, "Auflösung": "15 min" if s.dt == 0.25 else "1 h",
        "Ordner": s.folder, "Beschreibung": s.description,
    } for s in sims]), hide_index=True, width="stretch")

    names = ["➕ Neue Simulation"] + [s.name for s in sims]
    choice = st.selectbox("Bearbeiten", names)
    spec = sims[names.index(choice) - 1] if choice != names[0] else SimulationSpec(name="neue_simulation")
    with st.form(f"simulation_{choice}", border=True):
        c1, c2, c3, c4 = st.columns(4)
        name = c1.text_input("Name", spec.name)
        source = c2.selectbox("Quelle", ["synthetic", "folder"], index=0 if spec.source == "synthetic" else 1,
                              format_func={"synthetic": "synthetisch (Generator)", "folder": "Szenario-Ordner"}.get)
        dt = c3.selectbox("Zeitauflösung", [0.25, 1.0], index=0 if spec.dt == 0.25 else 1,
                          format_func=lambda v: "15 min" if v == 0.25 else "1 h")
        share = c4.slider("Anteil Regression", 0.1, 0.9, float(spec.regression_share), 0.05)
        n_paths = c1.number_input("Pfade", value=int(spec.n_paths), min_value=4, step=50)
        n_days = c2.number_input("Tage", value=int(spec.n_days), min_value=2, max_value=3 * 366)
        start = c3.date_input("Startdatum", value=date.fromisoformat(spec.start))
        seed = c4.number_input("Seed", value=int(spec.seed), min_value=0)
        generator = c1.text_input("Generator-Konfiguration", spec.generator_config)
        folder = c2.text_input("Szenario-Ordner", spec.folder)
        description = st.text_input("Beschreibung", spec.description)
        saved = st.form_submit_button("Speichern", type="primary", icon=":material/save:")
    if saved:
        try:
            new = SimulationSpec(name=name.strip(), source=source, n_paths=int(n_paths), n_days=int(n_days),
                                 start=start.isoformat(), seed=int(seed), dt=dt, regression_share=share,
                                 generator_config=generator, folder=folder, description=description)
            save_simulation(new, CATALOG)
            if choice != names[0] and new.name != spec.name:
                delete_entry("simulations", spec.name, CATALOG)
            st.toast(f"Simulation {new.name} gespeichert", icon=":material/check:")
            st.rerun()
        except ValueError as exc:
            st.error(f"Ungültige Parameter: {exc}")
    if choice != names[0] and st.button("Löschen", icon=":material/delete:"):
        delete_entry("simulations", spec.name, CATALOG)
        st.rerun()


# ------------------------------------------------------------------ page: result


def result_page() -> None:
    """Key figures, figures, runtimes, parameters and downloads of one finished run."""
    done = [j for j in list_jobs(JOBS_ROOT) if j["state"] == "done"]
    if not done:
        _header("Bewertung", "Ergebnis", "Noch kein abgeschlossener Lauf.")
        return
    ids = [j["id"] for j in done]
    run_id = st.query_params.get("run")
    labels = {j["id"]: f"{j['name']} · {j['id']}" for j in done}
    chosen = st.selectbox("Lauf", ids, index=ids.index(run_id) if run_id in ids else 0,
                          format_func=labels.get)
    st.query_params["run"] = chosen
    job = read_status(JOBS_ROOT / chosen)
    run_dir = Path(job["run_dir"])
    summary = json.loads((run_dir / "summary.json").read_text())
    spec = json.loads((Path(job["job_dir"]) / "job.json").read_text())

    _header("Ergebnis", summary["name"],
            f'{spec.get("simulation_name", "")} · {summary["n_paths_regression"]} Regressions- + '
            f'{summary["n_paths_oos"]} Out-of-sample-Pfade · {summary["n_days"]} Tage ab {summary["start"]} · '
            f'{summary["soc_grid_size"]} SoC-Gitterpunkte · Löser {summary.get("daily_solver", "")}')
    m = st.columns(6)
    m[0].metric("LSMC", f"{summary['value_oos_eur'] / 1e3:,.1f} k€",
                help=f"Out-of-sample-Wert der Strategie, ± {summary['value_oos_stderr_eur'] / 1e3:.2f} k€ "
                     f"Standardfehler; in-sample {summary['value_in_sample_eur'] / 1e3:,.1f} k€")
    m[1].metric("Intrinsisch", f"{summary['value_intrinsic_eur'] / 1e3:,.1f} k€",
                help="optimaler Fahrplan auf dem mittleren Preispfad")
    m[2].metric("Extrinsisch", f"{summary['value_extrinsic_eur'] / 1e3:,.1f} k€",
                f"{100 * summary['extrinsic_share']:.1f} % des Werts", delta_color="off")
    m[3].metric("Perfect Foresight", f"{summary['value_perfect_foresight_eur'] / 1e3:,.1f} k€")
    m[4].metric("Zyklen pro Tag", f"{summary['cycles_per_day_mean']:.2f}")
    m[5].metric("R² Regression (Median)", f"{summary.get('regression_r2_median', float('nan')):.2f}")

    fig_dir = run_dir / "figures_dark"
    if not fig_dir.exists():
        fig_dir = run_dir / "figures"
    figures = sorted(fig_dir.glob("*.png"))
    if figures:
        tabs = st.tabs([FIGURE_TITLES.get(f.stem, f.stem) for f in figures])
        for tab, fig in zip(tabs, figures):
            tab.image(str(fig), width="stretch")

    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.subheader("Kennzahlen")
        rows = [(k, v) for k, v in summary.items() if not k.startswith("runtime_")]
        st.dataframe(pd.DataFrame(rows, columns=["Größe", "Wert"]).astype({"Wert": str}),
                     hide_index=True, width="stretch", height=420)
    with right:
        st.subheader("Laufzeiten")
        times = [(k[len("runtime_"):-2], v) for k, v in summary.items() if k.startswith("runtime_")]
        st.dataframe(pd.DataFrame(times, columns=["Schritt", "Sekunden"]), hide_index=True, width="stretch",
                     column_config={"Sekunden": st.column_config.NumberColumn(format="%.1f")})
        st.subheader("Downloads")
        files = [f for f in ("summary.csv", "daily.parquet", "regression.parquet", "paths.parquet")
                 if (run_dir / f).exists()]
        cols = st.columns(2)
        for k, f in enumerate(files):
            cols[k % 2].download_button(f, (run_dir / f).read_bytes(), f, width="stretch",
                                        icon=":material/download:")
        with st.expander("Parameter des Laufs"):
            st.json(spec)


def main() -> None:
    """Page setup and navigation."""
    st.set_page_config(page_title="BESS Valuation", page_icon=":material/battery_charging_full:",
                       layout="wide")
    st.logo(str(Path(__file__).with_name("logo.svg")), size="large")
    with st.sidebar:
        st.caption("Bewertung einer Batterie am Day-Ahead-Markt (LSMC)")
    nav = st.navigation({
        "Bewertung": [
            st.Page(runs_page, title="Läufe", icon=":material/play_circle:", url_path="runs", default=True),
            st.Page(result_page, title="Ergebnis", icon=":material/insights:", url_path="result"),
        ],
        "Stammdaten": [
            st.Page(batteries_page, title="Batterien", icon=":material/battery_full:", url_path="batteries"),
            st.Page(simulations_page, title="Simulationen", icon=":material/stacked_line_chart:",
                    url_path="simulations"),
        ],
    })
    nav.run()


if __name__ == "__main__":  # streamlit runs the script as __main__
    main()
