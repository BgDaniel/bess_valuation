"""Streamlit front end of bess_valuation: master data, valuation runs, results.

Start from the repository root:

    poetry install --with app
    poetry run streamlit run app/streamlit_app.py

Pages
* Runs: choose a stored battery and simulation set plus the numerical settings,
  "Run valuation" starts a background job (bess_valuation.jobs); the table shows all runs
  with progress, remaining time, key values and a link to the result.
* Result: key figures, figures (dark theme), runtimes, parameters, downloads.
* Batteries / Simulations: tables of the stored entries (bess_valuation.catalog) with
  forms to add, edit and delete them.
* Settings: display units (currency label, MWh or kWh). Computations always run in MW / MWh;
  forms convert on save, tables and figures convert for display.
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
                                    list_simulations, load_units, save_battery, save_simulation,
                                    save_units, seed_defaults)
from bess_valuation.jobs import DEFAULT_JOBS_ROOT, list_jobs, read_status, submit_job
from bess_valuation.units import CURRENCY_SYMBOLS, ENERGY_FACTORS, Units

JOBS_ROOT = DEFAULT_JOBS_ROOT
CATALOG = DEFAULT_ROOT
STATE_LABELS = {"queued": "◌ queued", "running": "● running", "done": "✓ done", "failed": "✕ failed"}
SOLVER_LABELS = {"dp": "DP over SoC (fast)", "library": "Vertex library + simplex",
                 "cold": "Simplex, cold start", "cluster": "Simplex, cluster warm start"}
FEATURE_LABELS = {"auto": "automatic", "state": "model state (L, A, B)", "prices": "daily mean and spread"}
FIGURE_TITLES = {
    "01_value_fan": "Value over time",
    "02_value_decomposition": "Value decomposition",
    "03_daily_cashflow": "Daily cashflows",
    "04_soc": "State of charge",
    "05_dispatch_profile": "Dispatch profile",
    "09_continuation_value": "Continuation value",
    "12_vertex_library": "Vertex library",
    "13_regression": "Regression (R²)",
    "15_convexity": "Convexity",
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


def _money(value: float, u: Units) -> str:
    """Amount in thousands, e.g. "86.9 k€"."""
    return f"{value / 1e3:,.1f} k{u.symbol}"


def _battery_facts(c: BatteryConfig, u: Units) -> str:
    return (f'<div class="facts"><b>{u.energy_value(c.e_max - c.e_min):g} {u.energy}</b> usable · '
            f'<b>{u.power_value(c.p_charge_max):g}/{u.power_value(c.p_discharge_max):g} {u.power}</b> '
            f'charge/discharge · η <b>{c.eta_charge:g}/{c.eta_discharge:g}</b><br>'
            f'{"15 min" if c.dt == 0.25 else "1 h"} · start SoC <b>{u.energy_value(c.soc_init):g} {u.energy}</b> · '
            f'<b>{c.soc_grid_size}</b> SoC grid points<br>'
            f'capacity fade <b>{100 * c.capacity_fade_per_year:g} %/yr</b> · wear cost '
            f'<b>{u.price_value(c.wear_cost_per_mwh):g} {u.price}</b></div>')


def _simulation_facts(s: SimulationSpec) -> str:
    if s.source == "folder":
        return f'<div class="facts">Scenario folder <b>{s.folder}</b> · {"15 min" if s.dt == 0.25 else "1 h"}</div>'
    n_reg = s.n_regression_paths()
    return (f'<div class="facts"><b>{s.n_paths}</b> paths (<b>{n_reg}</b> regression, '
            f'<b>{s.n_paths - n_reg}</b> out-of-sample) · <b>{s.n_days}</b> days from <b>{s.start}</b><br>'
            f'seed <b>{s.seed}</b> · {"15 min" if s.dt == 0.25 else "1 h"}</div>')


# ------------------------------------------------------------------ page: runs


def _new_run(u: Units) -> None:
    batteries = list_batteries(CATALOG)
    simulations = list_simulations(CATALOG)
    defaults = EngineSettings()
    with st.container(border=True):
        st.subheader("New run")
        if not batteries or not simulations:
            st.info("Create a battery and a simulation set first.")
            return
        col_b, col_s, col_n = st.columns([1, 1, 1.2], gap="large")
        with col_b:
            names = [b.name for b in batteries]
            battery = batteries[names.index(st.selectbox("Battery", names))]
            st.markdown(_battery_facts(battery, u), unsafe_allow_html=True)
        with col_s:
            snames = [s.name for s in simulations]
            sim = simulations[snames.index(st.selectbox("Simulation", snames))]
            st.markdown(_simulation_facts(sim), unsafe_allow_html=True)
        with col_n:
            c1, c2 = st.columns(2)
            solver = c1.selectbox("Daily solver", list(SOLVER_LABELS), format_func=SOLVER_LABELS.get)
            degree = c2.number_input("Basis degree", value=defaults.degree, min_value=1, max_value=4)
            features = c1.selectbox("Regression features", list(FEATURE_LABELS), format_func=FEATURE_LABELS.get)
            grid = c2.number_input("SoC grid points", value=int(battery.soc_grid_size), min_value=2,
                                   max_value=41, help="overrides the battery's value for this run")
            projection = c1.toggle("Concave projection", value=defaults.concave_projection)
            control_variate = c2.toggle("Control variate", value=defaults.control_variate)
            cache = c1.toggle("Cache daily values", value=True)
        if st.button("Run valuation", type="primary", icon=":material/play_arrow:"):
            if battery.dt != sim.dt:
                st.error(f"Time resolution mismatch: battery {battery.dt} h, simulation {sim.dt} h.")
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
                                  "simulation_name": sim.name, "units": u.to_dict()}, JOBS_ROOT)
            st.toast(f"Run {job_dir.name} started", icon=":material/rocket_launch:")


def _jobs_frame(jobs: list[dict], u: Units) -> pd.DataFrame:
    rows = []
    for job in jobs:
        spec_path = Path(job["job_dir"]) / "job.json"
        spec = json.loads(spec_path.read_text()) if spec_path.exists() else {}
        summ = job.get("summary") or {}

        def k(key: str) -> str:
            return f"{summ[key] / 1e3:,.1f}" if key in summ else "–"

        rows.append({
            "Status": STATE_LABELS.get(job["state"], job["state"]),
            "Battery": spec.get("battery_name", job.get("name", "")),
            "Simulation": spec.get("simulation_name", ""),
            "Progress": 100.0 * float(job.get("progress") or 0.0),
            "Step": job.get("step") if job["state"] != "failed" else (job.get("error") or "")[:80],
            "Elapsed": _fmt_duration(job.get("elapsed_s")),
            "Remaining": _fmt_duration(job.get("eta_s")) if job["state"] == "running" else "–",
            f"LSMC [k{u.symbol}]": k("value_oos_eur"),
            f"Intrinsic [k{u.symbol}]": k("value_intrinsic_eur"),
            f"Perfect Foresight [k{u.symbol}]": k("value_perfect_foresight_eur"),
            "Solver": spec.get("settings", {}).get("daily_solver", ""),
            "Started": _fmt_time(job.get("started") or job.get("created")),
            "Result": f"/result?run={job['id']}" if job["state"] == "done" else None,
        })
    return pd.DataFrame(rows)


@st.fragment(run_every="2s")
def _runs_table() -> None:
    jobs = list_jobs(JOBS_ROOT)
    st.subheader("Runs")
    if not jobs:
        st.caption("No runs yet – choose a battery and a simulation set and click “Run valuation”.")
        return
    st.dataframe(
        _jobs_frame(jobs, load_units(CATALOG)), hide_index=True, width="stretch",
        column_config={
            "Progress": st.column_config.ProgressColumn("Progress", min_value=0, max_value=100,
                                                           format="%.0f %%"),
            "Result": st.column_config.LinkColumn("Result", display_text="open →"),
        },
    )


def runs_page() -> None:
    """Combine a battery and a simulation set into a run; table of all runs."""
    seed_defaults(CATALOG)
    _header("Valuation", "Runs", "Pick a battery and a simulation set, set the numerics, start the valuation. "
            "Runs keep computing in the background, also when the browser is closed.")
    _new_run(load_units(CATALOG))
    _runs_table()


# ------------------------------------------------------------------ page: batteries


def batteries_page() -> None:
    """Table of stored battery configurations; add, edit, delete."""
    seed_defaults(CATALOG)
    _header("Master data", "Batteries", "Stored battery configurations. Add new ones or edit existing "
            "ones; the name is the key.")
    batteries = list_batteries(CATALOG)
    u = load_units(CATALOG)
    E, P, C = f"[{u.energy}]", f"[{u.power}]", f"[{u.price}]"
    st.caption(f"Units: energy {u.energy}, power {u.power}, prices {u.price} (change on the Settings page).")
    st.dataframe(pd.DataFrame([{
        "Name": c.name, f"Capacity {E}": u.energy_value(c.e_max - c.e_min), f"e_min {E}": u.energy_value(c.e_min),
        f"e_max {E}": u.energy_value(c.e_max), f"Charge {P}": u.power_value(c.p_charge_max),
        f"Discharge {P}": u.power_value(c.p_discharge_max), "η charge": c.eta_charge,
        "η discharge": c.eta_discharge, "Resolution": "15 min" if c.dt == 0.25 else "1 h",
        f"Start SoC {E}": u.energy_value(c.soc_init), "SoC grid": c.soc_grid_size,
        f"Terminal value {C}": u.price_value(c.terminal_soc_value),
        "Fade [%/yr]": 100 * c.capacity_fade_per_year, f"Wear cost {C}": u.price_value(c.wear_cost_per_mwh),
        "Simultaneous charge/discharge": c.allow_simultaneous_charge_discharge,
    } for c in batteries]), hide_index=True, width="stretch")

    names = ["➕ New battery"] + [c.name for c in batteries]
    choice = st.selectbox("Edit", names)
    cfg = batteries[names.index(choice) - 1] if choice != names[0] else BatteryConfig(
        name="new_battery", e_max=2.0, p_charge_max=1.0, p_discharge_max=1.0, soc_init=1.0, soc_grid_size=9)
    f = u.energy_factor
    with st.form(f"battery_{choice}_{u.energy}", border=True):
        c1, c2, c3, c4 = st.columns(4)
        name = c1.text_input("Name", cfg.name)
        e_min = c2.number_input(f"e_min {E}", value=u.energy_value(float(cfg.e_min)), min_value=0.0, step=0.5 * f)
        e_max = c3.number_input(f"e_max {E}", value=u.energy_value(float(cfg.e_max)), min_value=0.1 * f,
                                step=0.5 * f)
        soc_init = c4.number_input(f"Start SoC {E}", value=u.energy_value(float(cfg.soc_init)), min_value=0.0,
                                   step=0.25 * f)
        p_ch = c1.number_input(f"Charge power {P}", value=u.power_value(float(cfg.p_charge_max)),
                               min_value=0.01 * f, step=0.5 * f)
        p_dis = c2.number_input(f"Discharge power {P}", value=u.power_value(float(cfg.p_discharge_max)),
                                min_value=0.01 * f, step=0.5 * f)
        eta_c = c3.number_input("η charge", value=float(cfg.eta_charge), min_value=0.5, max_value=1.0,
                                step=0.01, format="%.3f")
        eta_d = c4.number_input("η discharge", value=float(cfg.eta_discharge), min_value=0.5, max_value=1.0,
                                step=0.01, format="%.3f")
        dt = c1.selectbox("Time resolution", [0.25, 1.0], index=0 if cfg.dt == 0.25 else 1,
                          format_func=lambda v: "15 min" if v == 0.25 else "1 h")
        grid = c2.number_input("SoC grid points", value=int(cfg.soc_grid_size), min_value=2, max_value=41)
        terminal = c3.number_input(f"Terminal value {C}", value=u.price_value(float(cfg.terminal_soc_value)),
                                   step=10.0 / f, format="%.4g")
        simultaneous = c4.checkbox("Simultaneous charge/discharge", value=cfg.allow_simultaneous_charge_discharge)
        fade = c1.number_input("Capacity fade [%/yr]", value=100 * float(cfg.capacity_fade_per_year), min_value=0.0,
                               max_value=99.0, step=0.5,
                               help="linear loss of usable capacity, applied in monthly steps")
        wear = c2.number_input(f"Wear cost {C}", value=u.price_value(float(cfg.wear_cost_per_mwh)), min_value=0.0,
                               step=1.0 / f, format="%.4g",
                               help=f"degradation cost per discharged {u.energy}, charged in the optimisation")
        saved = st.form_submit_button("Save", type="primary", icon=":material/save:")
    if saved:
        try:
            new = dataclasses.replace(cfg, name=name.strip(), e_min=e_min / f, e_max=e_max / f,
                                      soc_init=soc_init / f, p_charge_max=p_ch / f, p_discharge_max=p_dis / f,
                                      eta_charge=eta_c, eta_discharge=eta_d, dt=dt, soc_grid_size=int(grid),
                                      terminal_soc_value=terminal * f,
                                      allow_simultaneous_charge_discharge=simultaneous,
                                      max_cycles_per_day_avg=None, capacity_fade_per_year=fade / 100.0,
                                      wear_cost_per_mwh=wear * f)
            save_battery(new, CATALOG)
            if choice != names[0] and new.name != cfg.name:
                delete_entry("batteries", cfg.name, CATALOG)
            st.toast(f"Battery {new.name} saved", icon=":material/check:")
            st.rerun()
        except ValueError as exc:
            st.error(f"Invalid parameters: {exc}")
    if choice != names[0] and st.button("Delete", icon=":material/delete:"):
        delete_entry("batteries", cfg.name, CATALOG)
        st.rerun()


# ------------------------------------------------------------------ page: simulations


def simulations_page() -> None:
    """Table of stored simulation sets; add, edit, delete."""
    seed_defaults(CATALOG)
    _header("Master data", "Simulations", "Stored price scenario sets. Synthetic sets are reproducible via "
            "the seed: same seed, same paths (and hits in the daily-value cache).")
    sims = list_simulations(CATALOG)
    st.dataframe(pd.DataFrame([{
        "Name": s.name, "Source": "synthetic" if s.source == "synthetic" else "folder",
        "Paths": s.n_paths if s.source == "synthetic" else None,
        "Regression": s.n_regression_paths() if s.source == "synthetic" else None,
        "Days": s.n_days if s.source == "synthetic" else None, "Start": s.start if s.source == "synthetic" else "",
        "Seed": s.seed if s.source == "synthetic" else None, "Resolution": "15 min" if s.dt == 0.25 else "1 h",
        "Folder": s.folder, "Description": s.description,
    } for s in sims]), hide_index=True, width="stretch")

    names = ["➕ New simulation"] + [s.name for s in sims]
    choice = st.selectbox("Edit", names)
    spec = sims[names.index(choice) - 1] if choice != names[0] else SimulationSpec(name="new_simulation")
    with st.form(f"simulation_{choice}", border=True):
        c1, c2, c3, c4 = st.columns(4)
        name = c1.text_input("Name", spec.name)
        source = c2.selectbox("Source", ["synthetic", "folder"], index=0 if spec.source == "synthetic" else 1,
                              format_func={"synthetic": "synthetic (generator)", "folder": "scenario folder"}.get)
        dt = c3.selectbox("Time resolution", [0.25, 1.0], index=0 if spec.dt == 0.25 else 1,
                          format_func=lambda v: "15 min" if v == 0.25 else "1 h")
        share = c4.slider("Regression share", 0.1, 0.9, float(spec.regression_share), 0.05)
        n_paths = c1.number_input("Paths", value=int(spec.n_paths), min_value=4, step=50)
        n_days = c2.number_input("Days", value=int(spec.n_days), min_value=2, max_value=3 * 366)
        start = c3.date_input("Start date", value=date.fromisoformat(spec.start))
        seed = c4.number_input("Seed", value=int(spec.seed), min_value=0)
        generator = c1.text_input("Generator config", spec.generator_config)
        folder = c2.text_input("Scenario folder", spec.folder)
        description = st.text_input("Description", spec.description)
        saved = st.form_submit_button("Save", type="primary", icon=":material/save:")
    if saved:
        try:
            new = SimulationSpec(name=name.strip(), source=source, n_paths=int(n_paths), n_days=int(n_days),
                                 start=start.isoformat(), seed=int(seed), dt=dt, regression_share=share,
                                 generator_config=generator, folder=folder, description=description)
            save_simulation(new, CATALOG)
            if choice != names[0] and new.name != spec.name:
                delete_entry("simulations", spec.name, CATALOG)
            st.toast(f"Simulation {new.name} saved", icon=":material/check:")
            st.rerun()
        except ValueError as exc:
            st.error(f"Invalid parameters: {exc}")
    if choice != names[0] and st.button("Delete", icon=":material/delete:"):
        delete_entry("simulations", spec.name, CATALOG)
        st.rerun()


# ------------------------------------------------------------------ page: settings


def settings_page() -> None:
    """Display units: currency label and energy unit."""
    _header("App", "Settings", "Display units for forms, tables and figures. The valuation itself always "
            "runs in MW / MWh; the currency is a label for the price data, amounts are not converted.")
    u = load_units(CATALOG)
    codes = list(CURRENCY_SYMBOLS)
    with st.form("units", border=True):
        c1, c2 = st.columns(2)
        currency = c1.selectbox("Currency", codes + ["other"],
                                index=codes.index(u.currency) if u.currency in codes else len(codes),
                                format_func=lambda c: f"{c} ({CURRENCY_SYMBOLS[c]})" if c in CURRENCY_SYMBOLS
                                else "other …")
        other = c1.text_input("Other currency code", "" if u.currency in codes else u.currency,
                              help="used if “other” is selected")
        energy = c2.radio("Energy unit", list(ENERGY_FACTORS), index=list(ENERGY_FACTORS).index(u.energy),
                          horizontal=True, help="power follows (MW / kW), prices are per energy unit")
        saved = st.form_submit_button("Save", type="primary", icon=":material/save:")
    if saved:
        try:
            save_units(Units(currency=other.strip().upper() if currency == "other" else currency,
                             energy=energy), CATALOG)
            st.toast("Units saved", icon=":material/check:")
            st.rerun()
        except ValueError as exc:
            st.error(f"Invalid units: {exc}")
    st.markdown(f'<div class="facts">Current: amounts in <b>{u.symbol}</b> · energy <b>{u.energy}</b> · '
                f'power <b>{u.power}</b> · prices <b>{u.price}</b><br>New runs store these units; '
                f'their figures and result page use them.</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ page: result


def result_page() -> None:
    """Key figures, figures, runtimes, parameters and downloads of one finished run."""
    done = [j for j in list_jobs(JOBS_ROOT) if j["state"] == "done"]
    if not done:
        _header("Valuation", "Result", "No finished run yet.")
        return
    ids = [j["id"] for j in done]
    run_id = st.query_params.get("run")
    labels = {j["id"]: f"{j['name']} · {j['id']}" for j in done}
    chosen = st.selectbox("Run", ids, index=ids.index(run_id) if run_id in ids else 0,
                          format_func=labels.get)
    st.query_params["run"] = chosen
    job = read_status(JOBS_ROOT / chosen)
    run_dir = Path(job["run_dir"])
    summary = json.loads((run_dir / "summary.json").read_text())
    spec = json.loads((Path(job["job_dir"]) / "job.json").read_text())
    u = Units.from_dict(spec["units"]) if "units" in spec else load_units(CATALOG)

    _header("Result", summary["name"],
            f'{spec.get("simulation_name", "")} · {summary["n_paths_regression"]} regression + '
            f'{summary["n_paths_oos"]} out-of-sample paths · {summary["n_days"]} days from {summary["start"]} · '
            f'{summary["soc_grid_size"]} SoC grid points · solver {summary.get("daily_solver", "")}')
    m = st.columns(6)
    m[0].metric("LSMC", _money(summary["value_oos_eur"], u),
                help=f"Out-of-sample value of the policy, ± {_money(summary['value_oos_stderr_eur'], u)} "
                     f"standard error; in-sample {_money(summary['value_in_sample_eur'], u)}")
    m[1].metric("Intrinsic", _money(summary["value_intrinsic_eur"], u),
                help="optimal schedule on the mean price path")
    m[2].metric("Extrinsic", _money(summary["value_extrinsic_eur"], u),
                f"{100 * summary['extrinsic_share']:.1f} % of the value", delta_color="off")
    m[3].metric("Perfect Foresight", _money(summary["value_perfect_foresight_eur"], u))
    m[4].metric("Cycles per day", f"{summary['cycles_per_day_mean']:.2f}")
    m[5].metric("Regression R² (median)", f"{summary.get('regression_r2_median', float('nan')):.2f}")

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
        st.subheader("Key figures")
        st.caption(f"Raw values: amounts in {u.currency} ({u.symbol}), energy in MWh, power in MW.")
        rows = [(k, v) for k, v in summary.items() if not k.startswith("runtime_")]
        st.dataframe(pd.DataFrame(rows, columns=["Quantity", "Value"]).astype({"Value": str}),
                     hide_index=True, width="stretch", height=420)
    with right:
        st.subheader("Runtimes")
        times = [(k[len("runtime_"):-2], v) for k, v in summary.items() if k.startswith("runtime_")]
        st.dataframe(pd.DataFrame(times, columns=["Step", "Seconds"]), hide_index=True, width="stretch",
                     column_config={"Seconds": st.column_config.NumberColumn(format="%.1f")})
        st.subheader("Downloads")
        files = [f for f in ("summary.csv", "daily.parquet", "regression.parquet", "convexity.parquet",
                             "paths.parquet")
                 if (run_dir / f).exists()]
        cols = st.columns(2)
        for k, f in enumerate(files):
            cols[k % 2].download_button(f, (run_dir / f).read_bytes(), f, width="stretch",
                                        icon=":material/download:")
        with st.expander("Run parameters"):
            st.json(spec)


def main() -> None:
    """Page setup and navigation."""
    st.set_page_config(page_title="BESS Valuation", page_icon=":material/battery_charging_full:",
                       layout="wide")
    st.logo(str(Path(__file__).with_name("logo.svg")), size="large")
    with st.sidebar:
        st.caption("Battery valuation on the day-ahead market (LSMC)")
    nav = st.navigation({
        "Valuation": [
            st.Page(runs_page, title="Runs", icon=":material/play_circle:", url_path="runs", default=True),
            st.Page(result_page, title="Result", icon=":material/insights:", url_path="result"),
        ],
        "Master data": [
            st.Page(batteries_page, title="Batteries", icon=":material/battery_full:", url_path="batteries"),
            st.Page(simulations_page, title="Simulations", icon=":material/stacked_line_chart:",
                    url_path="simulations"),
        ],
        "App": [
            st.Page(settings_page, title="Settings", icon=":material/tune:", url_path="settings"),
        ],
    })
    nav.run()


if __name__ == "__main__":  # streamlit runs the script as __main__
    main()
