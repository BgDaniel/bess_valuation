"""Background valuation jobs for the app: submit, run, report status.

A job is a folder ``<jobs_root>/<job id>/`` with

* ``job.json``: the specification (battery config fields, price source, engine settings);
* ``status.json``: state ("queued", "running", "done", "failed"), current step, overall
  progress (0..1), timestamps, estimated remaining seconds, the result folder and key
  figures when done, the error message if failed;
* ``log.txt``: log output of the worker process.

:func:`submit_job` starts ``python -m bess_valuation.jobs <job folder>`` as a separate
process, so a run survives a restart of the app and the app only reads files.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_JOBS_ROOT = Path("results") / "_jobs"
ACTIVE = ("queued", "running")
SUMMARY_KEYS = ("value_oos_eur", "value_oos_stderr_eur", "value_in_sample_eur", "value_intrinsic_eur",
                "value_perfect_foresight_eur", "extrinsic_share", "cycles_per_day_mean")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def submit_job(spec: dict[str, Any], jobs_root: str | Path = DEFAULT_JOBS_ROOT,
               start: bool = True) -> Path:
    """Create a job folder and start the worker process.

    Args:
        spec: ``{"battery": {...BatteryConfig fields}, "prices": {...}, "settings":
            {...EngineSettings fields}, "results_root": "results"}``. Prices are either
            ``{"source": "synthetic", "generator_config": path, "n_paths", "n_days",
            "start", "seed"}`` or ``{"source": "folder", "folder": path}``. Optional
            ``"units"``: display units ``{"currency", "energy"}`` for the figures.
        jobs_root: Folder holding all job folders.
        start: Start the worker process (False: only create the folder, e.g. in tests).

    Returns:
        The job folder.
    """
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    job_dir = Path(jobs_root) / job_id
    job_dir.mkdir(parents=True)
    _write_json(job_dir / "job.json", spec)
    status = {"id": job_id, "name": spec["battery"].get("name", ""), "state": "queued",
              "step": "waiting", "progress": 0.0, "created": _now(), "started": None,
              "finished": None, "eta_s": None, "run_dir": None, "summary": None, "error": None,
              "pid": None}
    _write_json(job_dir / "status.json", status)
    if start:
        with open(job_dir / "log.txt", "w", encoding="utf-8") as log:  # the child keeps its own copy
            proc = subprocess.Popen([sys.executable, "-m", "bess_valuation.jobs", str(job_dir)],
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                    cwd=os.getcwd())
        status["pid"] = proc.pid
        _write_json(job_dir / "status.json", status)
    return job_dir


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():  # a finished but not yet reaped child is a zombie ("Z")
        try:
            return stat.read_text().split(")")[-1].split()[0] != "Z"
        except OSError:
            return True
    return True


def read_status(job_dir: str | Path) -> dict[str, Any]:
    """Status of a job; an active job whose worker process has died is reported as failed."""
    job_dir = Path(job_dir)
    status = _read_json(job_dir / "status.json")
    if status["state"] in ACTIVE and status.get("pid") and not _alive(status["pid"]):
        status = _read_json(job_dir / "status.json")  # it may have finished meanwhile
        if status["state"] in ACTIVE:
            status["state"] = "failed"
            status["error"] = "worker process ended unexpectedly (see log.txt)"
    status["job_dir"] = str(job_dir)
    return status


def list_jobs(jobs_root: str | Path = DEFAULT_JOBS_ROOT) -> list[dict[str, Any]]:
    """Status of all jobs, newest first."""
    root = Path(jobs_root)
    if not root.exists():
        return []
    jobs = [read_status(d) for d in root.iterdir() if (d / "status.json").exists()]
    return sorted(jobs, key=lambda s: s["created"], reverse=True)


def run_job(job_dir: str | Path) -> dict[str, Any]:
    """Run the valuation of a job folder and keep ``status.json`` up to date.

    Returns:
        The final status.
    """
    from bess_valuation import BatteryConfig, EngineSettings, PriceScenarios, run_engine
    from bess_valuation.price_generator import PriceGeneratorConfig, generate_prices

    job_dir = Path(job_dir)
    spec = _read_json(job_dir / "job.json")
    status = _read_json(job_dir / "status.json")
    t_start = time.time()
    last_write = [0.0]

    def update(force: bool = False, **fields: Any) -> None:
        status.update(fields)
        now = time.time()
        if force or now - last_write[0] >= 1.0:
            elapsed = now - t_start
            p = status["progress"]
            status["elapsed_s"] = elapsed
            status["eta_s"] = elapsed * (1.0 - p) / p if p > 0.01 and status["state"] == "running" else None
            _write_json(job_dir / "status.json", status)
            last_write[0] = now

    update(force=True, state="running", step="preparing", started=_now(), pid=os.getpid())
    try:
        config = BatteryConfig.from_dict(spec["battery"])
        settings = EngineSettings(**spec.get("settings", {}))
        src = spec["prices"]
        update(force=True, step="generating prices" if src["source"] == "synthetic" else "loading prices")
        if src["source"] == "synthetic":
            gen = PriceGeneratorConfig.from_yaml(src["generator_config"])
            prices = generate_prices(gen, n_paths=int(src["n_paths"]), start=src["start"],
                                     n_days=int(src["n_days"]), dt=config.dt, seed=int(src["seed"]))
        else:
            prices = PriceScenarios.from_files(src["folder"])
        result = run_engine(config, prices, settings,
                            progress=lambda step, p: update(step=step, progress=p))
        update(force=True, step="saving results and figures", progress=0.999)
        from bess_valuation.units import Units

        run_dir = result.save(spec.get("results_root", "results"), plots=True, dark_figures=True,
                              units=Units.from_dict(spec.get("units")))
        summary = result.summary()
        update(force=True, state="done", step="done", progress=1.0, finished=_now(),
               run_dir=str(run_dir), summary={k: summary[k] for k in SUMMARY_KEYS})
    except Exception as exc:  # noqa: BLE001 - reported to the app, re-raised for the log
        update(force=True, state="failed", finished=_now(), error=f"{type(exc).__name__}: {exc}",
               traceback=traceback.format_exc())
        raise
    return status


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_job(sys.argv[1])
