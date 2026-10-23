"""Background jobs, engine progress reporting and the Streamlit pages."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bess_valuation import BatteryConfig, EngineSettings, PriceGeneratorConfig, generate_prices, run_engine
from bess_valuation.jobs import list_jobs, read_status, run_job, submit_job

REPO = Path(__file__).resolve().parents[1]
BATTERY = BatteryConfig(name="app_test", e_max=2.0, p_charge_max=1.0, p_discharge_max=1.0,
                        dt=1.0, soc_init=1.0, soc_grid_size=3)


def tiny_spec(results_root, n_paths=6, n_days=4):
    return {"battery": BATTERY.to_dict(),
            "prices": {"source": "synthetic",
                       "generator_config": str(REPO / "configs" / "prices_synthetic.yaml"),
                       "n_paths": n_paths, "n_days": n_days, "start": "2027-06-01", "seed": 1},
            "settings": {"library_dir": None, "value_cache_dir": None, "n_regression_paths": 3},
            "results_root": str(results_root)}


class TestProgress(unittest.TestCase):
    def test_progress_is_monotone_and_complete(self):
        p = generate_prices(PriceGeneratorConfig(), n_paths=6, start="2027-06-01", n_days=4, dt=1.0, seed=2)
        calls = []
        run_engine(BATTERY, p, EngineSettings(library_dir=None, value_cache_dir=None),
                   progress=lambda step, frac: calls.append((step, frac)))
        fracs = [f for _, f in calls]
        self.assertTrue(np.all(np.diff(fracs) >= -1e-12))
        self.assertAlmostEqual(fracs[-1], 1.0)
        steps = [s for s, _ in calls]
        for name in ("daily values (prices)", "backward induction", "forward simulation", "schedules"):
            self.assertIn(name, steps)


class TestJobs(unittest.TestCase):
    def test_run_job_writes_status_and_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = submit_job(tiny_spec(Path(tmp) / "res"), Path(tmp) / "jobs", start=False)
            self.assertEqual(read_status(job_dir)["state"], "queued")
            status = run_job(job_dir)
            self.assertEqual(status["state"], "done")
            stored = read_status(job_dir)
            self.assertEqual(stored["state"], "done")
            self.assertEqual(stored["progress"], 1.0)
            run_dir = Path(stored["run_dir"])
            self.assertEqual(len(list((run_dir / "figures").glob("*.png"))), 6)
            self.assertEqual(len(list((run_dir / "figures_dark").glob("*.png"))), 6)
            self.assertIn("value_oos_eur", stored["summary"])
            self.assertEqual([j["id"] for j in list_jobs(Path(tmp) / "jobs")], [job_dir.name])

    def test_failed_job_reports_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = tiny_spec(Path(tmp) / "res")
            spec["battery"]["e_max"] = -1.0
            job_dir = submit_job(spec, Path(tmp) / "jobs", start=False)
            with self.assertRaises(ValueError):
                run_job(job_dir)
            status = read_status(job_dir)
            self.assertEqual(status["state"], "failed")
            self.assertIn("e_min", status["error"])

    def test_dead_worker_is_reported_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = submit_job(tiny_spec(Path(tmp) / "res"), Path(tmp) / "jobs", start=False)
            proc = subprocess.Popen([sys.executable, "-c", "pass"])
            proc.wait()
            status = json.loads((job_dir / "status.json").read_text())
            status.update(state="running", pid=proc.pid)
            (job_dir / "status.json").write_text(json.dumps(status))
            self.assertEqual(read_status(job_dir)["state"], "failed")

    def test_worker_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = submit_job(tiny_spec(Path(tmp) / "res", n_paths=4, n_days=3), Path(tmp) / "jobs")
            pid = read_status(job_dir)["pid"]
            os.waitpid(pid, 0)
            status = read_status(job_dir)
            self.assertEqual(status["state"], "done", (job_dir / "log.txt").read_text()[-2000:])


try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # optional dependency group "app"
    AppTest = None


@unittest.skipIf(AppTest is None, "streamlit not installed (poetry install --with app)")
class TestStreamlitApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.cwd = os.getcwd()
        os.chdir(cls.tmp.name)  # the app works relative to the current folder
        cls.job_dir = submit_job(tiny_spec("results"), Path("results") / "_jobs", start=False)
        run_job(cls.job_dir)

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.cwd)
        cls.tmp.cleanup()

    def test_runs_page(self):
        at = AppTest.from_file(str(REPO / "app" / "streamlit_app.py"), default_timeout=60)
        at.run()
        self.assertFalse(at.exception, at.exception)
        self.assertEqual(at.title[0].value, "BESS Valuation")
        self.assertTrue(any(b.label == "Berechnen" for b in at.button))
        frame = at.dataframe[0].value
        self.assertEqual(len(frame), 1)
        self.assertIn("fertig", frame["Status"].iloc[0])
        self.assertTrue(frame["Ergebnis"].iloc[0].endswith(self.job_dir.name))

    def test_result_page(self):
        def script(app_dir):
            import sys

            sys.path.insert(0, app_dir)
            import streamlit_app

            streamlit_app.result_page()

        at = AppTest.from_function(script, args=(str(REPO / "app"),), default_timeout=60)
        at.query_params["run"] = self.job_dir.name
        at.run()
        self.assertFalse(at.exception, at.exception)
        self.assertEqual(at.title[0].value, "app_test")
        self.assertEqual(len(at.metric), 5)
        self.assertEqual(len(at.tabs), 6)

if __name__ == "__main__":
    unittest.main()
