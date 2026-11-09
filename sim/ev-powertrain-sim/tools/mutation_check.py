"""A10 mutation meta-test: break one layer in a COPY of the repo and confirm the matching tests fail (exit code != 0).
A mutation that no test notices is a blind spot. Run: python tools/mutation_check.py"""
import shutil, subprocess, sys, tempfile, pathlib, time
SRC = pathlib.Path(__file__).resolve().parent.parent
FAST = ["tests/test_sensors.py", "tests/test_inverter.py", "tests/test_bms.py"]
FW = ["tests/test_realism_firewall.py", "-k", "U26 or U27 or U28 or U30 or U31 or U33"]
M = [  # name, file, old, new, pytest args
 ("noise x10", "sensors.py", "sig_i = cfg.i_noise_lsb * self.LSB_i", "sig_i = 10.0 * cfg.i_noise_lsb * self.LSB_i", FAST),
 ("delay bypassed", "inverter.py", "if self.cfg.delay_steps > 0:", "if False:", FAST),
 ("dead-time sign flipped", "inverter.py", "v = [-K * self._s(it)", "v = [+K * self._s(it)", FAST),
 ("angle error dropped", "sensors.py", "        self.last_delta = delta", "        delta = 0.0\n        self.last_delta = delta", FAST),
 ("BMS looks at the present (no latency)", "bms.py", "target = t - self.cfg.latency_s", "target = t + self.cfg.latency_s", FAST),
 ("controller imports the plant (reads truth)", "control_fw.py", "from __future__ import annotations", "from __future__ import annotations\nimport plant as _leak", FW),
 ("plant ignores the inverter error", "plant.py", "u_gem = u_app if u_extra is None else u_app + np.asarray(u_extra, dtype=float)", "u_gem = u_app", FW + ["tests/test_inverter.py"] if False else FW),
]


def run(args, cwd):
    return subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", *args], cwd=cwd, capture_output=True, text=True, timeout=900)


def main():
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="mut_"))
    ign = shutil.ignore_patterns("runs", "runs_fw", "dashboard", "__pycache__", "matrix*", "cycles", "sensitivity", "figures", "*.parquet", "*.zip", "viewer", "probes")
    shutil.copytree(SRC, tmp / "repo", ignore=ign)
    repo = tmp / "repo"
    t0 = time.time(); base = run(FAST + [], repo); print("baseline (unmutated) fast suite:", "PASS" if base.returncode == 0 else "FAIL", f"{time.time()-t0:.0f}s")
    if base.returncode != 0:
        print(base.stdout[-1500:]); sys.exit(2)
    bad = 0
    for name, fn, old, new, args in M:
        p = repo / fn; src = p.read_text()
        assert old in src, f"mutation text not found in {fn}: {old!r}"
        p.write_text(src.replace(old, new, 1))
        t0 = time.time(); r = run(args, repo); p.write_text(src)
        caught = r.returncode != 0
        failed = [l for l in r.stdout.splitlines() if l.startswith("FAILED")][:1]
        print(f"{'CAUGHT ' if caught else 'MISSED '} {name:45s} {time.time()-t0:5.0f}s  {failed[0][:110] if failed else ''}")
        bad += not caught
    print("blind spots:", bad)
    shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
