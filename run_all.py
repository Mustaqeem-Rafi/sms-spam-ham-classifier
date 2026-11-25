"""Run every phase of the case study in order and rebuild the report.

    python run_all.py              # full run (data -> EDA -> train -> evaluate -> notebook -> report -> PDF)
    python run_all.py --no-notebook --no-pdf
"""
import argparse
import importlib.metadata as md
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402

PACKAGES = ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "seaborn", "wordcloud",
            "nltk", "joblib", "python-docx", "jupyter", "nbconvert", "ipykernel"]


def _sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def record_environment():
    """versions.txt + env.json (report §5.3)."""
    C.ensure_dirs()
    lines = []
    for p in PACKAGES:
        try:
            lines.append(f"{p}=={md.version(p)}")
        except md.PackageNotFoundError:
            pass
    (C.RESULTS_DIR / "versions.txt").write_text("\n".join(lines) + "\n")

    system = platform.system()
    cpu, ram_gb, os_name = platform.processor(), None, f"{system} {platform.release()}"
    if system == "Darwin":
        cpu = _sh("sysctl -n machdep.cpu.brand_string") or cpu
        mem = _sh("sysctl -n hw.memsize")
        ram_gb = round(int(mem) / 1024 ** 3, 1) if mem.isdigit() else None
        os_name = f"macOS {_sh('sw_vers -productVersion')}"
        cores = _sh("sysctl -n hw.ncpu")
    elif system == "Linux":
        cpu = _sh("lscpu | grep 'Model name' | sed 's/Model name:\\s*//'") or cpu
        mem = _sh("grep MemTotal /proc/meminfo | awk '{print $2}'")
        ram_gb = round(int(mem) / 1024 ** 2, 1) if mem.isdigit() else None
        cores = _sh("nproc")
    else:
        cores = ""
    env = {"python_version": platform.python_version(), "os": os_name, "machine": platform.machine(),
           "cpu": cpu, "logical_cores": int(cores) if str(cores).isdigit() else None, "ram_gb": ram_gb,
           "packages": dict(l.split("==") for l in lines)}
    (C.RESULTS_DIR / "env.json").write_text(json.dumps(env, indent=2))
    return env


def step(title, fn, *a, **k):
    t0 = time.perf_counter()
    print(f"\n=== {title} ===")
    out = fn(*a, **k)
    print(f"--- {title}: {time.perf_counter() - t0:.1f}s")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-notebook", action="store_true", help="skip executing the notebook")
    ap.add_argument("--no-pdf", action="store_true", help="skip LibreOffice PDF export")
    args = ap.parse_args()
    t_start = time.perf_counter()

    from src.ablation import run_ablation
    from src.data import build_clean_dataset, save_split_table
    from src.eda import run_eda
    from src.evaluate import run_evaluation
    from src.flowchart import draw_workflow
    from src.interpret import run_interpretation
    from src.predict import save_cli_demo
    from src.preprocess import _self_check
    from src.train import train_all, worked_example

    step("Environment", record_environment)
    step("Preprocessing self-check", _self_check)
    step("Data: download, clean, deduplicate", build_clean_dataset)
    step("Data: split table", save_split_table)
    step("EDA", run_eda)
    step("Worked BoW/TF-IDF example", worked_example)
    step("Training (6 x GridSearchCV)", train_all)
    step("Evaluation", run_evaluation)
    step("Interpretation", run_interpretation)
    step("Workflow figure", draw_workflow)
    step("Ablation + learning curve", run_ablation)

    print(step("Prediction CLI demo", save_cli_demo))

    if not args.no_notebook:
        nb = ROOT / "notebooks" / "SMS_Spam_Detection.ipynb"
        step("Execute notebook", subprocess.run,
             [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
              "--ExecutePreprocessor.timeout=1200", str(nb)], cwd=ROOT, check=True)

    (C.RESULTS_DIR / "runtime.json").write_text(json.dumps(
        {"run_all_seconds_before_report": round(time.perf_counter() - t_start, 1)}, indent=2))

    sys.path.insert(0, str(ROOT / "report"))
    import build_report
    step("Report (.docx)", build_report.build)
    if not args.no_pdf:
        step("Report (.pdf)", build_report.to_pdf)
    print(f"\nAll done in {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
