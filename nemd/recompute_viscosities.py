"""Check the fixed-window stress calculation for the conditional NEMD study."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
plan = json.loads((ROOT / "registration.json").read_text())
reference = {row["id"]: row for row in json.loads((ROOT / "archived_viscosities.json").read_text())}
for case in plan["cases"]:
    id_ = case["id"]
    path = ROOT / f"cases/{id_}/flow/stress_blocks.txt"
    data = np.loadtxt(path, comments="#")
    strain = data[:, 0] * case["dt"] * case["rate"]
    selected = (strain > 20 + 1e-9) & (strain <= 50 + 1e-9)
    if data[-1, 0] != case["flow_steps"]:
        raise ValueError(f"Incomplete flow in case {id_}")
    if np.max(np.abs(data[:, 4] + data[:, 2] / case["rate"])) > 1e-3:
        raise ValueError("Configurational stress sign or scale mismatch")
    value = float(data[selected, 4].mean())
    archived = reference[id_]["primary_eta"]
    if abs(value - archived) > 1e-10 or int(selected.sum()) != reference[id_]["measured_rows"]:
        raise ValueError("Archived result mismatch")
    print(id_, f"{value:.10f}")
