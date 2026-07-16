"""
xtb_extra_properties.py

Companion to fukui_batch.py. Reuses the already-optimized geometries in
<fukui_output_dir>/molecules/<safe_id>/conf_*/xtbopt.xyz (no re-optimization)
and runs a fast GFN2-xTB single point + implicit water solvation (GBSA) on
each, in parallel across molecules, to extract additional descriptors useful
for a metabolic/hepatocyte-stability model:

  Whole-molecule, Boltzmann-averaged across the kept conformer ensemble
  (same energy window / weighting scheme as fukui_batch.py):
    - HOMO energy, LUMO energy, HOMO-LUMO gap        (oxidation susceptibility)
    - Global electrophilicity index, chemical hardness
    - Dipole moment
    - Isotropic polarizability
    - GBSA solvation free energy in water

  Taken from the single lowest-energy conformer only (geometry-specific,
  not averaged):
    - Per-atom Mulliken-type partial charges + coordination number (JSON)
    - Weakest Wiberg bond order in the molecule + which atoms it connects
      (cheap proxy for a likely site of metabolism)

No Hessian / vibrational thermochemistry is computed (kept fast, single
points only), per your call on the runtime/relevance tradeoff.

Usage:
    python xtb_extra_properties.py fukui_output/aspirin_batch_fukui.csv fukui_output/ --workers 8

    (first positional arg = the CSV produced by fukui_batch.py,
     second = the fukui_batch.py output_dir that contains molecules/)

Output:
    <output_dir>/<fukui_csv_stem>_xtbprops.csv   -- fukui CSV + new columns
"""

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ENERGY_WINDOW_KCAL = 3.0
KT_298K_KCAL = 0.593
GFN_LEVEL = "2"
GBSA_SOLVENT = "water"
XTB_BIN = "xtb"
HARTREE_TO_KCAL = 627.509

NEW_COLUMNS = [
    "xtb_status",
    "xtb_n_conformers_used",
    "xtb_homo_ev",
    "xtb_lumo_ev",
    "xtb_gap_ev",
    "xtb_electrophilicity_ev",
    "xtb_hardness_ev",
    "xtb_dipole_debye",
    "xtb_polarizability_au",
    "xtb_gsolv_kcal",
    "xtb_charge_min",
    "xtb_charge_max",
    "xtb_charge_max_heteroatom",
    "xtb_weakest_bond_order",
    "xtb_weakest_bond_atoms",
    "xtb_atoms_json",
    "xtb_error",
]

HETEROATOMS = {"N", "O", "S", "P", "F", "Cl", "Br", "I"}


# --------------------------------------------------------------------------
# xtb invocation (same stack-ulimit wrapper as fukui_batch.py)
# --------------------------------------------------------------------------

def _wrapped_xtb_cmd(xtb_args):
    inner = ("ulimit -s unlimited 2>&1; echo XTB_WRAPPER_ULIMIT=$(ulimit -s) 1>&2; exec "
              + shlex.join([XTB_BIN] + xtb_args))
    return ["bash", "-c", inner]


def _xtb_env(n_threads=1):
    env = os.environ.copy()
    env["OMP_STACKSIZE"] = "4G"
    env["OMP_NUM_THREADS"] = str(n_threads)
    env["MKL_NUM_THREADS"] = str(n_threads)
    return env


def run_xtb_singlepoint(xyz_path, workdir, charge=0, n_threads=1):
    xyz_path = Path(xyz_path)
    xtb_args = [xyz_path.name, "--gfn", GFN_LEVEL, "--chrg", str(charge),
                "--gbsa", GBSA_SOLVENT]
    cmd = _wrapped_xtb_cmd(xtb_args)
    proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True,
                           timeout=900, env=_xtb_env(n_threads))
    if proc.returncode != 0:
        return None, f"xtb single point failed:\n{proc.stderr[-2000:]}"
    return proc.stdout, None


# --------------------------------------------------------------------------
# Parsing helpers. Written against typical xtb 6.x stdout formatting --
# validate on a small test batch before trusting on a large run.
# --------------------------------------------------------------------------

def parse_energy(stdout):
    for line in stdout.splitlines():
        if "TOTAL ENERGY" in line:
            return float(line.split()[3])
    return None


def parse_homo_lumo(stdout):
    homo = lumo = None
    for line in stdout.splitlines():
        if "(HOMO)" in line:
            toks = line.split()
            homo = float(toks[-2])  # ... Energy/Eh  Energy/eV  (HOMO)
        elif "(LUMO)" in line:
            toks = line.split()
            lumo = float(toks[-2])
    return homo, lumo


def parse_dipole(stdout):
    for line in stdout.splitlines():
        if line.strip().startswith("full:"):
            toks = line.split()
            try:
                return float(toks[-1])
            except (ValueError, IndexError):
                continue
    return None


def parse_polarizability(stdout):
    m = re.search(r"Mol\.\s*[Aa]lpha\(0\)\s*/?\s*au\s*:\s*(-?\d+\.\d+)", stdout)
    if m:
        return float(m.group(1))
    m = re.search(r"Mol\.\s*\S*\(0\)\s*/\s*au\s*:\s*(-?\d+\.\d+)", stdout)
    return float(m.group(1)) if m else None


def parse_gsolv(stdout):
    m = re.search(r"->\s*Gsolv\s+-?\d+\.\d+\s*Eh\s+(-?\d+\.\d+)\s*kcal", stdout)
    return float(m.group(1)) if m else None


def parse_atomic_properties(stdout):
    """Returns list of (atom_idx0, element, charge, covCN) in xyz atom order."""
    lines = stdout.splitlines()
    start = None
    for i, line in enumerate(lines):
        if "covCN" in line and ("q" in line) and ("(0)" in line or "alpha" in line.lower()):
            start = i + 1
            break
    if start is None:
        return []

    row_re = re.compile(
        r"^\s*(\d+)\s+(\d+)([A-Za-z]{1,2})\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$"
    )
    out = []
    for line in lines[start:]:
        m = row_re.match(line)
        if m:
            idx1, _z, elem, covcn, q, _c6, _a0 = m.groups()
            out.append((int(idx1) - 1, elem, float(q), float(covcn)))
        elif out:
            break
    return out


def parse_wbo(workdir):
    wbo_path = Path(workdir) / "wbo"
    if not wbo_path.exists():
        return []
    pairs = []
    for line in wbo_path.read_text().splitlines():
        toks = line.split()
        if len(toks) != 3:
            continue
        try:
            i, j, val = int(toks[0]) - 1, int(toks[1]) - 1, float(toks[2])
        except ValueError:
            continue
        pairs.append((i, j, val))
    return pairs


# --------------------------------------------------------------------------
# Per-molecule driver
# --------------------------------------------------------------------------

def process_molecule_extra(mol_dir, charge, n_threads):
    result = {c: "" for c in NEW_COLUMNS}
    result["xtb_status"] = "error"

    try:
        mol_dir = Path(mol_dir)
        conf_dirs = sorted(mol_dir.glob("conf_*"))
        conf_records = []  # (energy, stdout, workdir)
        for cd in conf_dirs:
            xyz = cd / "xtbopt.xyz"
            if not xyz.exists():
                continue
            stdout, err = run_xtb_singlepoint(xyz, cd, charge=charge, n_threads=n_threads)
            if stdout is None:
                continue
            energy = parse_energy(stdout)
            if energy is None:
                continue
            conf_records.append((energy, stdout, cd))

        if not conf_records:
            result["xtb_error"] = "no conformer produced a usable single-point result"
            return result

        min_e = min(r[0] for r in conf_records)
        kept = [r for r in conf_records if (r[0] - min_e) * HARTREE_TO_KCAL <= ENERGY_WINDOW_KCAL]
        rel_e_kcal = np.array([(r[0] - min_e) * HARTREE_TO_KCAL for r in kept])
        weights = np.exp(-rel_e_kcal / KT_298K_KCAL)
        weights /= weights.sum()

        homos, lumos, dipoles, alphas, gsolvs, omegas, etas = [], [], [], [], [], [], []
        for (_e, stdout, _wd) in kept:
            homo, lumo = parse_homo_lumo(stdout)
            if homo is None or lumo is None:
                continue
            gap = lumo - homo
            mu = (homo + lumo) / 2.0
            eta = gap
            omega = (mu ** 2) / (2 * eta) if eta > 0 else None
            homos.append(homo)
            lumos.append(lumo)
            etas.append(eta)
            omegas.append(omega if omega is not None else np.nan)
            dipoles.append(parse_dipole(stdout))
            alphas.append(parse_polarizability(stdout))
            gsolvs.append(parse_gsolv(stdout))

        if not homos:
            result["xtb_error"] = "HOMO/LUMO not found in any conformer's xtb output"
            return result

        # weights array must match the (possibly shorter) list that had valid HOMO/LUMO
        w = weights[: len(homos)]
        w = w / w.sum()

        def wavg(vals):
            arr = np.array([v if v is not None else np.nan for v in vals], dtype=float)
            mask = ~np.isnan(arr)
            if not mask.any():
                return None
            ww = w[mask] / w[mask].sum()
            return float(np.dot(ww, arr[mask]))

        result["xtb_status"] = "ok"
        result["xtb_n_conformers_used"] = len(kept)
        result["xtb_homo_ev"] = round(wavg(homos), 5)
        result["xtb_lumo_ev"] = round(wavg(lumos), 5)
        result["xtb_gap_ev"] = round(wavg(etas), 5)
        result["xtb_electrophilicity_ev"] = round(wavg(omegas), 5) if wavg(omegas) is not None else ""
        result["xtb_hardness_ev"] = round(wavg(etas), 5)
        result["xtb_dipole_debye"] = round(wavg(dipoles), 5) if wavg(dipoles) is not None else ""
        result["xtb_polarizability_au"] = round(wavg(alphas), 5) if wavg(alphas) is not None else ""
        result["xtb_gsolv_kcal"] = round(wavg(gsolvs), 5) if wavg(gsolvs) is not None else ""

        # Geometry-specific descriptors: lowest-energy conformer only.
        best_energy, best_stdout, best_wd = min(conf_records, key=lambda r: r[0])
        atoms = parse_atomic_properties(best_stdout)
        if atoms:
            charges = [a[2] for a in atoms]
            result["xtb_charge_min"] = round(min(charges), 5)
            result["xtb_charge_max"] = round(max(charges), 5)
            hetero_charges = [a[2] for a in atoms if a[1] in HETEROATOMS]
            if hetero_charges:
                result["xtb_charge_max_heteroatom"] = round(max(hetero_charges, key=abs), 5)
            result["xtb_atoms_json"] = json.dumps([
                {"idx": a[0], "elem": a[1], "charge": round(a[2], 5), "covCN": round(a[3], 3)}
                for a in atoms
            ])

        wbo = parse_wbo(best_wd)
        if wbo:
            # ignore near-zero / non-bonded entries some xtb versions include
            bonded = [t for t in wbo if t[2] > 0.3]
            if bonded:
                i, j, val = min(bonded, key=lambda t: t[2])
                elem_i = next((a[1] for a in atoms if a[0] == i), "?")
                elem_j = next((a[1] for a in atoms if a[0] == j), "?")
                result["xtb_weakest_bond_order"] = round(val, 4)
                result["xtb_weakest_bond_atoms"] = f"{i}-{j}({elem_i}-{elem_j})"

        result["xtb_error"] = ""
        return result

    except Exception as exc:  # noqa: BLE001
        result["xtb_error"] = f"{type(exc).__name__}: {exc}"
        return result


def _worker_task(args):
    row_index, mol_dir, charge, n_threads = args
    t0 = time.time()
    result = process_molecule_extra(mol_dir, charge, n_threads)
    result["_row_index"] = row_index
    result["_elapsed_s"] = round(time.time() - t0, 1)
    return result


def _safe_dirname(mol_id, seen):
    """Must exactly match the naming scheme in fukui_batch.py so directories line up."""
    safe = re.sub(r"[^A-Za-z0-9_\-]", "_", str(mol_id)) or "mol"
    candidate = safe
    n = 1
    while candidate in seen:
        n += 1
        candidate = f"{safe}_dup{n}"
    seen.add(candidate)
    return candidate


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fukui_csv", help="CSV produced by fukui_batch.py")
    ap.add_argument("fukui_output_dir", help="output_dir passed to fukui_batch.py (contains molecules/)")
    ap.add_argument("--id-col", default="ID")
    ap.add_argument("--charge-col", default=None,
                     help="Must match the --charge-col used with fukui_batch.py, if any")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--only-ok", action="store_true", default=True,
                     help="Skip molecules where fukui_status != ok (default: on)")
    args = ap.parse_args()

    fukui_csv = Path(args.fukui_csv)
    fukui_output_dir = Path(args.fukui_output_dir)
    mol_root = fukui_output_dir / "molecules"
    if not mol_root.exists():
        sys.exit(f"ERROR: {mol_root} not found -- is fukui_output_dir correct?")

    out_csv = Path(args.out_csv) if args.out_csv else fukui_output_dir / f"{fukui_csv.stem}_xtbprops.csv"

    with open(fukui_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if args.id_col not in fieldnames:
        sys.exit(f"ERROR: ID column '{args.id_col}' not found. Columns: {fieldnames}")
    if "fukui_status" not in fieldnames:
        sys.exit("ERROR: input CSV doesn't look like fukui_batch.py output (no 'fukui_status' column)")

    n_total = len(rows)
    n_workers = args.workers or max(1, min(os.cpu_count() or 1, n_total))
    threads_per_worker = max(1, (os.cpu_count() or 1) // n_workers)
    print(f"Rows: {n_total}  |  workers: {n_workers}  |  xtb threads/worker: {threads_per_worker}")

    seen_dirs = set()
    jobs = []
    skipped = 0
    for i, row in enumerate(rows):
        safe_dir_name = _safe_dirname(row[args.id_col], seen_dirs)  # must run for every row to keep dedup in sync
        if args.only_ok and row.get("fukui_status") != "ok":
            skipped += 1
            continue
        charge = 0
        if args.charge_col:
            raw = row.get(args.charge_col, "")
            charge = int(raw) if str(raw).strip() != "" else 0
        mol_dir = mol_root / safe_dir_name
        jobs.append((i, mol_dir, charge, threads_per_worker))

    if skipped:
        print(f"Skipping {skipped} rows with fukui_status != ok (use --only-ok=False to override)")

    for c in NEW_COLUMNS:
        if c not in fieldnames:
            fieldnames.append(c)
    for row in rows:
        for c in NEW_COLUMNS:
            row.setdefault(c, "")

    def write_csv(current_rows):
        tmp = out_csv.with_suffix(out_csv.suffix + ".tmp")
        with open(tmp, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(current_rows)
        tmp.replace(out_csv)

    if not jobs:
        print("Nothing to run.")
        write_csv(rows)
        return

    done = 0
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_worker_task, job): job[0] for job in jobs}
        for fut in as_completed(futures):
            result = fut.result()
            idx = result.pop("_row_index")
            elapsed = result.pop("_elapsed_s")
            rows[idx].update(result)
            done += 1
            status = result["xtb_status"]
            mol_id = rows[idx][args.id_col]
            print(f"[{done}/{len(jobs)}] {mol_id}: {status} ({elapsed}s)"
                  + (f" -- {result['xtb_error']}" if status != "ok" else ""))

            if done % args.checkpoint_every == 0 or done == len(jobs):
                write_csv(rows)

    total_min = (time.time() - t_start) / 60
    n_ok = sum(1 for r in rows if r.get("xtb_status") == "ok")
    print(f"\nDone: {n_ok}/{len(jobs)} succeeded in {total_min:.1f} min")
    print(f"Results written to {out_csv}")


if __name__ == "__main__":
    main()