"""
fukui_batch.py

Batch, multi-process version of fukui_single.py. Reads a CSV of SMILES + ID,
runs the RDKit -> xtb -> Fukui pipeline for every molecule in parallel across
processes, and writes an augmented copy of the input CSV with the results
appended (Option A: one summary row per molecule, with a JSON blob of
per-atom detail plus a handful of aggregate columns for direct use as
molecule-level features, e.g. in chemprop).

Requirements (install in your own environment, not available in this sandbox):
    conda create -n xtb-env -c conda-forge xtb rdkit numpy -y
    conda activate xtb-env

Usage:
    python fukui_batch.py input.csv output_dir/
    python fukui_batch.py input.csv output_dir/ --workers 8
    python fukui_batch.py input.csv output_dir/ --smiles-col smiles --id-col mol_id
    python fukui_batch.py input.csv output_dir/ --charge-col formal_charge

Output:
    output_dir/<input_stem>_fukui.csv   -- copy of input CSV + result columns
    output_dir/molecules/<safe_id>/     -- per-molecule xtb scratch + conf dirs
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
from rdkit import Chem
from rdkit.Chem import AllChem

N_CONFS_INITIAL = 50
N_CONFS_KEEP = 10
RMSD_PRUNE_THRESHOLD = 0.5     # angstrom
ENERGY_WINDOW_KCAL = 3.0
KT_298K_KCAL = 0.593
GFN_LEVEL = "2"
XTB_BIN = "xtb"
HARTREE_TO_KCAL = 627.509

NEW_COLUMNS = [
    "fukui_status",
    "fukui_n_atoms",
    "fukui_n_conformers_kept",
    "fukui_f_plus_max",
    "fukui_f_plus_mean",
    "fukui_f_minus_max",
    "fukui_f_minus_mean",
    "fukui_f_zero_max",
    "fukui_f_zero_mean",
    "fukui_json",
    "fukui_error",
]


# --------------------------------------------------------------------------
# Per-molecule mechanics (same logic as fukui_single.py, with n_threads
# threaded through so each worker process pins xtb to its own core budget
# instead of every worker trying to grab all cores at once).
# --------------------------------------------------------------------------

def generate_conformers(mol):
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    params.pruneRmsThresh = RMSD_PRUNE_THRESHOLD
    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=N_CONFS_INITIAL, params=params))
    if not conf_ids:
        raise RuntimeError("RDKit embedding failed to generate any conformers")

    scored = []
    for cid in conf_ids:
        props = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
        if ff is None:
            continue
        ff.Minimize(maxIts=2000)
        scored.append((cid, ff.CalcEnergy()))

    scored.sort(key=lambda x: x[1])
    keep_ids = [cid for cid, _ in scored[:N_CONFS_KEEP]]
    return mol, keep_ids


def write_xyz(mol, conf_id, path):
    conf = mol.GetConformer(conf_id)
    lines = [str(mol.GetNumAtoms()), ""]
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")
    Path(path).write_text("\n".join(lines))


def _wrapped_xtb_cmd(xtb_args):
    """
    Raise the stack ulimit immediately before exec'ing xtb (needed for its
    OpenMP code regardless of how many threads it's given -- without this it
    can segfault with an opaque backtrace on deep recursion in the SCF code).
    """
    inner = ("ulimit -s unlimited 2>&1; echo XTB_WRAPPER_ULIMIT=$(ulimit -s) 1>&2; exec "
              + shlex.join([XTB_BIN] + xtb_args))
    return ["bash", "-c", inner]


def _xtb_env(n_threads=1):
    env = os.environ.copy()
    env["OMP_STACKSIZE"] = "4G"
    env["OMP_NUM_THREADS"] = str(n_threads)
    env["MKL_NUM_THREADS"] = str(n_threads)
    return env


def run_xtb_opt(xyz_path, workdir, charge=0, uhf=0, n_threads=1):
    xyz_path = Path(xyz_path)
    xtb_args = [xyz_path.name, "--gfn", GFN_LEVEL, "--opt", "tight",
                "--chrg", str(charge), "--uhf", str(uhf)]
    cmd = _wrapped_xtb_cmd(xtb_args)
    proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True,
                           timeout=1800, env=_xtb_env(n_threads))
    if proc.returncode != 0:
        return None, None, "xtb --opt failed:\n" + proc.stderr[-2000:]

    energy = None
    for line in proc.stdout.splitlines():
        if "TOTAL ENERGY" in line:
            energy = float(line.split()[3])
    opt_xyz = Path(workdir) / "xtbopt.xyz"
    if energy is None or not opt_xyz.exists():
        return None, None, "xtb --opt produced no energy/xtbopt.xyz"
    return energy, opt_xyz, None


def run_xtb_vfukui(xyz_path, workdir, charge=0, n_threads=1):
    xyz_path = Path(xyz_path)
    xtb_args = [xyz_path.name, "--gfn", GFN_LEVEL, "--chrg", str(charge), "--vfukui"]
    cmd = _wrapped_xtb_cmd(xtb_args)
    proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True,
                           timeout=1800, env=_xtb_env(n_threads))
    if proc.returncode != 0:
        return None, "xtb --vfukui failed:\n" + proc.stderr[-2000:]

    lines = proc.stdout.splitlines()
    start = None
    for i, line in enumerate(lines):
        if "f(+)" in line and "f(-)" in line:
            start = i + 1
            break
    if start is None:
        return None, "could not locate Fukui table header in xtb output"

    # xtb prints rows like "   1C      -0.017   -0.002   -0.010" -- atom index
    # and element symbol are concatenated with no separator.
    row_re = re.compile(r"^\s*(\d+)([A-Za-z]{1,2})\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")

    fukui = []
    for line in lines[start:]:
        m = row_re.match(line)
        if m:
            _, _, f_plus, f_minus, f_zero = m.groups()
            fukui.append((float(f_plus), float(f_minus), float(f_zero)))
        elif fukui:
            break
    if not fukui:
        return None, "Fukui header matched but zero data rows parsed"
    return fukui, None


# --------------------------------------------------------------------------
# Single-molecule driver -- runs entirely inside one worker process.
# --------------------------------------------------------------------------

def process_molecule(mol_id, smiles, charge, mol_out_dir, n_threads):
    """
    Returns a dict of result columns (see NEW_COLUMNS). Never raises --
    all failure modes are captured into fukui_status/fukui_error so one bad
    molecule can't take down the batch.
    """
    result = {c: "" for c in NEW_COLUMNS}
    result["fukui_status"] = "error"

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            result["fukui_error"] = "RDKit could not parse SMILES"
            return result

        mol_out_dir = Path(mol_out_dir)
        mol_out_dir.mkdir(parents=True, exist_ok=True)

        mol, conf_ids = generate_conformers(mol)

        per_conf = []
        for i, cid in enumerate(conf_ids):
            conf_dir = mol_out_dir / f"conf_{i}"
            conf_dir.mkdir(exist_ok=True)
            raw_xyz = conf_dir / "input.xyz"
            write_xyz(mol, cid, raw_xyz)

            energy, opt_xyz, err = run_xtb_opt(raw_xyz, conf_dir, charge=charge, n_threads=n_threads)
            if energy is None:
                continue

            fukui, err = run_xtb_vfukui(opt_xyz, conf_dir, charge=charge, n_threads=n_threads)
            if fukui is None:
                continue

            per_conf.append({"energy_hartree": energy, "fukui": fukui})

        if not per_conf:
            result["fukui_error"] = "no conformer completed opt + vfukui successfully"
            return result

        min_e = min(r["energy_hartree"] for r in per_conf)
        kept = [r for r in per_conf
                if (r["energy_hartree"] - min_e) * HARTREE_TO_KCAL <= ENERGY_WINDOW_KCAL]

        rel_e_kcal = np.array([(r["energy_hartree"] - min_e) * HARTREE_TO_KCAL for r in kept])
        weights = np.exp(-rel_e_kcal / KT_298K_KCAL)
        weights /= weights.sum()

        n_atoms = len(kept[0]["fukui"])
        f_plus = np.zeros(n_atoms)
        f_minus = np.zeros(n_atoms)
        f_zero = np.zeros(n_atoms)
        for w, r in zip(weights, kept):
            arr = np.array(r["fukui"])
            f_plus += w * arr[:, 0]
            f_minus += w * arr[:, 1]
            f_zero += w * arr[:, 2]

        atoms_json = []
        for idx, atom in enumerate(mol.GetAtoms()):
            if idx >= n_atoms:
                break
            atoms_json.append({
                "idx": idx,
                "elem": atom.GetSymbol(),
                "f_plus": round(float(f_plus[idx]), 5),
                "f_minus": round(float(f_minus[idx]), 5),
                "f_zero": round(float(f_zero[idx]), 5),
            })

        result["fukui_status"] = "ok"
        result["fukui_n_atoms"] = n_atoms
        result["fukui_n_conformers_kept"] = len(kept)
        result["fukui_f_plus_max"] = round(float(f_plus.max()), 5)
        result["fukui_f_plus_mean"] = round(float(f_plus.mean()), 5)
        result["fukui_f_minus_max"] = round(float(f_minus.max()), 5)
        result["fukui_f_minus_mean"] = round(float(f_minus.mean()), 5)
        result["fukui_f_zero_max"] = round(float(f_zero.max()), 5)
        result["fukui_f_zero_mean"] = round(float(f_zero.mean()), 5)
        result["fukui_json"] = json.dumps(atoms_json)
        result["fukui_error"] = ""
        return result

    except Exception as exc:  # noqa: BLE001 - must never propagate to the pool
        result["fukui_error"] = f"{type(exc).__name__}: {exc}"
        return result


def _worker_task(args):
    """Top-level, picklable wrapper so ProcessPoolExecutor can dispatch it."""
    row_index, mol_id, smiles, charge, mol_out_dir, n_threads = args
    t0 = time.time()
    result = process_molecule(mol_id, smiles, charge, mol_out_dir, n_threads)
    result["_row_index"] = row_index
    result["_elapsed_s"] = round(time.time() - t0, 1)
    return result


def _safe_dirname(mol_id, seen):
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
    ap.add_argument("input_csv")
    ap.add_argument("output_dir")
    ap.add_argument("--smiles-col", default="SMILES")
    ap.add_argument("--id-col", default="ID")
    ap.add_argument("--charge-col", default=None,
                     help="Optional column with per-molecule formal charge (default: 0 for all)")
    ap.add_argument("--workers", type=int, default=None,
                     help="Number of worker processes (default: min(cpu_count, n_molecules))")
    ap.add_argument("--out-csv", default=None,
                     help="Output CSV path (default: <output_dir>/<input_stem>_fukui.csv)")
    ap.add_argument("--checkpoint-every", type=int, default=10,
                     help="Rewrite the output CSV after this many molecules complete (default: 10)")
    args = ap.parse_args()

    input_csv = Path(args.input_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mol_root = out_dir / "molecules"
    mol_root.mkdir(exist_ok=True)

    out_csv = Path(args.out_csv) if args.out_csv else out_dir / f"{input_csv.stem}_fukui.csv"

    with open(input_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if args.smiles_col not in fieldnames:
        sys.exit(f"ERROR: SMILES column '{args.smiles_col}' not found. Columns: {fieldnames}")
    if args.id_col not in fieldnames:
        sys.exit(f"ERROR: ID column '{args.id_col}' not found. Columns: {fieldnames}")

    n_mols = len(rows)
    if n_mols == 0:
        sys.exit("ERROR: input CSV has no data rows")

    n_workers = args.workers or max(1, min(os.cpu_count() or 1, n_mols))
    threads_per_worker = max(1, (os.cpu_count() or 1) // n_workers)
    print(f"Molecules: {n_mols}  |  workers: {n_workers}  |  xtb threads/worker: {threads_per_worker}")

    # Build per-worker jobs, keeping directories collision-free.
    seen_dirs = set()
    jobs = []
    for i, row in enumerate(rows):
        mol_id = row[args.id_col]
        smiles = row[args.smiles_col]
        charge = 0
        if args.charge_col:
            raw = row.get(args.charge_col, "")
            charge = int(raw) if str(raw).strip() != "" else 0
        safe_dir = mol_root / _safe_dirname(mol_id, seen_dirs)
        jobs.append((i, mol_id, smiles, charge, safe_dir, threads_per_worker))

    for c in NEW_COLUMNS:
        if c not in fieldnames:
            fieldnames.append(c)

    def write_csv(current_rows):
        tmp = out_csv.with_suffix(out_csv.suffix + ".tmp")
        with open(tmp, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(current_rows)
        tmp.replace(out_csv)

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
            status = result["fukui_status"]
            mol_id = rows[idx][args.id_col]
            print(f"[{done}/{n_mols}] {mol_id}: {status} ({elapsed}s)"
                  + (f" -- {result['fukui_error']}" if status != "ok" else ""))

            if done % args.checkpoint_every == 0 or done == n_mols:
                write_csv(rows)

    total_min = (time.time() - t_start) / 60
    n_ok = sum(1 for r in rows if r.get("fukui_status") == "ok")
    print(f"\nDone: {n_ok}/{n_mols} succeeded in {total_min:.1f} min")
    print(f"Results written to {out_csv}")


if __name__ == "__main__":
    main()