"""
fukui_single.py

Test script: compute Boltzmann-averaged, per-atom Fukui indices (f+, f-, f0)
for ONE molecule given as a SMILES string. No protonation-state or tautomer
handling yet -- this is just to validate the RDKit -> xtb -> Fukui mechanics
before folding in the rest of the pipeline.

Requirements (install in your own environment, not available in this sandbox):
    conda create -n xtb-env -c conda-forge xtb rdkit numpy -y
    conda activate xtb-env

Usage:
    python fukui_single.py "CC(=O)Oc1ccccc1C(=O)O" aspirin_test/
    (SMILES for aspirin as an example)
"""

import sys
import re
import shlex
import subprocess
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
    print(f"  generated {len(conf_ids)} conformers, keeping best {len(keep_ids)} after MMFF prescreen")
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
    Build a command that raises the stack ulimit immediately before exec'ing
    xtb, so the fix doesn't depend on the calling shell having done this
    already (xtb's OpenMP code needs a large stack or it segfaults with an
    opaque backtrace).

    Also echoes the resulting ulimit to stderr so we can confirm whether the
    ulimit call actually succeeded in this subprocess (it can silently fail
    if the HARD limit is capped, since we chain with ';' not '&&').
    """
    inner = ("ulimit -s unlimited 2>&1; echo XTB_WRAPPER_ULIMIT=$(ulimit -s) 1>&2; exec "
              + shlex.join([XTB_BIN] + xtb_args))
    return ["bash", "-c", inner]


def _xtb_env():
    import os
    env = os.environ.copy()
    env["OMP_STACKSIZE"] = "4G"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    return env


def run_xtb_opt(xyz_path, workdir, charge=0, uhf=0):
    xyz_path = Path(xyz_path)
    xtb_args = [xyz_path.name, "--gfn", GFN_LEVEL, "--opt", "tight",
                "--chrg", str(charge), "--uhf", str(uhf)]
    cmd = _wrapped_xtb_cmd(xtb_args)
    proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True,
                           timeout=1800, env=_xtb_env())
    if proc.returncode != 0:
        print("  xtb optimization FAILED, full stderr:")
        print("  " + "\n  ".join(proc.stderr.splitlines()))
        return None, None

    energy = None
    for line in proc.stdout.splitlines():
        if "TOTAL ENERGY" in line:
            energy = float(line.split()[3])
    opt_xyz = Path(workdir) / "xtbopt.xyz"
    if energy is None or not opt_xyz.exists():
        return None, None
    return energy, opt_xyz


def run_xtb_vfukui(xyz_path, workdir, charge=0):
    xyz_path = Path(xyz_path)
    xtb_args = [xyz_path.name, "--gfn", GFN_LEVEL, "--chrg", str(charge), "--vfukui"]
    cmd = _wrapped_xtb_cmd(xtb_args)
    proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True,
                           timeout=1800, env=_xtb_env())
    if proc.returncode != 0:
        print("  xtb --vfukui FAILED, full stderr:")
        print("  " + "\n  ".join(proc.stderr.splitlines()))
        return None

    lines = proc.stdout.splitlines()
    start = None
    header_idx = None
    for i, line in enumerate(lines):
        if "f(+)" in line and "f(-)" in line:
            header_idx = i
            start = i + 1
            break
    if start is None:
        print("  could not locate Fukui table header -- last 40 lines of stdout:")
        print("  " + "\n  ".join(lines[-40:]))
        return None

    # xtb prints rows like "   1C      -0.017   -0.002   -0.010" -- the atom
    # index and element symbol are concatenated with no separator, so a naive
    # split()+isdigit() check rejects every row. Match it explicitly instead.
    row_re = re.compile(r"^\s*(\d+)([A-Za-z]{1,2})\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")

    fukui = []
    for line in lines[start:]:
        m = row_re.match(line)
        if m:
            _, _, f_plus, f_minus, f_zero = m.groups()
            fukui.append((float(f_plus), float(f_minus), float(f_zero)))
        elif fukui:
            break  # table ended (blank line / next section) after we had rows
        # else: keep skipping until the first matching row appears

    if not fukui:
        context = lines[max(0, header_idx - 3):header_idx + 25]
        print("  Fukui header matched but zero data rows parsed -- context around header:")
        for line in context:
            print(f"  RAW> {line!r}")
        return None

    return fukui


def main(smiles, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"ERROR: could not parse SMILES: {smiles}")
        sys.exit(1)

    print(f"Molecule: {smiles}  ({mol.GetNumAtoms()} heavy atoms)")
    mol, conf_ids = generate_conformers(mol)

    results = []
    for i, cid in enumerate(conf_ids):
        print(f"  conformer {i}: optimizing with xtb ...")
        conf_dir = out_dir / f"conf_{i}"
        conf_dir.mkdir(exist_ok=True)
        raw_xyz = conf_dir / "input.xyz"
        write_xyz(mol, cid, raw_xyz)

        energy, opt_xyz = run_xtb_opt(raw_xyz, conf_dir)
        if energy is None:
            print(f"  conformer {i}: skipped (optimization failed)")
            continue

        fukui = run_xtb_vfukui(opt_xyz, conf_dir)
        if fukui is None:
            print(f"  conformer {i}: skipped (vfukui failed)")
            continue

        print(f"  conformer {i}: OK, energy = {energy:.6f} Eh")
        results.append({"energy_hartree": energy, "fukui": fukui})

    if not results:
        print("No successful conformers -- nothing to report.")
        sys.exit(1)

    min_e = min(r["energy_hartree"] for r in results)
    kept = [r for r in results if (r["energy_hartree"] - min_e) * HARTREE_TO_KCAL <= ENERGY_WINDOW_KCAL]
    print(f"\n{len(kept)}/{len(results)} conformers within {ENERGY_WINDOW_KCAL} kcal/mol window")

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

    print(f"\n{'idx':>4} {'elem':>5} {'f+':>8} {'f-':>8} {'f0':>8}")
    csv_path = out_dir / "fukui_result.csv"
    with open(csv_path, "w") as fh:
        fh.write("atom_idx,element,f_plus,f_minus,f_zero\n")
        for idx, atom in enumerate(mol.GetAtoms()):
            print(f"{idx:>4} {atom.GetSymbol():>5} {f_plus[idx]:>8.4f} {f_minus[idx]:>8.4f} {f_zero[idx]:>8.4f}")
            fh.write(f"{idx},{atom.GetSymbol()},{f_plus[idx]:.5f},{f_minus[idx]:.5f},{f_zero[idx]:.5f}\n")

    print(f"\nWritten to {csv_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage: python fukui_single.py "<SMILES>" output_dir/')
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])