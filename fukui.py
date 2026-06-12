"""
fukui_indices.py
version 0.0.1 120626
================
Calculate Fukui indices (f+, f-, f0) for atoms in a drug molecule using PySCF.
GPU acceleration via gpu4pyscf is used when available, with automatic CPU fallback.

Fukui Index Theory:
-------------------
  f+(r)  = rho(N+1) - rho(N)   -> susceptibility to nucleophilic attack
  f-(r)  = rho(N)   - rho(N-1) -> susceptibility to electrophilic attack
  f0(r)  = 0.5 * (f+ + f-)     -> susceptibility to radical attack

  In condensed (atom-resolved) form, atomic electron populations q(N) are used:
  f+_A = q_A(N+1) - q_A(N)
  f-_A = q_A(N)   - q_A(N-1)
  f0_A = 0.5 * (f+_A + f-_A)

Usage:
------
  # From SMILES
  python fukui_indices.py --smiles "CC(=O)Oc1ccccc1C(=O)O" --name aspirin

  # From XYZ file
  python fukui_indices.py --xyz molecule.xyz

  # With geometry optimization
  python fukui_indices.py --smiles "CCO" --optimize

  # Full custom example
  python fukui_indices.py --smiles "c1ccccc1" --basis 6-311g* --functional M062X \
      --population lowdin --optimize --charge 0 --spin 0

Dependencies:
-------------
  pyscf, gpu4pyscf (optional), rdkit, numpy, pandas
  Install: pip install pyscf rdkit numpy pandas
           pip install gpu4pyscf-cuda12x   # for GPU support
"""

import argparse
import csv
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np

# ─────────────────────────────────────────────
# CONFIG: All tuneable parameters are defined here with descriptions
# ─────────────────────────────────────────────

# Supported DFT functionals with descriptions
FUNCTIONALS = {
    "B3LYP":  "Hybrid GGA; workhorse for organic molecules (default)",
    "M062X":  "Minnesota hybrid; good for non-covalent interactions",
    "wB97X-D":"Range-separated hybrid with dispersion; accurate for drug-like molecules",
    "PBE":    "Pure GGA; fast, good for large systems",
    "PBE0":   "Hybrid GGA; balanced accuracy/cost",
    "TPSS":   "Meta-GGA; good for transition metals",
    "B97-D3": "GGA with empirical dispersion correction",
}

# Supported basis sets with descriptions
BASIS_SETS = {
    "6-31G*":    "Pople split-valence + polarization; fast, standard for medium molecules (default)",
    "6-31+G*":   "Adds diffuse functions; better for anions/lone pairs",
    "6-311G**":  "Triple-zeta; more accurate, more costly",
    "6-311+G**": "Triple-zeta + diffuse + polarization; high accuracy",
    "cc-pVDZ":   "Dunning DZ; systematically improvable",
    "cc-pVTZ":   "Dunning TZ; high accuracy",
    "def2-SVP":  "Ahlrichs SV; efficient for larger molecules",
    "def2-TZVP": "Ahlrichs TZ; good accuracy/cost balance",
}

# Supported population analysis methods with descriptions
POPULATION_METHODS = {
    "mulliken": "Mulliken charges; fast, basis-set sensitive (default)",
    "lowdin":   "Löwdin charges; less basis-set sensitive, more robust",
}

# Default configuration
DEFAULTS = {
    "functional":  "B3LYP",
    "basis":       "6-31G*",
    "population":  "mulliken",
    "charge":      0,
    "spin":        None,   # None = auto-detect
    "optimize":    False,
    "output_dir":  ".",
}


# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────

def setup_logging(output_dir: str, mol_name: str) -> logging.Logger:
    """
    Configure logging to both console (INFO) and a timestamped log file (DEBUG).
    The log file captures full PySCF output, SCF convergence, and all intermediate values.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"fukui_{mol_name}_{timestamp}.log")

    logger = logging.getLogger("fukui")
    logger.setLevel(logging.DEBUG)

    # File handler: full DEBUG detail for diagnosing convergence issues
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    # Console handler: INFO-level summary only
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Log file: {log_path}")
    return logger


# ─────────────────────────────────────────────
# GPU DETECTION
# ─────────────────────────────────────────────

def get_dft_engine(logger: logging.Logger):
    """
    Attempt to import gpu4pyscf for GPU-accelerated DFT.
    Falls back to standard pyscf if gpu4pyscf is unavailable or no GPU present.

    Returns:
        (dft_module, using_gpu: bool)
    """
    try:
        import gpu4pyscf.dft as dft  # GPU-accelerated DFT
        # Quick check that CUDA is actually available
        import cupy
        cupy.array([1])  # Triggers CUDA init; raises if no GPU
        logger.info("GPU detected — using gpu4pyscf for DFT calculations")
        return dft, True
    except Exception as e:
        logger.info(f"GPU not available ({e}); falling back to CPU PySCF")
        from pyscf import dft
        return dft, False


# ─────────────────────────────────────────────
# MOLECULE BUILDING
# ─────────────────────────────────────────────

def mol_from_smiles(smiles: str, logger: logging.Logger) -> list:
    """
    Convert a SMILES string to a PySCF atom list via RDKit.
    RDKit adds implicit hydrogens and generates 3D coordinates using ETKDG.

    Returns:
        List of (element, (x, y, z)) tuples in Angstrom.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        logger.error("RDKit not installed. Run: pip install rdkit")
        sys.exit(1)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.error(f"Invalid SMILES: {smiles}")
        sys.exit(1)

    mol = Chem.AddHs(mol)  # Add explicit hydrogens
    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())  # 3D conformer generation
    if result != 0:
        logger.warning("ETKDG conformer generation failed; trying random coords")
        AllChem.EmbedMolecule(mol, randomSeed=42)

    AllChem.MMFFOptimizeMolecule(mol)  # Quick force-field pre-optimization
    conf = mol.GetConformer()
    atoms = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        atoms.append((atom.GetSymbol(), (pos.x, pos.y, pos.z)))

    logger.info(f"Molecule built from SMILES: {len(atoms)} atoms")
    logger.debug(f"Atom list: {atoms}")
    return atoms


def mol_from_xyz(xyz_path: str, logger: logging.Logger) -> list:
    """
    Parse a standard XYZ file into a PySCF atom list.

    XYZ format:
        Line 1: number of atoms
        Line 2: comment
        Lines 3+: element  x  y  z

    Returns:
        List of (element, (x, y, z)) tuples in Angstrom.
    """
    atoms = []
    with open(xyz_path) as f:
        lines = f.readlines()
    n_atoms = int(lines[0].strip())
    comment = lines[1].strip()
    logger.info(f"XYZ file: {xyz_path} | {n_atoms} atoms | comment: '{comment}'")
    for line in lines[2:2 + n_atoms]:
        parts = line.split()
        atoms.append((parts[0], (float(parts[1]), float(parts[2]), float(parts[3]))))
    logger.debug(f"Parsed atoms: {atoms}")
    return atoms


# ─────────────────────────────────────────────
# SPIN AUTO-DETECTION
# ─────────────────────────────────────────────

def auto_spin(n_electrons: int, logger: logging.Logger) -> int:
    """
    Auto-detect spin multiplicity (2S+1) from electron count.
      - Even electrons -> singlet (spin=0, i.e. 2S=0)
      - Odd electrons  -> doublet (spin=1, i.e. 2S=1)

    PySCF uses 'spin' = 2S (number of unpaired electrons).
    """
    spin = n_electrons % 2  # 0 for even, 1 for odd
    multiplicity = spin + 1
    logger.debug(f"Auto-spin: {n_electrons} electrons -> spin={spin} (multiplicity {multiplicity})")
    return spin


# ─────────────────────────────────────────────
# PySCF MOLECULE BUILDER
# ─────────────────────────────────────────────

def build_pyscf_mol(atoms: list, charge: int, spin: int,
                    basis: str, logger: logging.Logger):
    """
    Build a pyscf.gto.Mole object from atom list, charge, spin, and basis set.
    Verbose=3 in PySCF sends detailed SCF info to stdout; we redirect that to the log.
    """
    from pyscf import gto
    mol = gto.Mole()
    mol.atom = atoms
    mol.charge = charge
    mol.spin = spin       # 2S = number of unpaired electrons
    mol.basis = basis
    mol.verbose = 4       # Level 4 = detailed SCF convergence info in PySCF logs
    mol.output = None     # Will be captured via our logger below
    mol.build()

    n_elec = mol.nelectron
    logger.debug(f"PySCF mol built | charge={charge} spin={spin} "
                 f"n_electrons={n_elec} basis={basis} n_basis={mol.nao_nr()}")
    return mol


# ─────────────────────────────────────────────
# GEOMETRY OPTIMIZATION
# ─────────────────────────────────────────────

def optimize_geometry(mol, functional: str, dft_module, logger: logging.Logger):
    """
    Optionally optimize molecular geometry at the specified DFT level using
    PySCF's built-in geometric optimizer (requires the 'geometric' package).

    Returns an updated pyscf.gto.Mole with optimized coordinates.
    """
    logger.info(f"Optimizing geometry at {functional}/{mol.basis} ...")
    try:
        from pyscf.geomopt.geometric_solver import optimize
        mf = dft_module.RKS(mol)
        mf.xc = functional
        mol_opt = optimize(mf)
        logger.info("Geometry optimization converged")
        return mol_opt
    except ImportError:
        logger.error("Geometry optimization requires 'geometric': pip install geometric")
        sys.exit(1)


# ─────────────────────────────────────────────
# SCF SINGLE-POINT CALCULATION
# ─────────────────────────────────────────────

def run_scf(mol, functional: str, dft_module, label: str,
            logger: logging.Logger) -> object:
    """
    Run a DFT single-point calculation (RKS for closed-shell, UKS for open-shell).
    Uses GPU-accelerated DFT if dft_module is gpu4pyscf.dft, else standard pyscf.dft.

    Args:
        mol:        pyscf.gto.Mole object
        functional: DFT exchange-correlation functional string
        dft_module: pyscf.dft or gpu4pyscf.dft
        label:      Human-readable label for logging (e.g. "neutral (N)")

    Returns:
        Converged mean-field object (mf).
    """
    t0 = time.time()
    logger.info(f"Running SCF for {label} | charge={mol.charge} spin={mol.spin} "
                f"functional={functional} basis={mol.basis}")

    # Choose RKS (restricted, closed-shell) or UKS (unrestricted, open-shell)
    if mol.spin == 0:
        mf = dft_module.RKS(mol)
        logger.debug(f"{label}: Using RKS (closed-shell)")
    else:
        mf = dft_module.UKS(mol)
        logger.debug(f"{label}: Using UKS (open-shell, spin={mol.spin})")

    mf.xc = functional                  # Set exchange-correlation functional
    mf.grids.level = 3                  # Integration grid level: 0(coarse)–5(fine); 3 is balanced
    mf.conv_tol = 1e-9                  # SCF convergence threshold for energy (Hartree)
    mf.conv_tol_grad = 1e-6             # SCF convergence threshold for gradient
    mf.max_cycle = 200                  # Max SCF iterations before declaring non-convergence

    energy = mf.kernel()                # Run SCF; returns total energy in Hartree

    elapsed = time.time() - t0
    if mf.converged:
        logger.info(f"  {label}: SCF converged | E = {energy:.8f} Ha | time = {elapsed:.1f}s")
    else:
        logger.warning(f"  {label}: SCF DID NOT CONVERGE after {mf.max_cycle} cycles!")

    logger.debug(f"  {label}: HOMO = {_homo_energy(mf):.4f} Ha | "
                 f"LUMO = {_lumo_energy(mf):.4f} Ha")
    return mf


def _homo_energy(mf) -> float:
    """Extract HOMO energy from converged mean-field object."""
    mo_occ = mf.mo_occ
    mo_energy = mf.mo_energy
    if isinstance(mo_energy, np.ndarray):
        return float(mo_energy[mo_occ > 0][-1])
    # UKS returns (alpha, beta) arrays
    return float(mo_energy[0][mo_occ[0] > 0][-1])


def _lumo_energy(mf) -> float:
    """Extract LUMO energy from converged mean-field object."""
    mo_occ = mf.mo_occ
    mo_energy = mf.mo_energy
    if isinstance(mo_energy, np.ndarray):
        lumo = mo_energy[mo_occ == 0]
        return float(lumo[0]) if len(lumo) > 0 else float("nan")
    lumo = mo_energy[0][mo_occ[0] == 0]
    return float(lumo[0]) if len(lumo) > 0 else float("nan")


# ─────────────────────────────────────────────
# POPULATION ANALYSIS
# ─────────────────────────────────────────────

def get_atomic_populations(mf, method: str, logger: logging.Logger) -> np.ndarray:
    """
    Compute per-ATOM electron populations via Mulliken or Löwdin analysis.

    PySCF API clarification:
        mulliken_pop()  -> returns per-AO charges (shape: n_AOs), NOT per-atom.
        mulliken_charges() -> returns per-ATOM charges (shape: n_atoms). USE THIS.
        For Löwdin: use the 'meta-Löwdin' variant via mf.analyze() or manual contraction.

    We compute per-atom electron population as:
        population_A = Z_A - charge_A
    where charge_A is the net atomic charge from population analysis.

    Returns:
        np.ndarray of shape (n_atoms,) — electron population per atom.
    """
    mol = mf.mol

    # Move density matrix to CPU (gpu4pyscf returns cupy arrays; .get() converts to numpy)
    try:
        dm = mf.make_rdm1().get()
    except AttributeError:
        dm = mf.make_rdm1()

    # For UKS (open-shell), dm is (2, nao, nao); sum alpha+beta for total density
    if dm.ndim == 3:
        dm = dm[0] + dm[1]
        logger.debug("  UKS: summing alpha+beta density matrices for total DM")

    ovlp = mol.intor("int1e_ovlp")   # AO overlap matrix S, shape (nao, nao)

    if method == "mulliken":
        # mulliken_charges contracts AO-level populations to per-atom charges.
        # Internally: pop_AO = diag(D * S), then summed per atom by shell offsets.
        # Returns: charges shape (n_atoms,)  where charge_A = Z_A - sum_{mu in A} D_mu_mu * S_mu_mu
        from pyscf.scf import hf as scf_hf
        _, charges = scf_hf.mulliken_pop(mol, dm, ovlp, verbose=0)

    elif method == "lowdin":
        # Löwdin: orthogonalize AOs via S^{-1/2}, then contract to atoms.
        # We use numpy directly since PySCF's meta-Löwdin helper is internal.
        # Step 1: compute S^{-1/2} via eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(ovlp)
        eigvals = np.maximum(eigvals, 1e-15)          # avoid sqrt of near-zero
        s_half = eigvecs * np.sqrt(eigvals)            # S^{1/2}
        s_inv_half = eigvecs / np.sqrt(eigvals)        # S^{-1/2}
        # Step 2: transform DM to Löwdin basis: D' = S^{1/2} D S^{1/2}
        dm_lowdin = s_half.T @ dm @ s_half
        # Step 3: per-AO population in orthogonal basis = diagonal of D'
        ao_pops = np.diag(dm_lowdin)                   # shape (nao,)
        # Step 4: contract AO populations to atoms using PySCF's shell-to-atom mapping
        charges = np.zeros(mol.natm)
        ao_labels = mol.ao_labels(fmt=None)            # list of (atom_idx, shell, l, m)
        for ao_idx, (atom_idx, *_) in enumerate(ao_labels):
            charges[atom_idx] += ao_pops[ao_idx]       # accumulate electron pop per atom
        # Convert electron population -> charge: charge_A = Z_A - pop_A
        nuclear_charges = np.array([mol.atom_charge(i) for i in range(mol.natm)])
        populations = charges                           # already electron populations here
        charges = nuclear_charges - populations        # for logging consistency
        logger.debug(f"  Löwdin AO pops summed to {mol.natm} atoms")
        nuclear_charges = np.array([mol.atom_charge(i) for i in range(mol.natm)])
        populations = nuclear_charges - charges
        logger.debug(f"  Atomic charges ({method}): {np.round(charges,     4)}")
        logger.debug(f"  Electron pops  ({method}): {np.round(populations, 4)}")
        return populations

    else:
        raise ValueError(f"Unknown population method: {method}")

    # For Mulliken path: convert per-atom charges -> electron populations
    # charge_A = Z_A - q_A  =>  q_A = Z_A - charge_A
    nuclear_charges = np.array([mol.atom_charge(i) for i in range(mol.natm)])
    populations = nuclear_charges - np.array(charges)

    logger.debug(f"  n_atoms={mol.natm} | n_AOs={mol.nao_nr()}")
    logger.debug(f"  Atomic charges ({method}): {np.round(charges,     4)}")
    logger.debug(f"  Electron pops  ({method}): {np.round(populations, 4)}")
    return populations


# ─────────────────────────────────────────────
# FUKUI INDEX CALCULATION
# ─────────────────────────────────────────────

def compute_fukui(pop_neutral: np.ndarray,
                  pop_cation: np.ndarray,
                  pop_anion: np.ndarray,
                  logger: logging.Logger) -> dict:
    """
    Compute condensed Fukui indices from atomic electron populations.

    Condensed (finite-difference) approximation:
      f+_A = q_A(N+1) - q_A(N)    [anion minus neutral]
      f-_A = q_A(N)   - q_A(N-1)  [neutral minus cation]
      f0_A = 0.5*(f+_A + f-_A)    [average]

    Args:
        pop_neutral: populations for N-electron system
        pop_cation:  populations for (N-1)-electron system
        pop_anion:   populations for (N+1)-electron system

    Returns:
        dict with keys 'f_plus', 'f_minus', 'f_zero' each as np.ndarray
    """
    f_plus  = pop_anion   - pop_neutral   # nucleophilic attack susceptibility
    f_minus = pop_neutral - pop_cation    # electrophilic attack susceptibility
    f_zero  = 0.5 * (f_plus + f_minus)   # radical attack susceptibility

    logger.debug(f"f+:  {np.round(f_plus,  4)}")
    logger.debug(f"f-:  {np.round(f_minus, 4)}")
    logger.debug(f"f0:  {np.round(f_zero,  4)}")

    # Sanity checks: Fukui indices should roughly sum to 1 (or -1 for f-)
    logger.debug(f"Sum f+={f_plus.sum():.4f} (should be ~1.0)")
    logger.debug(f"Sum f-={f_minus.sum():.4f} (should be ~1.0)")
    logger.debug(f"Sum f0={f_zero.sum():.4f} (should be ~1.0)")

    return {"f_plus": f_plus, "f_minus": f_minus, "f_zero": f_zero}


# ─────────────────────────────────────────────
# OUTPUT: CONSOLE + CSV
# ─────────────────────────────────────────────

def print_results(mol, fukui: dict, logger: logging.Logger):
    """Pretty-print Fukui indices to console."""
    print("\n" + "=" * 60)
    print(f"{'Atom':<6} {'Index':<6} {'Element':<8} {'f+ (nucl)':<12} {'f- (elec)':<12} {'f0 (rad)':<10}")
    print("-" * 60)
    for i in range(mol.natm):
        elem = mol.atom_symbol(i)
        fp   = fukui["f_plus"][i]
        fm   = fukui["f_minus"][i]
        fz   = fukui["f_zero"][i]
        print(f"{i:<6} {i:<6} {elem:<8} {fp:<12.5f} {fm:<12.5f} {fz:<10.5f}")
    print("=" * 60)
    print("f+ -> nucleophilic attack | f- -> electrophilic attack | f0 -> radical attack\n")


def save_csv(mol, fukui: dict, output_dir: str, mol_name: str, logger: logging.Logger):
    """Save Fukui indices to a CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"fukui_{mol_name}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["atom_index", "element", "f_plus", "f_minus", "f_zero"])
        for i in range(mol.natm):
            writer.writerow([
                i,
                mol.atom_symbol(i),
                round(float(fukui["f_plus"][i]),  6),
                round(float(fukui["f_minus"][i]), 6),
                round(float(fukui["f_zero"][i]),  6),
            ])
    logger.info(f"CSV saved: {csv_path}")


# ─────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Fukui indices (f+, f-, f0) for a drug molecule using PySCF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # ── Input ──────────────────────────────────────────────────────────────
    inp = parser.add_mutually_exclusive_group(required=True)
    inp.add_argument("--smiles", type=str,
                     help="SMILES string of the molecule (e.g. 'CC(=O)Oc1ccccc1C(=O)O')")
    inp.add_argument("--xyz", type=str,
                     help="Path to an XYZ coordinate file")

    parser.add_argument("--name", type=str, default="molecule",
                        help="Molecule name used for output file naming (default: 'molecule')")

    # ── DFT settings ───────────────────────────────────────────────────────
    parser.add_argument("--functional", type=str, default=DEFAULTS["functional"],
                        choices=list(FUNCTIONALS.keys()),
                        help="DFT exchange-correlation functional.\n" +
                             "\n".join(f"  {k}: {v}" for k, v in FUNCTIONALS.items()))

    parser.add_argument("--basis", type=str, default=DEFAULTS["basis"],
                        choices=list(BASIS_SETS.keys()),
                        help="Gaussian basis set.\n" +
                             "\n".join(f"  {k}: {v}" for k, v in BASIS_SETS.items()))

    parser.add_argument("--population", type=str, default=DEFAULTS["population"],
                        choices=list(POPULATION_METHODS.keys()),
                        help="Atomic population analysis method.\n" +
                             "\n".join(f"  {k}: {v}" for k, v in POPULATION_METHODS.items()))

    # ── Charge & Spin ──────────────────────────────────────────────────────
    parser.add_argument("--charge", type=int, default=DEFAULTS["charge"],
                        help="Total charge of the neutral molecule (default: 0)")

    parser.add_argument("--spin", type=int, default=DEFAULTS["spin"],
                        help="Spin (2S) for the neutral molecule. Default: auto-detect "
                             "(0 for even electrons, 1 for odd)")

    parser.add_argument("--anion-spin", type=int, default=None,
                        help="Override spin (2S) for anion (N+1). Default: auto-detect")

    parser.add_argument("--cation-spin", type=int, default=None,
                        help="Override spin (2S) for cation (N-1). Default: auto-detect")

    # ── Geometry optimization ──────────────────────────────────────────────
    parser.add_argument("--optimize", action="store_true", default=DEFAULTS["optimize"],
                        help="Optimize geometry at the chosen DFT level before "
                             "Fukui calculations (requires 'geometric' package)")

    # ── Output ─────────────────────────────────────────────────────────────
    parser.add_argument("--output-dir", type=str, default=DEFAULTS["output_dir"],
                        help="Directory for CSV and log file output (default: current dir)")

    return parser.parse_args()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    # 1. Setup logging
    logger = setup_logging(args.output_dir, args.name)
    logger.info("=" * 50)
    logger.info(f"Fukui Index Calculation | {args.name}")
    logger.info(f"Functional: {args.functional} | Basis: {args.basis} | "
                f"Population: {args.population}")
    logger.info("=" * 50)

    # 2. Get DFT engine (GPU or CPU)
    dft_module, using_gpu = get_dft_engine(logger)

    # 3. Build atom list from input
    if args.smiles:
        atoms = mol_from_smiles(args.smiles, logger)
    else:
        atoms = mol_from_xyz(args.xyz, logger)

    # 4. Build neutral PySCF molecule (charge=N, user-specified or default)
    #    First build a temp mol to count electrons for auto-spin
    from pyscf import gto
    tmp = gto.Mole()
    tmp.atom = atoms
    tmp.charge = args.charge
    tmp.spin = 0
    tmp.basis = args.basis
    tmp.verbose = 0
    tmp.build()
    n_elec_neutral = tmp.nelectron

    # Determine spins (auto or user-specified)
    spin_neutral = args.spin if args.spin is not None else auto_spin(n_elec_neutral, logger)
    spin_cation  = args.cation_spin if args.cation_spin is not None else auto_spin(n_elec_neutral - 1, logger)
    spin_anion   = args.anion_spin  if args.anion_spin  is not None else auto_spin(n_elec_neutral + 1, logger)

    logger.info(f"Electron counts -> N={n_elec_neutral} | N-1={n_elec_neutral-1} | N+1={n_elec_neutral+1}")
    logger.info(f"Spins (2S)      -> neutral={spin_neutral} | cation={spin_cation} | anion={spin_anion}")

    # 5. Build all three PySCF molecules
    mol_neutral = build_pyscf_mol(atoms, args.charge,     spin_neutral, args.basis, logger)
    mol_cation  = build_pyscf_mol(atoms, args.charge + 1, spin_cation,  args.basis, logger)
    mol_anion   = build_pyscf_mol(atoms, args.charge - 1, spin_anion,   args.basis, logger)

    # 6. Optional geometry optimization on the neutral molecule
    if args.optimize:
        mol_neutral = optimize_geometry(mol_neutral, args.functional, dft_module, logger)
        # Use optimized geometry for ion calculations too
        opt_atoms = [(mol_neutral.atom_symbol(i), mol_neutral.atom_coord(i, unit="Ang"))
                     for i in range(mol_neutral.natm)]
        mol_cation = build_pyscf_mol(opt_atoms, args.charge + 1, spin_cation,  args.basis, logger)
        mol_anion  = build_pyscf_mol(opt_atoms, args.charge - 1, spin_anion,   args.basis, logger)

    # 7. Run SCF for neutral, cation, anion
    mf_neutral = run_scf(mol_neutral, args.functional, dft_module, "neutral (N)",    logger)
    mf_cation  = run_scf(mol_cation,  args.functional, dft_module, "cation  (N-1)",  logger)
    mf_anion   = run_scf(mol_anion,   args.functional, dft_module, "anion   (N+1)",  logger)

    # 8. Population analysis -> atomic electron populations
    pop_neutral = get_atomic_populations(mf_neutral, args.population, logger)
    pop_cation  = get_atomic_populations(mf_cation,  args.population, logger)
    pop_anion   = get_atomic_populations(mf_anion,   args.population, logger)

    # 9. Compute Fukui indices
    fukui = compute_fukui(pop_neutral, pop_cation, pop_anion, logger)

    # 10. Output results
    print_results(mol_neutral, fukui, logger)
    save_csv(mol_neutral, fukui, args.output_dir, args.name, logger)

    logger.info("Fukui calculation complete.")


if __name__ == "__main__":
    main()
