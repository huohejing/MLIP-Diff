#!/usr/bin/env python3
"""
Target preparation: build every pocket-derived file a guided run needs.

A guided run uses the pocket twice, for two different purposes, and the two
uses need different files:

  1. DiffGui conditioning  — the pocket as it appears in the crystal structure,
                             heavy atoms only, no hydrogens.
  2. MACE protein environment — the same residues with missing atoms completed
                             and hydrogens added. Polar hydrogens fix the
                             hydrogen-bond geometry that the guidance force is
                             computed from, so this file must be protonated.

This script produces both from one invocation, plus the flat atom array the
energy-evaluation scripts read:

    {name}_pocket.pdb     pocket residues within <radius> Å of the reference
                          ligand, heavy atoms as they appear in the input
                          -> model.target

    {name}_pocket_h.pdb   the same residues after PDBFixer: missing atoms and
                          residues completed, non-standard residues replaced,
                          heterogens removed, hydrogens added at --ph
                          -> physical_guidance.pocket_pdb

    full_pocket.npz       coords (N,3) float32 + elements (N,) int32 of every
                          atom in the pocket, for the MACE energy evaluation
                          scripts (full_eval*.py, elig_precise.py)

Usage:
    python scripts/prepare_target_mace.py \
        --protein 1w51_protein.pdb \
        --ligand  1w51_ligand.sdf \
        --name    1w51 \
        --outdir  sample

Hydrogenation requires PDBFixer (OpenMM). If it is missing this script stops
with a clear error — it will not silently emit a non-hydrogenated file under
the _h name, because a pocket without polar hydrogens yields wrong guidance
forces and the failure would otherwise go unnoticed.

    pip install pdbfixer
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data import PDBProtein


ELEMENT_MAP = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16, 'P': 15,
               'F': 9, 'Cl': 17, 'Br': 35, 'I': 53}


def read_sdf_coords(sdf_path):
    """Read heavy-atom coordinates from an SDF file without RDKit."""
    with open(sdf_path) as f:
        lines = f.readlines()
    num_atoms = int(lines[3].strip()[:3])
    coords = []
    for i in range(4, 4 + num_atoms):
        parts = lines[i].split()
        coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.array(coords, dtype=np.float32)


def extract_pocket(protein_pdb, ligand_sdf, radius, outdir, name):
    """Pocket residues within <radius> Å of the ligand -> {name}_pocket.pdb."""
    protein = PDBProtein(protein_pdb)
    ligand = {"pos": read_sdf_coords(ligand_sdf)}
    residues = protein.query_residues_ligand(ligand, radius)
    if len(residues) == 0:
        raise ValueError(f"No protein residues within {radius} Å of the ligand — "
                         f"check that the protein and ligand are in the same "
                         f"coordinate frame.")

    block = protein.residues_to_pdb_block(residues, name="pocket")
    path = os.path.join(outdir, f"{name}_pocket.pdb")
    with open(path, "w") as f:
        f.write(block)

    print(f"  [1/3] {len(residues)} residues within {radius} Å -> {path}")
    return path, len(residues)


def hydrogenate_pocket(pocket_pdb, out_path, ph=7.0):
    """Complete missing atoms and add hydrogens with PDBFixer."""
    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
    except ImportError as e:
        raise SystemExit(
            "\n[ERROR] PDBFixer is required to build the hydrogenated pocket.\n"
            "        Install it with:  pip install pdbfixer\n"
            f"        (underlying import error: {e})\n"
            "        The hydrogens are not optional — MACE guidance needs them "
            "for correct hydrogen-bond geometry,\n"
            "        so this script stops rather than emitting a file that "
            "looks prepared but is not.\n"
        )

    fixer = PDBFixer(filename=pocket_pdb)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)

    with open(out_path, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)

    n_atoms = sum(1 for _ in open(out_path)
                  if _.startswith(("ATOM", "HETATM")))
    print(f"  [2/3] completed + hydrogenated at pH {ph} -> {out_path} "
          f"({n_atoms} atoms)")
    return out_path


def extract_mace_pocket(pocket_pdb, outdir):
    """Every atom of the pocket as flat arrays for the evaluation scripts.

    Reads the NON-hydrogenated pocket: the evaluation scripts combine this
    array with the MACE-derived hydrogen positions themselves, and adding
    hydrogens here would double-count them.
    """
    coords, elements = [], []
    with open(pocket_pdb) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            coords.append([float(line[30:38]), float(line[38:46]),
                           float(line[46:54])])
            name = line[12:16].strip()
            elem = name[0] if name and not name[0].isdigit() else name[1:2]
            elements.append(ELEMENT_MAP.get(elem, 6))

    if not coords:
        raise ValueError(f"No atoms parsed from {pocket_pdb}")

    path = os.path.join(outdir, "full_pocket.npz")
    np.savez(path,
             coords=np.array(coords, dtype=np.float32),
             elements=np.array(elements, dtype=np.int32))
    print(f"  [3/3] {len(coords)} pocket atoms -> {path}")
    return path


def main():
    p = argparse.ArgumentParser(
        description="Prepare a target: non-hydrogenated and hydrogenated pockets")
    p.add_argument("--protein", required=True, help="Full protein PDB")
    p.add_argument("--ligand", required=True, help="Reference ligand SDF")
    p.add_argument("--name", required=True,
                   help="Target name used in output filenames, e.g. 1w51")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--pocket_radius", type=float, default=10.0,
                   help="Pocket radius in Å (default: 10)")
    p.add_argument("--ph", type=float, default=7.0,
                   help="pH for hydrogenation (default: 7.0)")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Protein: {args.protein}")
    print(f"Ligand:  {args.ligand}")
    print(f"Output:  {args.outdir}")
    print()

    pocket_pdb, _ = extract_pocket(args.protein, args.ligand,
                                   args.pocket_radius, args.outdir, args.name)
    hydrogenate_pocket(pocket_pdb,
                       os.path.join(args.outdir, f"{args.name}_pocket_h.pdb"),
                       args.ph)
    extract_mace_pocket(pocket_pdb, args.outdir)

    print("\nDone. Point the sampling config at:")
    print(f"  model.target                        {args.outdir}/{args.name}_pocket.pdb")
    print(f"  physical_guidance.pocket_pdb        {args.outdir}/{args.name}_pocket_h.pdb")


if __name__ == "__main__":
    main()
