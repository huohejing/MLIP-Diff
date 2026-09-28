# MM/GBSA Binding Energy Evaluation

Single-frame MM/GBSA scoring of generated ligand conformations, as described in
the manuscript's *MM/GBSA Binding Energy Evaluation* section.

## Method

Calculations are run **directly on the generated conformations** — with no
additional energy minimization and no molecular-dynamics sampling. A
**single-trajectory** protocol is used: the complex, receptor and ligand energies
are all derived from the same structural snapshot.

| Manuscript | Implementation |
|:---|:---|
| single-frame, no minimization or MD | `startframe=1, endframe=1, interval=1`; no minimization stage exists in the pipeline |
| single-trajectory protocol | complex, receptor and ligand parameters are all read against the same `com.inpcrd` snapshot |
| Amber framework, ff19SB for protein | `source leaprc.protein.ff19SB` |
| GAFF2 / AM1-BCC for ligand | `antechamber -c bcc -at gaff2`, then `parmchk2` |
| generalized Born, `igb=2` | `&gb igb=2` in `mmpbsa.in` |
| mbondi2 Born radii | `set default PBRadii mbondi2` in the tleap input |
| salt concentration 0.1 M | `saltcon=0.100` |
| no entropic contribution | no `&nmode` / `&entropy` block; only `VDWAALS`, `EEL`, `EGB`, `ESURF` are reported |
| failed calculations excluded | rows are written with status `FAILED` and dropped before analysis |

The reported ΔG is the `DELTA TOTAL` line of `FINAL.dat`, in kcal/mol.

Pairs are counted as improved when the guided member has the more favourable
estimated binding energy than its unguided partner of the same canonical SMILES.

## Requirements

- **AmberTools 20**, with `MMPBSA.py` version 14.0. AmberTools 24 is **not**
  usable — its `MMPBSA.py` is broken.
- `python3` with **RDKit**, used before `antechamber` for Kekulé-ization,
  hydrogen addition and formal-charge extraction.
- The scripts assume `AMBERHOME=$HOME/soft/amber20` and prepend
  `$AMBERHOME/bin` to `PATH`. Export a different `AMBERHOME` if yours differs.

## Directory Layout

Every path is relative to a single root, `$MMGBSA_ROOT`, which defaults to the
parent directory of `scripts/`. Override it with `MLIPDIFF_MMGBSA`.

```
$MMGBSA_ROOT/
├── scripts/                      these five scripts
├── proteins/
│   └── {target}_protein.pdb      INPUT — whole receptor, one per target
├── prep/{target}/                prep_protein.sh output
│   ├── protein_noh.pdb           cleaned receptor
│   ├── rec.prmtop, rec.inpcrd    receptor topology
│   ├── rec.lib                   receptor library (for reuse)
│   └── gap_report.txt            SEQRES-vs-ATOM missing-residue report
├── work/{target}/{group}/{mol_id}/    per-molecule intermediates
└── results/{target}/{group}/
    ├── {mol_id}.txt              one result row per molecule
    └── {group}_mmgbsa.csv        aggregated over the group
```

Ligand SDFs are read from any directory you point `run_group.sh` at, one file per
molecule.

## Usage

```
export MLIPDIFF_MMGBSA=/path/to/mmgbsa       # optional, defaults to scripts/..
export MLIPDIFF_PYTHON=/usr/bin/python3      # optional, a python with RDKit

# Step A — once per target
bash scripts/prep_protein.sh <target>

# Step B — one molecule
bash scripts/score_mol.sh <target> <group> <molecule.sdf>

# Step C — a whole group, 8 molecules in parallel at nice 10
bash scripts/run_group.sh <target> <group> <sdf_dir>
```

All three steps are re-runnable: results are written per molecule, and the group
CSV is rebuilt from the individual `.txt` rows.

### Pipeline

`prep_protein.sh` (Step A):

1. `clean_protein.py` replaces the broken `pdb4amber` — see Notes.
2. Disulfide bonds are detected by SG–SG distance < 2.5 Å and `CONECT` records
   are rebuilt from scratch.
3. Sequence gaps are reported to `gap_report.txt`. The report is informational
   and does not abort the run.
4. `tleap` builds the receptor topology under ff19SB with mbondi2 radii.

`score_mol.sh` (Step B):

1. RDKit: `SanitizeMol` → record the formal charge → `AddHs(addCoords=True)` →
   `Kekulize`, then write `lig_h.sdf`.
2. `antechamber` with AM1-BCC/GAFF2 at the recorded formal charge, then
   `parmchk2`.
3. `tleap`: receptor by `loadpdb`, ligand by `loadmol2`, `combine`, then save
   `com.prmtop` / `lig.prmtop`. Radii set to mbondi2.
4. `MMPBSA.py` single frame, `igb=2`, `saltcon=0.100`.
5. Parse the `DELTA` block of `FINAL.dat` into one CSV row.
6. Remove the bulky intermediate files, keeping `FINAL.dat` and the logs.

## Output

`{group}_mmgbsa.csv`, one row per molecule:

| Column | Meaning |
|:---|:---|
| `status` | `OK` or `FAILED` |
| `target`, `group`, `mol_id` | identifiers |
| `nc` | formal charge passed to `antechamber -nc` |
| `vdwaals`, `eel`, `egb`, `esurf` | the four MM/GBSA energy components, kcal/mol |
| `total` | ΔG, kcal/mol — empty for `FAILED` rows |

On failure the row is still written, with the failing stage recorded in the
`total` column (`rdkit`, `antechamber`, `tleap`, `mmpbsa` or `parse`), so every
excluded molecule remains accounted for.

## Notes

- **`pdb4amber` is broken in some AmberTools 20 builds** — its shebang points at
  a build-tree Python that does not exist. `clean_protein.py` reproduces the
  `-p --noter --most-populous -y` behaviour: keeps only standard residues
  (20 amino acids plus NME/ACE, HID/HIE/HIP, CYX and the protonation variants),
  resolves altlocs by highest occupancy, strips hydrogens so tleap rebuilds them
  from ff19SB templates, drops `TER` cards, rebuilds disulfide `CONECT` records,
  and maps MSE to MET.
- **The receptor must be loaded with `loadpdb`, not `loadoff`.** `saveoff` /
  `loadoff` do not carry coordinates, so a receptor restored that way loses its
  position relative to the ligand and the complex is meaningless.
- **The SDF must be Kekulé-ized and hydrogenated before `antechamber`.**
  `antechamber` does not recognize aromatic bond type 4 and reads aromatic
  carbons as trivalent, which makes `acdoctor` abort with "Weird atomic
  valence".
- **Ligand formal charge is not assumed to be zero.** It is read with RDKit and
  passed explicitly to `antechamber -nc`.
- **Do not `source $AMBERHOME/amber.sh` inside these scripts.** Under `set -u`
  it exits silently when `LD_LIBRARY_PATH`, `PERL5LIB` or `PYTHONPATH` are
  unset. The scripts export `AMBERHOME` and extend `PATH` directly instead.
- `check_pocket_align.py` verifies that a pocket PDB shares the coordinate frame
  of the receptor PDB, which is the precondition for scoring a pocket-derived
  pose against the whole-receptor topology:

  ```
  python3 scripts/check_pocket_align.py <pocket.pdb> <receptor.pdb> [tolerance_A]
  ```

  It passes when at least 99 % of pocket atoms are present in the receptor and
  the largest coordinate difference is within tolerance (0.01 Å by default).
