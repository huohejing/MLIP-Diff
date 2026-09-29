# MM/GBSA

Single-frame MM/GBSA binding energy of the generated ligand conformations.

No energy minimization and no MD: the complex, receptor and ligand energies are
all derived from the same structure. Protein ff19SB, ligand GAFF2/AM1-BCC,
`igb=2` with `mbondi2` radii, salt 0.1 M, no entropy term.

## Requirements

- AmberTools, with `MMPBSA.py`
- `python3` with RDKit

The scripts use `$AMBERHOME`; export it if yours is not already set.

## Usage

```
export MLIPDIFF_MMGBSA=/path/to/mmgbsa     # optional, defaults to ~/mmgbsa
export MLIPDIFF_PYTHON=/usr/bin/python3    # optional, a python with RDKit

bash eval/mmgbsa/prep_protein.sh <target>                  # once per target
bash eval/mmgbsa/run_group.sh  <target> <group> <sdf_dir>  # a whole group
bash eval/mmgbsa/score_mol.sh  <target> <group> <mol.sdf>  # a single molecule
```

Every step is re-runnable; the group table is rebuilt from the per-molecule rows.

## Layout

```
$MMGBSA_ROOT/                       default: ~/mmgbsa
├── proteins/{target}_protein.pdb   INPUT — whole receptor
├── prep/{target}/                  prep_protein.sh output
├── work/{target}/{group}/          intermediates
└── results/{target}/{group}/       results
```

## Output

`results/{target}/{group}/{group}_mmgbsa.csv`:

| Column | |
|:---|:---|
| `status` | `OK` or `FAILED` |
| `target`, `group`, `mol_id` | |
| `nc` | ligand formal charge passed to antechamber |
| `vdwaals`, `eel`, `egb`, `esurf` | energy components, kcal/mol |
| `total` | ΔG, kcal/mol; on failure, the stage that failed |

## Check

```
python3 eval/mmgbsa/check_pocket_align.py <pocket.pdb> <receptor.pdb>
```

Verifies the pocket shares the receptor's coordinate frame. Exits non-zero if it
does not.
