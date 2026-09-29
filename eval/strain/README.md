# Strain

Conformational strain from a two-stage MACE geometry relaxation.

| Stage | |
|:---|:---|
| 1 | heavy atoms fixed (and the pocket, if given), hydrogens relaxed → `E_gen` |
| 2 | all ligand atoms relaxed (pocket still fixed) → `E_opt` |

`ΔE = E_gen - E_opt` is how far the generated conformation sits from the MACE
minimum. `RMSD` is the Kabsch-aligned heavy-atom displacement between the two.

## Requirements

```
pip install mace-torch ase
```

## Usage

```
# one molecule
python eval/strain/optimize.py --sdf mol.sdf --model ~/.cache/mace/MACE-OFF24_medium.model

# base and guided groups
python eval/strain/optimize.py --base base_sdf/ --guided guided_sdf/ \
    --model ~/.cache/mace/MACE-OFF24_medium.model

# in-pocket strain — ligand and pocket computed together, pocket fixed throughout
# give it the HYDROGENATED pocket, same as the MACE energy evaluation
python eval/strain/optimize.py --base base_sdf/ --guided guided_sdf/ \
    --pocket 3ctj_pocket_h.pdb --model ~/.cache/mace/MACE-OFF24_medium.model
```

`--device` defaults to `cuda`, `--steps2` to 300, `--max` limits the number of
molecules. `--output` sets the CSV path; otherwise it is written next to the
input.

## Output

One row per molecule: `sdf`, `success`, `n_heavy`, `n_pocket`, `E_gen_eV`,
`E_opt_eV`, `E_diff_eV`, `E_diff_kcal`, `E_diff_per_heavy_kcal`, `opt_rmsd_A`,
`steps_h`, `steps_all`, `converged`, `final_fmax`, `time_s`. With `--base` and
`--guided` the rows are the paired base/guided comparison.
