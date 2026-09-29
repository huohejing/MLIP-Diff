# MACE

MACE interaction energy of the generated conformations.

| | |
|:---|:---|
| `E_lig` | ligand alone |
| `E_total` | ligand + pocket |
| `E_int` | `E_total - E_lig - E_pocket` |

## Requirements

```
pip install mace-torch
```

The MACE-OFF24 checkpoint is from the
[MACE-OFF](https://github.com/ACEsuit/mace-off) repository. Default path is
`~/.cache/mace/MACE-OFF24_medium.model`; override with `--mace-model`.

## Usage

```
python eval/mace/energy.py --target 3ctj_split --root /path/to/project
```

`--device` defaults to CUDA if available. `--root` defaults to
`$MLIPDIFF_ROOT`, then to the current directory.

## Input

```
{root}/validation/{target}/
├── full_pocket.npz              pocket coords + elements
├── baseline/*_SDF/*.sdf         unguided molecules
└── scale_*/*_SDF/*.sdf          guided molecules
```

`full_pocket.npz` is produced by `scripts/prepare_target_mace.py`. Ligands are
hydrogenated inside the script with RDKit, `Chem.AddHs(mol, addCoords=True)`.

## Output

`{root}/validation/{target}_full_eval_off24.csv` — one row per base/guided pair:

| Column | |
|:---|:---|
| `group`, `smiles`, `n_heavy` | |
| `base_E_total_eV`, `guided_E_total_eV`, `diff_E_total_eV`, `E_total_better` | |
| `base_E_lig_eV`, `guided_E_lig_eV`, `diff_E_lig_eV`, `E_lig_better` | |
| `base_E_int_eV`, `guided_E_int_eV`, `diff_E_int_eV`, `E_int_better` | |

`diff_*` is `base - guided`, so positive means the guided molecule is lower in
energy. Base and guided molecules are paired by canonical SMILES, then zipped in
order within each SMILES group.
