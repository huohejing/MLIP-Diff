# PBR (PoseBusters)

Conformational plausibility of the generated molecules, scored with
[PoseBusters](https://github.com/maabuu/posebusters) `config='dock'`, which
excludes the RMSD-based tests — de novo molecules have no reference
conformation.

**PBR** is the fraction of molecules passing every check.

## Two steps

| Step | Script | Does |
|:---|:---|:---|
| 1 | `results8_pbr.py` | runs PoseBusters, writes one row per molecule |
| 2 | `build_pbr_paired.py` | pairs by SMILES, applies the water convention, computes gain/loss and McNemar |

The split matters. Step 1 records **all 22 checks, water included**: `pbr_pass`
is "all 22 passed" and `failed_checks` lists every failure. Step 2 is where
water is excluded — because it produces *both* the `nowater` and `full`
conventions from the same input. If step 1 filtered water out early, the `full`
convention could never be computed.

## Environment

Step 1 runs in the **`pb`** environment (PoseBusters 0.6.5). It needs a recent
RDKit — the shared `diffgui_cpu` environment has RDKit 2022.09, which lacks the
`GetProp(autoConvert)` argument PoseBusters 0.6.5 calls, and crashes.

Step 2 needs RDKit only, and imports the shared pairing helper from
`../eval_utils.py`.

```
conda activate pb
```

## Usage

```
# Step 1 — per target, over the named variants
python eval/pbr/results8_pbr.py --target 3ctj --variants base total split \
    --results-root /path/to/results8

# Step 2 — all targets at once
python eval/pbr/build_pbr_paired.py \
    --results-root /path/to/results8 --out-dir /path/to/out
```

Expected layout:

```
{results_root}/
├── {target}/native/{target}_protein.pdb    whole receptor — the clash checks need all of it
├── {target}/base/*.sdf                     unguided molecules
├── {target}/{total,split}/*.sdf            guided molecules
└── {target}/eval/                          ← step 1 output, step 2 output
```

`results_root` defaults to `$MLIPDIFF_RESULTS8`, then to `./results8`.

## Output

**Step 1** — `{target}/eval/{target}_{variant}_pbr.csv`

| Column | |
|:---|:---|
| `file` | molecule file name |
| `pbr_pass` | all 22 checks passed |
| `failed_checks` | `;`-joined names of every failed check, water included |

**Step 2** — paired detail `{target}/eval/{target}_{variant}_pbr_paired.csv`,
plus two summary tables in `--out-dir`:

| File | |
|:---|:---|
| `pbr_paired_nowater.csv` | the reported convention |
| `pbr_paired_full.csv` | all 22 checks, for supplementary material |

Columns: `n_paired`, `base_pass_rate`, `guided_pass_rate`, `delta_pp`,
`n_gain`, `n_loss`, `n_tie`, `better_rate`, `better_rate_discordant`,
`mcnemar_p`.

## The water convention

`nowater` passes a molecule when all 22 checks pass **or** its only failure is
`minimum_distance_to_waters`:

```python
WATER_CHECK = 'minimum_distance_to_waters'
```

DiffGui is trained without crystallographic water as a conditioning input, so the
model never sees a solvent shell and cannot learn to avoid one. Scoring with
water included would fail molecules for colliding with water they cannot see.
Only that one check is excluded; `volume_overlap_with_waters` still counts, so
`nowater` is not identical to "all water checks dropped". Changing this changes
reported numbers, so it is left exactly as it stands.

## Pairing

Both sides are matched by canonical SMILES with stereochemistry removed, then
zipped within each SMILES group in ascending numeric `mol_id` order
(`base[i]` ↔ `guided[i]`, stopping at the shorter side). The implementation is
`eval_utils.zip_by_smiles`, shared with the other indicators so that
cross-indicator comparisons use identical pairs. When the base group has fewer
instances of a SMILES than the guided group, the surplus guided instances are
reported as `n_unmatched_instances` rather than silently dropped.

## Note on the PoseBusters call

Step 1 passes **file paths** to `PoseBusters.bust()`, not RDKit molecule
objects. PoseBusters sorts its result table by `str(mol_pred)`
(`posebusters.py:263`); for a molecule object that is its memory address, so the
returned row order is unrelated to the input order and differs between runs.
Passing paths makes `str(mol_pred)` the path itself, which is stable, and the
`file` column can then be read straight out of each row.
