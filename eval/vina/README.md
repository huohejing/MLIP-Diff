# Vina

AutoDock Vina `score_only` scoring of the generated conformations, plus the
crystal-ligand baseline the BNC ratio is measured against.

## Environment

```
conda activate diffgui_cpu
```

Needs the `vina` Python package, `meeko` and `openbabel`. The `pb` environment
used for PBR does not have them.

## Scripts

| File | |
|:---|:---|
| `score.py` | scores a directory of generated SDFs, pairs them against the base group, writes CSV + JSON |
| `native_baseline.py` | scores the crystal ligand of each target, writes the baseline table |
| `docking_vina.py` | `VinaDockingTask` — the scoring engine, from DiffGui |
| `docking_qvina.py` | base docking helpers, required by `docking_vina.py` |

`score.py` is deliberately not called `vina.py`: Python puts the script's own
directory first on `sys.path`, so a local `vina.py` would shadow the `vina`
package that `docking_vina.py` imports.

## Usage

```
# generated molecules — guided vs base
python eval/vina/score.py --target 3ctj \
    --guided-dir /path/to/guided_sdf \
    --base-dir   /path/to/base_sdf \
    --pocket     /path/to/3ctj_pocket.pdb \
    --out-dir    /path/to/out

# crystal ligand baseline — incremental, never overwrites other targets
python eval/vina/native_baseline.py --targets 3ctj 1w51
```

`--base-dir` and `--pocket` fall back to per-target defaults resolved against
`$MLIPDIFF_ROOT` (default: the current directory), and both can be given
explicitly — the command above does. `--workers` defaults to 3.

`score.py` also takes `--discover`, which scans `<root>/data24` for every
`{target}_*_guided` directory and scores all of them.

## Protocol

Scoring is `score_only` against the given pocket: no re-docking, no
minimisation, the generated pose is left exactly as it is. This is what makes
the numbers comparable between the base and guided groups — both are scored on
their own conformations, not on a relaxed or re-docked one.

Scores with `|affinity| >= 100` kcal/mol are dropped as non-physical; a handful
of clashing molecules produce such values, and they would otherwise dominate any
mean.

## Output

`{out_dir}/vina_{target}_{variant}.csv`, one row per pair, full precision:

| Column | |
|:---|:---|
| `smiles` | canonical SMILES, stereochemistry removed |
| `base_vina`, `guided_vina` | score_only affinities, kcal/mol |
| `diff_kcal` | `guided - base`; negative means guidance improved the score |
| `guided_better` | `guided < base` |

and the matching `vina_{target}_{variant}.json` with the summary: `n_paired`,
`pct_better`, `mean_delta`, `median_delta` and the pairing `method` used.

`native_baseline.py` writes `{root}/validation/native_vina_baseline.csv` with
`target, native_vina_kcal, n_atoms`. It reads the existing file and updates only
the targets named on the command line, so scoring one target does not wipe the
others.

## Pairing

Base and guided molecules are matched by file name index when the two groups
agree on index and SMILES for more than 90 % of their common files — the usual
case, since both groups come from the same sampling run. Otherwise they fall
back to SMILES pairing. The method actually used is recorded in the JSON.
