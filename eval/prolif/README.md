# ProLIF

Protein–ligand interaction counts for the generated conformations.

| Script | |
|:---|:---|
| `hbond.py` | hydrogen bonds |
| `hydrophobic.py` | hydrophobic contacts |

## Requirements

```
pip install prolif
```

## Usage

```
python eval/prolif/hbond.py       --target 3ctj --variants base total split
python eval/prolif/hydrophobic.py --target 3ctj --variants base total split
```

`--workers` defaults to 4. `--results-root` defaults to `$MLIPDIFF_RESULTS8`,
then to `./results8`.

## Input

```
{results_root}/{target}/
├── native/{target}_pocket_h.pdb    hydrogenated pocket (PDBFixer)
├── native/{target}_pocket.pdb      heavy-atom pocket
├── base/*.sdf                      unguided molecules
└── {total,split}/*.sdf             guided molecules
```

`hbond.py` uses the hydrogenated pocket `{target}_pocket_h.pdb` — polar
hydrogens fix the hydrogen-bond geometry. `hydrophobic.py` uses the heavy-atom
pocket `{target}_pocket.pdb`, since a hydrophobic contact is a heavy-atom
distance and needs no hydrogens.

Ligands are hydrogenated inside the scripts with RDKit,
`Chem.AddHs(mol, addCoords=True)`.

## Output

`{results_root}/{target}/eval/{target}_{variant}_{name}.csv`, column `file,n` —
one count per molecule.
