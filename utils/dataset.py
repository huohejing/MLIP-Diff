import os
import pickle
import lmdb
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Data
from utils.data import PDBProtein, parse_lig_file, parse_drug3d_mol


def to_torch_dict(data):
    output = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            output[k] = torch.from_numpy(v)
        else:
            output[k] = v
    return output

class ProteinLigandData(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def protein_ligand_dicts(protein_dict=None, ligand_dict=None, frag_dict=None, **kwargs):
        instance = ProteinLigandData(**kwargs)
        if protein_dict is not None:
            for k, v in protein_dict.items():
                instance["protein_" + k] = v
        if ligand_dict is not None:
            for k, v in ligand_dict.items():
                instance["ligand_" + k] = v
        if frag_dict is not None:
            for k, v in frag_dict.items():
                instance["frag_" + k] = v

        instance["ligand_nbh_list"] = {
            i.item(): [j.item() for k, j in enumerate(instance.ligand_bond_index[1]) 
            if instance.ligand_bond_index[0, k].item() == i] for i in instance.ligand_bond_index[0]
        }
        return instance

    def __inc__(self, key, value, *args, **kwargs):
        if key == "ligand_bond_index":
            return self["ligand_element"].size(0)
        else:
            return super().__inc__(key, value)

class ProteinLigandDataset(Dataset):
    def __init__(self, path, transform=None, version="final"):
        super().__init__()
        self.path = path.rstrip('/')
        self.index_path = os.path.join(self.path, "index.pkl")
        self.processed_path = os.path.join(
            os.path.dirname(self.path), os.path.basename(self.path) + f"_processed_{version}.lmdb"
        )
        self.transform = transform
        self.database = None
        self.keys = None

        if not os.path.exists(self.processed_path):
            print(f"{self.processed_path} does not exist, start to process data!")
            self._process()

    def _process(self):
        database = lmdb.open(
            self.processed_path, map_size=10*(1024*1024*1024),
            create=True, subdir=False, readonly=False,
        )
        with open(self.index_path, "rb") as f:
            index = pickle.load(f)
        
        num_skip = 0
        with database.begin(write=True, buffers=True) as db:
            for i, (pocket_fn, _, ligand_fn, _, logp, tpsa, sa, qed, aff, _) in enumerate(tqdm(index)):
                if pocket_fn is not None:
                    try:
                        pocket_dict = PDBProtein(os.path.join(self.path, pocket_fn)).to_dict_atom()
                        ligand_dict = parse_drug3d_mol(os.path.join(self.path, ligand_fn))
                        data = ProteinLigandData.protein_ligand_dicts(
                            protein_dict=to_torch_dict(pocket_dict),
                            ligand_dict=to_torch_dict(ligand_dict)
                        )
                        data.protein_filename = pocket_fn
                        data.ligand_filename = ligand_fn
                        data.logp = logp
                        data.tpsa = tpsa
                        data.sa = sa
                        data.qed = qed
                        data.aff = aff
                        data = data.to_dict()
                        db.put(key=str(i).encode(), value=pickle.dumps(data))
                    except:
                        num_skip += 1
                        print(f"Skipping {num_skip} {pocket_fn} {ligand_fn}!")
                        continue
                else:
                    continue
        database.close()

    def _build_db(self):
        assert self.database is None
        self.database = lmdb.open(
            self.processed_path, map_size=10*(1024*1024*1024), create=False, 
            subdir=False, readonly=True, lock=False, readahead=False, meminit=False,
        )
        with self.database.begin() as db:
            self.keys = list(db.cursor().iternext(values=False))

    def __len__(self):
        if self.database is None:
            self._build_db()
        return len(self.keys)

    def get_ori_data(self, idx):
        if self.database is None:
            self._build_db()
        key = self.keys[idx]
        data = pickle.loads(self.database.begin().get(key))
        data = ProteinLigandData(**data)
        data.id = idx
        assert data.protein_pos.size(0) > 0
        return data

    def __getitem__(self, idx):
        data = self.get_ori_data(idx)
        if self.transform is not None:
            data = self.transform(data)
        return data

def get_dataset(config, *args, **kwargs):
    if config.name == "protein_ligand":
        dataset = ProteinLigandDataset(config.path, *args, **kwargs)
    else:
        raise NotImplementedError(f"Unknown dataset name: {config.name}")
    
    if "split" in config:
        split = torch.load(config.split)
        subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
        return dataset, subsets
    else:
        return dataset, None
