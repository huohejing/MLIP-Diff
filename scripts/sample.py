import os
import re
import sys
import shutil
import argparse
sys.path.append('.')

import torch
import numpy as np
from scipy import spatial
import torch.utils.tensorboard
from easydict import EasyDict
from rdkit import Chem

from torch_scatter import scatter_sum
from torch_geometric.data import Batch
from models.model import DiffGui
from models.bond_predictor import BondPredictor
from utils.sample_utils import seperate_outputs
from torch_geometric.transforms import Compose
from utils.diffgui_metrics.atom_num_config import CONFIG
from utils.data import PDBProtein, parse_drug3d_mol, parse_lig_file
from utils.dataset import to_torch_dict, get_dataset, ProteinLigandData
from utils.diffgui_metrics import scoring_func
# Vina removed - use separate batch scoring if needed
from utils.transforms import *
from utils.misc import *
from utils.reconstruct import *
from models.physical_guidance import init_mace_calculator


def print_pool_status(pool, logger):
    logger.info('[Pool] Finished %d | Failed %d' % (
        len(pool.finished), len(pool.failed)
    ))

def data_exists(data, prevs):
    for other in prevs:
        if len(data.logp_history) == len(other.logp_history):
            if (data.ligand_context_element == other.ligand_context_element).all().item() and \
                (data.ligand_context_feature_full == other.ligand_context_feature_full).all().item() and \
                torch.allclose(data.ligand_context_pos, other.ligand_context_pos):
                return True
    return False

def get_pocket_size(pocket_pos):
    aa_dist = spatial.distance.pdist(pocket_pos, metric="euclidean")
    aa_dist_sort = np.sort(aa_dist)[::-1]
    return np.median(aa_dist_sort[:10])

def get_bin_idx(pocket_size):
    bounds = CONFIG["bounds"]
    for i in range(len(bounds)):
        if bounds[i] > pocket_size:
            return i
    return len(bounds)

def sample_atom_num(pocket_size):
    bin_idx = get_bin_idx(pocket_size)
    num_atom_list, prob_list = CONFIG["bins"][bin_idx]
    atom_num = np.random.choice(num_atom_list, p=prob_list)
    return atom_num

def pdb_to_pocket(pocket_pdb_path, ligand_sdf_path, frag_sdf_path):
    pocket_dict = PDBProtein(pocket_pdb_path).to_dict_atom()
    if ligand_sdf_path != 'None':
        ligand_dict = parse_lig_file(ligand_sdf_path)
    else:
        ligand_dict={
            "element": torch.empty([0, ], dtype=torch.long),
            "hybridization": torch.empty([0, ], dtype=torch.long),
            "pos": torch.empty([0, 3], dtype=torch.float),
            "bond_index": torch.empty([2, 0], dtype=torch.long),
            "bond_type": torch.empty([0, ], dtype=torch.long),
            "atom_feature": torch.empty([0, 8], dtype=torch.float),
        }
    if frag_sdf_path != 'None':
        frag_dict = parse_lig_file(frag_sdf_path)
        data = ProteinLigandData.protein_ligand_dicts(
        protein_dict=to_torch_dict(pocket_dict),
        ligand_dict=to_torch_dict(ligand_dict),
        frag_dict=to_torch_dict(frag_dict)
    )
    else:
        data = ProteinLigandData.protein_ligand_dicts(
        protein_dict=to_torch_dict(pocket_dict),
        ligand_dict=to_torch_dict(ligand_dict)
    )

    return data

def main(args):
    # # Load configs
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    # The seed is exactly what is configured — it must not depend on where the
    # output is written. (Upstream DiffGui added sum(ord(c) for c in outdir) to
    # the configured seed, so the same config produced different molecules in
    # different output directories. To reproduce a run made with that scheme,
    # pass the already-offset value explicitly via --seed.)
    seed_all(args.seed if args.seed is not None else config.sample.seed)
    # load ckpt and train config
    import easydict
    try:
        torch.serialization.add_safe_globals([easydict.EasyDict])
    except:
        pass  # PyTorch < 2.4 doesn't have add_safe_globals
    ckpt = torch.load(config.model.checkpoint, map_location=torch.device('cpu'), weights_only=False)
    train_config = ckpt['config']

    # # Logging
    log_root = args.outdir.replace('outputs', 'outputs_vscode') if sys.argv[0].startswith('/data') else args.outdir
    log_dir = get_new_log_dir(log_root, prefix=config_name)
    #log_dir = args.logdir
    logger = get_logger('sample', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))

    # # Transform
    logger.info('Loading data placeholder...')
    ligand_atom_mode = ckpt["config"].data.transform.ligand_atom_mode
    if config.model.gen_mode == 'denovo':
        featurizer = FeatureComplex(ligand_atom_mode, sample=config.sample.sample)
    else:
        featurizer = FeatureComplexWithFrag(ligand_atom_mode, sample=config.sample.sample)
    transform = Compose([
        featurizer,
    ])
    max_size = None
    add_edge = getattr(config.sample, 'add_edge', None)
    
    # # Model
    logger.info('Loading diffusion model...')
    if train_config.model.name == 'diffgui':
        model = DiffGui(
                    config=train_config.model,
                    protein_node_types=featurizer.protein_feat_dim,
                    ligand_node_types=featurizer.atom_feat_dim,
                    num_edge_types=featurizer.bond_feat_dim,
                ).to(args.device)
    else:
        raise NotImplementedError('Model %s not implemented' % train_config.model.name)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    # label
    logp = torch.tensor([float(config.model.logp)], device=args.device).unsqueeze(-1)
    tpsa = torch.tensor([float(config.model.tpsa)], device=args.device).unsqueeze(-1)
    sa = torch.tensor([float(config.model.sa)], device=args.device).unsqueeze(-1)
    qed = torch.tensor([float(config.model.qed)], device=args.device).unsqueeze(-1)
    aff = torch.tensor([float(config.model.aff)], device=args.device).unsqueeze(-1)
    batch_lab = torch.cat((logp, tpsa, sa, qed, aff), dim=1)

    batch_size = config.sample.batch_size
    batch_lab = torch.tensor([list(batch_lab[0]) for _ in range(batch_size)]).to(args.device)

    # # Bond predictor and guidance
    if 'bond_predictor' in config:
        logger.info('Building bond predictor...')
        ckpt_bond = torch.load(config.bond_predictor, map_location=args.device)
        bond_predictor = BondPredictor(
            config=ckpt_bond['config']['model'],
            protein_node_types=featurizer.protein_feat_dim,
            ligand_node_types=featurizer.atom_feat_dim,
            num_edge_types=featurizer.bond_feat_dim
        ).to(args.device)
        bond_predictor.load_state_dict(ckpt_bond['model'])
        bond_predictor.eval()
    else:
        bond_predictor = None
    if 'guidance' in config.sample:
        guidance = config.sample.guidance  # tuple: (guidance_type[entropy/uncertainty], guidance_scale)
    else:
        guidance = None

    # Load pocket or test set data
    if config.sample.mode == 'pocket':
        data = pdb_to_pocket(config.model.target, config.model.ligand, config.model.frag)
        data = transform(data)
        data_list = []
        data_list.append(data)
    elif config.sample.mode == 'test':
        dataset, subsets = get_dataset(
            config = config.data,
            transform = transform,
        )
        data_list = subsets['test']
        logger.info(f'Test dataset: {len(data_list)}.')
    else:
        raise NotImplementedError('Sample mode should be pocket or test!')

    # prepare batch
    data_length = len(data_list)
    for i in tqdm(range(data_length), desc='Sample'):
        data = data_list[i]
        if config.sample.mode == 'pocket':
            # Extract PDB code from target path directory (e.g. .../1w51/pocket.pdb → 1w51)
            parts = config.model.target.replace('\\', '/').split('/')
            name = parts[-2] if len(parts) >= 2 else parts[-1].split('_')[0]
            logger.info(f'Protein Pocket: {config.model.target}')
            logger.info(f'Reference Ligand: {config.model.ligand}')
            logger.info(f'Optimization Fragment: {config.model.frag}')
        elif config.sample.mode == 'test':
            if config.data.dataset == 'pdbbind':
                name = data.protein_filename.split('/')[0]
            elif config.data.dataset == 'crossdocked':
                name = data.protein_filename.split('.')[0]
            logger.info(f'Protein Pocket: {data.protein_filename}.')

        pool = EasyDict({
            'failed': [],
            'finished': [],
        })
        # # physical guidance setup
        physical_guidance_config = None
        if 'physical_guidance' in config.sample:
            pg_cfg = config.sample.physical_guidance
            if pg_cfg.get('enabled', False):
                # Physical guidance assumes the aromatic atom-index mapping.
                # The index→atomic-number table is only valid for this mode.
                if ligand_atom_mode != 'aromatic':
                    raise ValueError(
                        f"Physical guidance currently assumes ligand_atom_mode='aromatic', "
                        f"got '{ligand_atom_mode}'")
                logger.info('Loading physical guidance...')
                pg_mode = pg_cfg.get('mode', 'mace_only')
                removed = [k for k in ('linesearch', 'direction_only', 'direction_only_split')
                           if pg_cfg.get(k, False)]
                if removed:
                    raise ValueError(
                        f"physical_guidance config sets {removed}, but "
                        f"{'that guidance mode was' if len(removed)==1 else 'those guidance modes were'} "
                        f"removed. They used to fall through to a different mode "
                        f"silently; remove the key or switch to 'complex_force' / "
                        f"'interaction_aware'.")
                if pg_mode not in ('mace_only',):
                    raise ValueError(
                        f"Unknown physical_guidance.mode '{pg_mode}'. "
                        f"Only 'mace_only' is supported — the AMBER guidance path "
                        f"was removed, and a config still carrying lambda_mace / "
                        f"lambda_amber / protein_params would otherwise run to "
                        f"completion with no guidance applied.")
                device = args.device
                protein_tensors = {}
                if pg_mode == 'amber':
                    pp = np.load(pg_cfg['protein_params'])
                    protein_tensors['coords'] = torch.from_numpy(pp['coords']).float().to(device)
                    protein_tensors['charges'] = torch.from_numpy(pp['charges']).float().to(device)
                    protein_tensors['rmins'] = torch.from_numpy(pp['rmins']).float().to(device)
                    protein_tensors['epsilons'] = torch.from_numpy(pp['epsilons']).float().to(device)
                elif pg_mode == 'mace_only':
                    # Parse pocket PDB with residue info.
                    # Prefer the PDB element column (76:78); fall back to atom-name
                    # first char only when the element column is empty.
                    # Unknown elements are skipped (never silently mapped to C).
                    ELEM_MAP = {'H':1,'C':6,'N':7,'O':8,'S':16,'P':15,'F':9,
                                'CL':17,'BR':35,'I':53}
                    pkt_coords = []; pkt_elems = []; pkt_resids = []; pkt_resnames = []
                    with open(pg_cfg['pocket_pdb']) as f:
                        for l in f:
                            if l.startswith('ATOM') or l.startswith('HETATM'):
                                elem = l[76:78].strip().upper()
                                if not elem:
                                    elem = l[12:16].strip().upper()
                                    if elem and elem[0].isdigit():
                                        elem = elem[1:]  # e.g. "1HG1" → "HG1"
                                if elem not in ELEM_MAP:
                                    continue  # unknown element — skip, don't fake C
                                pkt_coords.append([float(l[30:38]),float(l[38:46]),float(l[46:54])])
                                pkt_elems.append(ELEM_MAP[elem])
                                pkt_resids.append(l[22:27].strip())
                                pkt_resnames.append(l[17:20].strip())
                    # Keep coords on CPU (numpy): the pocket never enters DiffGui
                    # forward; selection uses numpy and MACE/ASE eats numpy. Moving
                    # to GPU then back on first selection is pure overhead.
                    protein_tensors['pocket_coords'] = np.array(pkt_coords, dtype=np.float32)
                    protein_tensors['pocket_elements'] = np.array(pkt_elems, dtype=np.int32)
                    protein_tensors['pocket_resids'] = np.array(pkt_resids)
                    protein_tensors['pocket_resnames'] = np.array(pkt_resnames)
                # Initialize MACE calculator (loaded once, reused for all steps)
                init_mace_calculator(
                    model_path=pg_cfg.get('mace_model_path',
                        os.path.expanduser('~/.cache/mace/MACE-OFF23_medium.model')),
                    device=pg_cfg.get('mace_device', 'cuda'),
                )
                physical_guidance_config = {
                    'enabled': True,
                    'mode': pg_mode,
                    'start_step': pg_cfg.get('start_step', 200),
                    'interval': pg_cfg['interval'],
                    'overall_scale': pg_cfg.get('overall_scale', 1.0),
                    'dir_scale': pg_cfg.get('dir_scale', 0.1),
                    # v2 weighted mode
                    'complex_force': pg_cfg.get('complex_force',
                                              pg_cfg.get('direction_only_weighted', False)),
                    'tanh_c': pg_cfg.get('tanh_c', 3.0),
                    'w_max': pg_cfg.get('w_max', 3.0),
                    'quantile': pg_cfg.get('quantile', 0.90),
                    # v2 split-weighted mode
                    'interaction_aware': pg_cfg.get('interaction_aware',
                                                  pg_cfg.get('direction_only_split_weighted', False)),
                    'alpha': pg_cfg.get('alpha', 5.0),
                    # readiness gating (fast joint check + minimal repair)
                    'readiness': pg_cfg.get('readiness', False),
                    # adaptive start
                    'adaptive_start': pg_cfg.get('adaptive_start', False),
                    'stable_threshold': pg_cfg.get('stable_threshold', 0.95),
                    'stable_steps': pg_cfg.get('stable_steps', 10),
                    'min_step': pg_cfg.get('min_step', 400),
                    # per-step guidance log (accumulated, dumped after each molecule)
                    'guidance_log': pg_cfg.get('guidance_log', None),
                    # proxy-confidence scaling knobs (consumed in model.py)
                    'alpha_proxy': pg_cfg.get('alpha_proxy', 1.0),
                    # per-atom displacement log (consumed in model.py)
                    'guidance_atom_log': pg_cfg.get('guidance_atom_log', None),
                    'atom_log_data': [],
                    # geometry guard (consumed in model.py)
                    'clamp_stretch': pg_cfg.get('clamp_stretch', True),
                    'max_stretch': pg_cfg.get('max_stretch', 0.3),
                    'alpha_min': pg_cfg.get('alpha_min', 0.2),
                    'protein_tensors': protein_tensors,
                }
                if pg_mode == 'amber':
                    physical_guidance_config['lambda_mace'] = pg_cfg['lambda_mace']
                    physical_guidance_config['lambda_amber'] = pg_cfg['lambda_amber']
                elif pg_mode == 'mace_only':
                    physical_guidance_config['lambda_lig'] = pg_cfg['lambda_lig']
                    physical_guidance_config['lambda_pocket'] = pg_cfg['lambda_pocket']
                logger.info(f'  Physical guidance ({pg_mode}): start={pg_cfg.get("start_step", "adaptive")}, '
                            f'interval={pg_cfg["interval"]}')
        # # generating molecules
        mol_list = []
        sdf_dir = log_dir + '/'+ f'{name}_SDF'
        os.makedirs(sdf_dir, exist_ok=True)
        log_path = os.path.join(sdf_dir, 'log.txt')
        with open(log_path, 'w') as f:
            f.write('number, smiles, sa, qed:' + '\n')
        while len(pool.finished) < config.sample.num_mols:
            if len(pool.failed) > 3 * (config.sample.num_mols):
                logger.info('Too many failed molecules. Stop sampling.')
                break

            batch_size = args.batch_size if args.batch_size > 0 else config.sample.batch_size
            n_graphs = min(batch_size, (config.sample.num_mols - len(pool.finished))*2)
            batch = Batch.from_data_list([data.clone() for _ in range(n_graphs)], follow_batch=featurizer.follow_batch).to(args.device)

            if config.sample.sample_method == "priori":
                pocket_size = get_pocket_size(batch.protein_pos.detach().cpu().numpy())
                ligand_num_atoms = [sample_atom_num(pocket_size).astype(int) for _ in range(n_graphs)]
                ligand_batch = torch.repeat_interleave(torch.arange(n_graphs), torch.tensor(ligand_num_atoms)).to(args.device)
            elif config.sample.sample_method == "range":
                ligand_num_atoms = np.random.normal(24.923464980477522, 5.516291901819105, size=n_graphs)
                ligand_num_atoms = ligand_num_atoms.astype('int64')
                ligand_batch = torch.repeat_interleave(torch.arange(n_graphs), torch.tensor(ligand_num_atoms)).to(args.device)
            elif config.sample.sample_method == "ref":
                ligand_batch = batch.ligand_element_batch
                ligand_num_atoms = scatter_sum(torch.ones_like(ligand_batch), ligand_batch, dim=0).tolist()
            else:
                raise ValueError
            
            if config.model.gen_mode != 'denovo':
                frag_batch = batch.frag_element_batch
                frag_num_atoms = scatter_sum(torch.ones_like(frag_batch), frag_batch, dim=0).tolist()
                all_greater = all(l > f for l, f in zip(ligand_num_atoms, frag_num_atoms))
                if not all_greater:
                    continue
            logger.info(f'ligand_num_atoms: {ligand_num_atoms}')

            batch_holder = make_data_placeholder(n_nodes_list=ligand_num_atoms, device=args.device)
            batch_node, halfedge_index, batch_halfedge = batch_holder['batch_node'], batch_holder['halfedge_index'], batch_holder['batch_halfedge']
            
            # inference
            if config.model.gen_mode == 'denovo':
                outputs = model.sample(
                    n_graphs=n_graphs,
                    protein_node=batch.protein_atom_feat.float(),
                    protein_pos=batch.protein_pos,
                    protein_batch=batch.protein_element_batch,
                    ligand_batch=batch_node,
                    halfedge_index=halfedge_index,
                    halfedge_batch=batch_halfedge,
                    batch_lab=batch_lab,
                    gui_strength=config.sample.gui_strength,
                    bond_predictor=bond_predictor,
                    guidance=guidance,
                    physical_guidance_config=physical_guidance_config,
                )
            elif config.model.gen_mode in ('frag_cond', 'frag_diff'):
                outputs = model.sample_frag(
                    n_graphs=n_graphs,
                    protein_node=batch.protein_atom_feat.float(),
                    protein_pos=batch.protein_pos,
                    protein_batch=batch.protein_element_batch,
                    frag_node=batch.frag_atom_feat_full,
                    frag_pos=batch.frag_pos,
                    frag_batch=batch.frag_element_batch,
                    frag_halfedge_type=batch.frag_halfedge_type,
                    frag_halfedge_index=batch.frag_halfedge_index,
                    frag_halfedge_batch=batch.frag_halfedge_type_batch,
                    ligand_batch=batch_node,
                    halfedge_index=halfedge_index,
                    halfedge_batch=batch_halfedge,
                    batch_lab=batch_lab,
                    gui_strength=config.sample.gui_strength,
                    bond_predictor=bond_predictor,
                    guidance=guidance,
                    gen_mode=config.model.gen_mode,
                    physical_guidance_config=physical_guidance_config,
                )

            outputs = {key:[v.cpu().numpy() for v in value] for key, value in outputs.items()}
            
            # decode outputs to molecules
            batch_node, halfedge_index, batch_halfedge = batch_node.cpu().numpy(), halfedge_index.cpu().numpy(), batch_halfedge.cpu().numpy()
            try:
                output_list = seperate_outputs(outputs, n_graphs, batch_node, halfedge_index, batch_halfedge)
            except Exception as e:
                logger.info(f'Separate results error: {e}')
                continue
            gen_list = []
            for i_mol, output_mol in enumerate(output_list):
                mol_info = featurizer.decode_output(
                    pred_node=output_mol['pred'][0],
                    pred_pos=output_mol['pred'][1],
                    pred_halfedge=output_mol['pred'][2],
                    halfedge_index=output_mol['halfedge_index'],
                )  # note: traj is not used
                if add_edge == 'openbabel':
                    del mol_info['bond_index']
                    del mol_info['bond_type']
                    del mol_info['bond_prob']

                # Generation outcome tracking — set before reconstruction so every
                # trajectory (success or fail) records its status and reason.
                mol_info['generation_status'] = 'unknown'
                mol_info['failure_reason'] = ''

                # Attach trajectory-level physical guidance diagnostics BEFORE
                # reconstruction — so molecules that fail reconstruction keep
                # their stats (low mace_ready ↔ reconstruction failure analysis).
                if physical_guidance_config is not None and 'last_stats' in physical_guidance_config:
                    mol_info['physical_stats'] = dict(physical_guidance_config['last_stats'])
                if physical_guidance_config is not None \
                        and physical_guidance_config.get('last_pocket') is not None:
                    lp = physical_guidance_config['last_pocket']
                    mol_info['physical_pocket'] = {
                        'coords': lp['coords'].copy(),
                        'elements': lp['elements'].copy(),
                        'resids': lp['resids'].copy(),
                        'resnames': lp['resnames'].copy(),
                        'atom_indices': lp['atom_indices'].copy(),
                        'fixed_step': lp.get('fixed_step'),
                    }

                try:
                    rdmol = reconstruct_from_generated_with_edges(mol_info, add_edge=add_edge)
                except MolReconsError:
                    mol_info['generation_status'] = 'failed'
                    mol_info['failure_reason'] = 'reconstruction'
                    pool.failed.append(mol_info)
                    logger.warning('Reconstruction error encountered.')
                    # Save placeholder to preserve generation-order alignment
                    idx = len(mol_list)
                    with open(os.path.join(sdf_dir, '%d.sdf' % idx), 'w') as pf:
                        pf.write('FAILED\nreconstruction error\n\n  0  0  0  0  0  0            999 V2000\nM  END\n$$$$\n')
                    with open(log_path, 'a') as lf:
                        lf.write(f'{idx}, FAILED, , , , \n')
                    continue
                mol_info['rdmol'] = rdmol
                smiles = Chem.MolToSmiles(rdmol)
                mol_info['smiles'] = smiles
                contain_B = re.search(r'B(?![rR]\b)', smiles)
                if '.' in smiles:
                    mol_info['generation_status'] = 'failed'
                    mol_info['failure_reason'] = 'disconnected'
                    logger.warning('Incomplete molecule: %s' % smiles)
                    pool.failed.append(mol_info)
                    idx = len(mol_list)
                    with open(os.path.join(sdf_dir, '%d.sdf' % idx), 'w') as pf:
                        pf.write('FAILED\nincomplete molecule\n\n  0  0  0  0  0  0            999 V2000\nM  END\n$$$$\n')
                    with open(log_path, 'a') as lf:
                        lf.write(f'{idx}, FAILED, , , , \n')
                    continue
                elif contain_B:
                    logger.warning('Element Boron in molecule: %s' % smiles)
                else:   # Pass checks!
                    mol_info['generation_status'] = 'success'
                    mol_info['failure_reason'] = ''
                    logger.info('Success: %s' % smiles)
                    mol_info['sa'] = 0.0
                    mol_info['qed'] = 0.0
                    # Try scoring but don't block SDF saving
                    try:
                        chem_results = scoring_func.get_chem(rdmol)
                        mol_info['sa'] = chem_results['sa']
                        mol_info['qed'] = chem_results['qed']
                    except Exception as e:
                        logger.warning('Chem scoring failed: %s' % str(e)[:100])
                    # (physical_stats / physical_pocket were attached right after
                    #  decode_output — see above — so they survive even if
                    #  reconstruction failed)
                    gen_list.append(mol_info)
                    mol_list.append(mol_info)
                    # Incremental save: write SDF + pocket.npz + log immediately,
                    # so a crash at molecule N keeps molecules 0..N-1 fully saved.
                    idx = len(mol_list) - 1
                    Chem.MolToMolFile(rdmol, os.path.join(sdf_dir, '%d.sdf' % idx))
                    lp_now = mol_info.get('physical_pocket')
                    if lp_now is not None:
                        np.savez(
                            os.path.join(sdf_dir, f'{idx}_pocket.npz'),
                            coords=lp_now['coords'],
                            elements=lp_now['elements'],
                            resids=lp_now['resids'],
                            resnames=lp_now['resnames'],
                            atom_indices=lp_now['atom_indices'],
                        )
                    with open(log_path, 'a') as lf:
                        lf.write(f'{idx}, {smiles}, {mol_info["sa"]}, {mol_info["qed"]}\n')
                    # Clear GPU cache to prevent memory accumulation
                    import gc
                    torch.cuda.empty_cache()
                    gc.collect()
                    # pool.finished.append(mol_info)
            pool.finished.extend(gen_list)
            print_pool_status(pool, logger)

        # # Already saved incrementally; just write SMILES summary
        with open(os.path.join(log_dir, 'SMILES.txt'), 'a') as smiles_f:
            for data_finished in mol_list:
                smiles_f.write(data_finished['smiles'] + '\n')

                if 'traj' in data_finished:
                    writer = Chem.SDWriter(os.path.join(sdf_dir, 'traj_%d.sdf' % (i)))
                    for m in data_finished['traj']:
                        try:
                            writer.write(m)
                        except:
                            writer.write(Chem.MolFromSmiles('O'))

        # # Per-molecule physical guidance stats + pocket → CSV / npz
        stats_csv = os.path.join(log_dir, 'guidance_stats.csv')
        with open(stats_csv, 'w') as gf:
            gf.write('status,failure_reason,smiles,'
                     'attempt,precheck_fail,repair_none,postcheck_fail,repair_ratio_fail,'
                     'kekulize_fail,aromatic_fail,valence_fail,rdkit_other,'
                     'mace_fail,mace_nonfinite,mace_ready,'
                     'delta_nonfinite,applied,guidance_exception,'
                     'proxy_direct,proxy_repaired,proxy_fail,'
                     'proxy_added_bonds,proxy_removed_bonds,proxy_order_changes,proxy_charge_adjustments,'
                     'proxy_aromatic_ambiguous,proxy_aromatic_fallback,proxy_tautomer_mean,'
                     'proxy_conf_mean,proxy_conf_min,effective_scale_mean,'
                     'first_mace_step,'
                     'pocket_n_atoms,pocket_n_residues,pocket_resids\n')
            # Include FAILED trajectories too (pool.failed) — their guidance stats
            # matter for the "low mace_ready ↔ reconstruction failure" question.
            all_mols = [(m, m.get('smiles', 'FAILED')) for m in mol_list]
            for m in pool.failed:
                all_mols.append((m, m.get('smiles', 'FAILED')))
            for mol_idx, (data_finished, smiles_row) in enumerate(all_mols):
                ps = data_finished.get('physical_stats')
                if ps is None:
                    continue
                lp = data_finished.get('physical_pocket')
                if lp is not None:
                    p_n_atoms = len(lp['elements'])
                    p_n_res = len(set(lp['resids'].tolist()))
                    p_resids = ';'.join(str(r) for r in sorted(set(lp['resids'].tolist())))
                else:
                    p_n_atoms, p_n_res, p_resids = '', '', ''
                first_step = ps.get('first_mace_step')
                status = data_finished.get('generation_status', '')
                reason = data_finished.get('failure_reason', '')
                proxy_direct = ps.get('proxy_direct', 0)  # projector-accumulated, not inferred
                # trajectory-level proxy confidence (accumulated in model.py)
                pn = ps.get('proxy_conf_n', 0)
                if pn:
                    proxy_conf_mean = ps.get('proxy_conf_sum', 0.0) / pn
                    effective_scale_mean = ps.get('effective_scale_sum', 0.0) / pn
                    proxy_conf_min = ps.get('proxy_conf_min', '')
                else:
                    proxy_conf_mean, proxy_conf_min, effective_scale_mean = '', '', ''
                ptn = ps.get('proxy_tautomer_n', 0)
                proxy_tautomer_mean = (ps.get('proxy_tautomer_sum', 0.0) / ptn) if ptn else ''
                gf.write(f"{status},{reason},{smiles_row},"
                         f"{ps.get('attempt', 0)},"
                         f"{ps.get('precheck_fail', 0)},"
                         f"{ps.get('repair_none', 0)},"
                         f"{ps.get('postcheck_fail', 0)},"
                         f"{ps.get('repair_ratio_fail', 0)},"
                         f"{ps.get('kekulize_fail', 0)},"
                         f"{ps.get('aromatic_fail', 0)},"
                         f"{ps.get('valence_fail', 0)},"
                         f"{ps.get('rdkit_other', 0)},"
                         f"{ps.get('mace_fail', 0)},"
                         f"{ps.get('mace_nonfinite', 0)},"
                         f"{ps.get('mace_ready', 0)},"
                         f"{ps.get('delta_nonfinite', 0)},"
                         f"{ps.get('applied', 0)},"
                         f"{ps.get('guidance_exception', 0)},"
                         f"{proxy_direct},"
                         f"{ps.get('proxy_repaired', 0)},"
                         f"{ps.get('proxy_fail', 0)},"
                         f"{ps.get('proxy_added_bonds', 0)},"
                         f"{ps.get('proxy_removed_bonds', 0)},"
                         f"{ps.get('proxy_order_changes', 0)},"
                         f"{ps.get('proxy_charge_adjustments', 0)},"
                         f"{ps.get('proxy_aromatic_ambiguous', 0)},"
                         f"{ps.get('proxy_aromatic_fallback', 0)},"
                         f"{proxy_tautomer_mean},"
                         f"{proxy_conf_mean},"
                         f"{proxy_conf_min},"
                         f"{effective_scale_mean},"
                         f"{first_step if first_step is not None else ''},"
                         f"{p_n_atoms},{p_n_res},{p_resids}\n")
        print(f"guidance stats saved: {stats_csv}")

        if config.data.dataset == 'pdbbind':
            torch.save(pool, os.path.join(log_dir, f'samples_{name}.pt'))
        elif config.data.dataset == 'crossdocked':
            name = name.replace('/', '-')
            torch.save(pool, os.path.join(log_dir, f'samples_{name}.pt'))


if __name__ == '__main__':
    # Usage: python scripts/sample.py --outdir ./outputs --config ./configs/sample/sample.yml --device cuda:0
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/sample/sample.yml')
    parser.add_argument('--outdir', type=str, default='./outputs')
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    main(args)
    
