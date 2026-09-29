from tqdm import tqdm
import torch
from torch.nn import Module
from torch.nn import functional as F
from torch_scatter import scatter_sum, scatter_mean
from models.transition import ContigousTransition, GeneralCategoricalTransition
from models.egnn import EgnnNet
from .common import *
from .diffusion import *
from models.physical_guidance import (
    mace_pocket_force_with_h,
    project_to_mace_proxy,
    AROMATIC_INDEX_TO_ATOMIC_NUMBER,
)


class UnsupportedGuidanceMode(RuntimeError):
    """A config asked for a guidance mode this code does not implement.

    Deliberately not a subclass of the errors the per-step try/except swallows:
    a misconfigured mode is a configuration error, not a failed molecule, and
    must abort the run instead of being counted and skipped.
    """


class DiffGui(Module):
    def __init__(self,
        config,
        protein_node_types,
        ligand_node_types,
        num_edge_types,  # explicit bond type: 0, 1, 2, 3, 4
        **kwargs
    ):
        super().__init__()
        self.config = config
        self.protein_node_types = protein_node_types
        self.ligand_node_types = ligand_node_types
        self.num_edge_types = num_edge_types
        self.k = config.knn
        self.cutoff_mode = config.cutoff_mode
        self.center_pos_mode = config.center_pos_mode
        self.bond_len_loss = getattr(config, 'bond_len_loss', False)

        # # define beta and alpha
        self.define_betas_alphas(config.diff)

        # # embedding
        if self.config.node_indicator:
            node_dim = config.node_dim - 1
        else:
            node_dim = config.node_dim
        edge_dim = config.edge_dim
        time_dim = config.diff.time_dim
        class_dim = config.class_dim
        class_emb_dim = config.class_emb_dim
        self.protein_node_embedder = nn.Linear(protein_node_types, node_dim, bias=False) # protein element type
        self.protein_edge_embedder = nn.Linear(num_edge_types, edge_dim, bias=False) # protein bond type
        if self.config.train_mode in ('ori', 'no_bond'):
            self.ligand_node_embedder = nn.Linear(ligand_node_types, node_dim - time_dim - class_emb_dim, bias=False)  # ligand element type
            self.ligand_edge_embedder = nn.Linear(num_edge_types, edge_dim - time_dim - class_emb_dim, bias=False) # ligand bond type
        elif self.config.train_mode in ('no_lab', 'no_both'):
            self.ligand_node_embedder = nn.Linear(ligand_node_types, node_dim - time_dim, bias=False)  # ligand element type
            self.ligand_edge_embedder = nn.Linear(num_edge_types, edge_dim - time_dim, bias=False) # ligand bond type
        self.time_emb = nn.Sequential(
            GaussianSmearing(stop=self.num_timesteps, num_gaussians=time_dim, type_='linear'),
        )
        self.class_emb = nn.Sequential(
            nn.Linear(class_dim, class_emb_dim * 4),
            nn.LayerNorm(class_emb_dim * 4),
            nn.GELU(),
            nn.Linear(class_emb_dim * 4, class_emb_dim)
        )
        
        # # denoiser
        if config.denoiser.backbone == 'EGNN':
            self.denoiser = EgnnNet(config.node_dim, config.edge_dim, **config.denoiser)
        else:
            raise NotImplementedError(config.denoiser.backbone)

        # # decoder
        self.ligand_node_decoder = MLP(config.node_dim, ligand_node_types, config.node_dim)
        self.ligand_edge_decoder = MLP(config.edge_dim, num_edge_types, config.edge_dim)


    def define_betas_alphas(self, config):
        self.num_timesteps = config.num_timesteps
        self.categorical_space = getattr(config, 'categorical_space', 'discrete')
        
        # try to get the scaling
        if self.categorical_space == 'continuous':
            self.scaling = getattr(config, 'scaling', [1., 1., 1.])
        else:
            self.scaling = [1., 1., 1.]  # actually not used for discrete space (defined for compatibility)

        # # diffusion for pos
        pos_betas = get_beta_schedule(
            num_timesteps=self.num_timesteps,
            **config.diff_pos
        )
        assert self.scaling[0] == 1, 'scaling for pos should be 1'
        self.pos_transition = ContigousTransition(pos_betas)

        # # diffusion for node type
        node_betas = get_beta_schedule(
            num_timesteps=self.num_timesteps,
            **config.diff_atom
        )
        if self.categorical_space == 'discrete':
            init_prob = config.diff_atom.init_prob
            self.node_transition = GeneralCategoricalTransition(node_betas, self.ligand_node_types,
                                                            init_prob=init_prob)
        elif self.categorical_space == 'continuous':
            scaling_node = self.scaling[1]
            self.node_transition = ContigousTransition(node_betas, self.ligand_node_types, scaling_node)
        else:
            raise ValueError(self.categorical_space)

        # # diffusion for edge type
        edge_betas = get_beta_schedule(
            num_timesteps=self.num_timesteps,
            **config.diff_bond
        )
        if self.categorical_space == 'discrete':
            init_prob = config.diff_bond.init_prob
            self.edge_transition = GeneralCategoricalTransition(edge_betas, self.num_edge_types,
                                                            init_prob=init_prob)
        elif self.categorical_space == 'continuous':
            scaling_edge = self.scaling[2]
            self.edge_transition = ContigousTransition(edge_betas, self.num_edge_types, scaling_edge)
        else:
            raise ValueError(self.categorical_space)

    def sample_time(self, num_graphs, device, **kwargs):
        time_step = torch.randint(
            0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=device)
        time_step = torch.cat(
            [time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
        pt = torch.ones_like(time_step).float() / self.num_timesteps
        return time_step, pt
    
    def fix_zero_time(self, num_graphs, device, **kwargs): 
        time_step = torch.zeros(num_graphs, dtype=torch.long, device=device)   
        pt = torch.ones_like(time_step).float() / self.num_timesteps  
        return time_step, pt

    def _get_edge_index(self, x, batch, ligand_mask):
        if self.cutoff_mode == "knn":
            edge_index = knn_graph(x, k=self.k, batch=batch, flow="target_to_source")
        elif self.cutoff_mode == "hybrid":
            edge_index = batch_hybrid_edge_connection(
                x, k=self.k, ligand_mask=ligand_mask, batch=batch, add_p_index=True
            )
        else:
            raise ValueError(
                f"Unsupported cutoff mode: {self.cutoff_mode}! Please select cutoff mode among knn, hybrid."
            )
        return edge_index

    def _get_edge_type(self, edge_index, ligand_mask):
        src, dst = edge_index
        edge_type = torch.zeros(len(src), dtype=torch.int64).to(edge_index.device)
        n_src = ligand_mask[src] == 1
        n_dst = ligand_mask[dst] == 1
        edge_type[n_src & n_dst] = 0
        edge_type[n_src & ~n_dst] = 1
        edge_type[~n_src & n_dst] = 2
        edge_type[~n_src & ~n_dst] = 3

        nonzero_indices = torch.nonzero(edge_type).flatten()
        edge_type = torch.index_select(edge_type, dim=0, index=nonzero_indices)
        edge_type = torch.zeros_like(edge_type)
        edge_index = torch.index_select(edge_index, dim=1, index=nonzero_indices)
        edge_type = F.one_hot(edge_type, num_classes=self.num_edge_types)
        return edge_type, edge_index

    def forward(
        self, protein_node, protein_pos, protein_batch, 
        ligand_node_pert, ligand_pos_pert, ligand_batch,
        ligand_edge_pert, ligand_edge_index, ligand_edge_batch, 
        t, lab
    ):
        """
        Predict Ligand at step `0` given perturbed Ligand at step `t` with hidden dims and time step
        """
        # 1 node, edge and time embedding
        time_embed_node = self.time_emb(t.index_select(0, ligand_batch))
        class_embed_node = self.class_emb(lab.index_select(0, ligand_batch))
        time_embed_edge = self.time_emb(t.index_select(0, ligand_edge_batch))
        class_embed_edge = self.class_emb(lab.index_select(0, ligand_edge_batch))
        if self.config.train_mode in ('ori', 'no_bond'):
            ligand_node_h_pert = torch.cat([self.ligand_node_embedder(ligand_node_pert), time_embed_node, class_embed_node], dim=-1)
            ligand_edge_h_pert = torch.cat([self.ligand_edge_embedder(ligand_edge_pert), time_embed_edge, class_embed_edge], dim=-1)
        elif self.config.train_mode in ('no_lab', 'no_both'):
            ligand_node_h_pert = torch.cat([self.ligand_node_embedder(ligand_node_pert), time_embed_node], dim=-1)
            ligand_edge_h_pert = torch.cat([self.ligand_edge_embedder(ligand_edge_pert), time_embed_edge], dim=-1)
        protein_h = self.protein_node_embedder(protein_node)

        if self.config.node_indicator:
            protein_h = torch.cat([protein_h, torch.zeros(len(protein_h), 1).to(protein_h)], -1)
            ligand_node_h_pert = torch.cat([ligand_node_h_pert, torch.ones(len(ligand_node_h_pert), 1).to(ligand_node_h_pert)], -1)

        # 2 combine protein and ligand input
        all_node_h, all_node_pos, all_node_batch, ligand_mask = compose(
            protein_h, protein_pos, protein_batch, ligand_node_h_pert, ligand_pos_pert, ligand_batch
        )

        sub_edge_index = self._get_edge_index(all_node_pos, all_node_batch, ligand_mask)
        sub_edge_type, sub_edge_index = self._get_edge_type(sub_edge_index, ligand_mask)
        sub_edge_batch = all_node_batch[sub_edge_index[0]]
        sub_edge_h = self.protein_edge_embedder(sub_edge_type.to(torch.float32))
        node_batch_counts = torch.bincount(all_node_batch)
        ligand_node_batch_counts = torch.bincount(ligand_batch)
        cumulative_nodes = torch.cat([torch.tensor([0]).to(all_node_batch.device), torch.cumsum(node_batch_counts, dim=0)[:-1]])
        cumulative_ligand_nodes = torch.cat([torch.tensor([0]).to(ligand_batch.device), torch.cumsum(ligand_node_batch_counts, dim=0)[:-1]])
        new_ligand_edge_index = ligand_edge_index + cumulative_nodes[ligand_edge_batch] - cumulative_ligand_nodes[ligand_edge_batch]
        all_edge_h, all_edge_index, all_edge_batch, ligand_edge_mask = edge_compose(
            sub_edge_h, sub_edge_index, sub_edge_batch, ligand_edge_h_pert, new_ligand_edge_index, ligand_edge_batch
        )

        # 3 diffuse to get the updated node embedding and bond embedding
        node_h, node_pos, edge_h = self.denoiser(
            node_h=all_node_h,
            node_pos=all_node_pos, 
            edge_h=all_edge_h, 
            edge_index=all_edge_index,
            node_time=t.index_select(0, all_node_batch).unsqueeze(-1) / self.num_timesteps,
            edge_time=t.index_select(0, all_edge_batch).unsqueeze(-1) / self.num_timesteps,
            ligand_mask=ligand_mask
        )
        
        ligand_node_h = node_h[ligand_mask]
        ligand_node_pos = node_pos[ligand_mask]
        ligand_edge_h = edge_h[ligand_edge_mask]
        n_halfedges = ligand_edge_h.shape[0] // 2
        pred_ligand_node = self.ligand_node_decoder(ligand_node_h)
        pred_ligand_halfedge = self.ligand_edge_decoder(ligand_edge_h[:n_halfedges] + ligand_edge_h[n_halfedges:])
        pred_ligand_pos = ligand_node_pos
        
        return {
            'pred_ligand_node': pred_ligand_node,
            'pred_ligand_pos': pred_ligand_pos,
            'pred_ligand_halfedge': pred_ligand_halfedge
        }  # ligand at step 0

    def get_loss(
        self, protein_node, protein_pos, protein_batch, 
        ligand_node, ligand_pos, ligand_batch,
        halfedge_type, halfedge_index, halfedge_batch,
        num_mol, batch_lab
    ):
        num_graphs = num_mol
        device = ligand_pos.device
        protein_pos, ligand_pos, _ = center_pos(
            protein_pos, ligand_pos, protein_batch, ligand_batch, mode=self.center_pos_mode
        )

        # 1. sample noise levels
        time_step, _ = self.sample_time(num_graphs, device)

        # 2. perturb pos, node, edge
        pos_pert = self.pos_transition.add_noise(ligand_pos, time_step, ligand_batch)
        node_pert = self.node_transition.add_noise(ligand_node, time_step, ligand_batch)
        halfedge_pert = self.edge_transition.add_noise(halfedge_type, time_step, halfedge_batch)
        ligand_edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)  # undirected edges
        ligand_edge_batch = torch.cat([halfedge_batch, halfedge_batch], dim=0)
        if self.categorical_space == 'discrete':
            ligand_node_pert, log_node_t, log_node_0 = node_pert
            ligand_halfedge_pert, log_halfedge_t, log_halfedge_0 = halfedge_pert
        else:
            ligand_node_pert, ligand_node_0 = node_pert
            ligand_halfedge_pert, ligand_halfedge_0 = halfedge_pert
        
        ligand_edge_pert = torch.cat([ligand_halfedge_pert, ligand_halfedge_pert], dim=0)
        ligand_pos_pert = pos_pert

        # 3. forward to denoise
        preds = self(
            protein_node, protein_pos, protein_batch,
            ligand_node_pert, ligand_pos_pert, ligand_batch,
            ligand_edge_pert, ligand_edge_index, ligand_edge_batch, 
            time_step, batch_lab
        )
        pred_ligand_node = preds['pred_ligand_node']
        pred_ligand_pos = preds['pred_ligand_pos']
        pred_ligand_halfedge = preds['pred_ligand_halfedge']

        # 4. loss
        # 4.1 pos loss
        loss_pos = F.mse_loss(pred_ligand_pos, ligand_pos)
        if self.bond_len_loss == True:
            bond_index = halfedge_index[:, halfedge_type > 0]
            true_length = torch.norm(ligand_pos[bond_index[0]] - ligand_pos[bond_index[1]], dim=-1)
            pred_length = torch.norm(pred_ligand_pos[bond_index[0]] - pred_ligand_pos[bond_index[1]], dim=-1)
            loss_len = F.mse_loss(pred_length, true_length)
    
        if self.categorical_space == 'discrete':
            # 4.2 node type loss
            log_node_recon = F.log_softmax(pred_ligand_node, dim=-1)
            log_node_post_true = self.node_transition.q_v_posterior(log_node_0, log_node_t, time_step, ligand_batch, v0_prob=True)
            log_node_post_pred = self.node_transition.q_v_posterior(log_node_recon, log_node_t, time_step, ligand_batch, v0_prob=True)
            kl_node = self.node_transition.compute_v_Lt(log_node_post_true, log_node_post_pred, log_node_0, t=time_step, batch=ligand_batch)
            loss_node = torch.mean(kl_node) * 100
            # 4.3 edge type loss
            log_halfedge_recon = F.log_softmax(pred_ligand_halfedge, dim=-1)
            log_edge_post_true = self.edge_transition.q_v_posterior(log_halfedge_0, log_halfedge_t, time_step, halfedge_batch, v0_prob=True)
            log_edge_post_pred = self.edge_transition.q_v_posterior(log_halfedge_recon, log_halfedge_t, time_step, halfedge_batch, v0_prob=True)
            kl_edge = self.edge_transition.compute_v_Lt(log_edge_post_true, log_edge_post_pred, log_halfedge_0, t=time_step, batch=halfedge_batch)
            loss_edge = torch.mean(kl_edge)  * 100
        else:
            loss_node = F.mse_loss(pred_ligand_node, ligand_node_0)  * 30
            loss_edge = F.mse_loss(pred_ligand_halfedge, ligand_halfedge_0) * 30

        # total loss
        if self.config.train_mode in ('ori', 'no_lab'):
            loss_total = loss_pos + loss_node + loss_edge + (loss_len if self.bond_len_loss else 0)
            loss_dict = {
            'loss': loss_total,
            'loss_pos': loss_pos,
            'loss_node': loss_node,
            'loss_edge': loss_edge
        }
        elif self.config.train_mode in ('no_bond', 'no_both'):
            loss_total = loss_pos + loss_node + (loss_len if self.bond_len_loss else 0)
            loss_dict = {
                'loss': loss_total,
                'loss_pos': loss_pos,
                'loss_node': loss_node
            }
        if self.bond_len_loss == True:
            loss_dict['loss_len'] = loss_len

        pred_dict = {
            'pred_ligand_node': F.softmax(pred_ligand_node, dim=-1),
            'pred_ligand_pos': pred_ligand_pos,
            'pred_ligand_halfedge': F.softmax(pred_ligand_halfedge, dim=-1)
        }
        return loss_dict, pred_dict

    def _predict_x0_from_eps(self, xt, eps, t, batch):
        pos0_from_eps = extract(self.pos_transition.sqrt_recip_alphas_bar, t, batch) * xt - \
                      extract(self.pos_transition.sqrt_recipm1_alphas_bar, t, batch) * eps
        return pos0_from_eps

    def _predict_eps_from_x0(self, xt, t, pred_x0, batch):
        return (
            (extract(self.pos_transition.sqrt_recip_alphas_bar, t, batch) * xt - pred_x0) /
            extract(self.pos_transition.sqrt_recipm1_alphas_bar, t, batch)
        )

    def classifier_free(
        self, protein_node, protein_pos, protein_batch,
        ligand_node_pert, ligand_pos_pert, ligand_batch, 
        ligand_edge_pert, ligand_edge_index, ligand_edge_batch, 
        gui_strength, time_step, batch_lab
    ):
        """
        Compute new results for the start step in classifier free diffusion sampling.
        """
        preds_cond = self(
            protein_node, protein_pos, protein_batch,
            ligand_node_pert, ligand_pos_pert, ligand_batch,
            ligand_edge_pert, ligand_edge_index, ligand_edge_batch, 
            time_step, batch_lab
        )

        batch_lab_zero = torch.zeros(batch_lab.shape, device=ligand_batch.device)
        preds_uncond = self(
            protein_node, protein_pos, protein_batch,
            ligand_node_pert, ligand_pos_pert, ligand_batch,
            ligand_edge_pert, ligand_edge_index, ligand_edge_batch, 
            time_step, batch_lab
        )

        pred_eps_cond = self._predict_eps_from_x0(
            xt=ligand_pos_pert, t=time_step, pred_x0=preds_cond['pred_ligand_pos'], batch=ligand_batch
        )
        pred_eps_uncond = self._predict_eps_from_x0(
            xt=ligand_pos_pert, t=time_step, pred_x0=preds_uncond['pred_ligand_pos'], batch=ligand_batch
        )
        pred_eps = (1 + gui_strength) * pred_eps_cond - gui_strength * pred_eps_uncond
        pred_ligand_pos = self._predict_x0_from_eps(xt=ligand_pos_pert, t=time_step, eps=pred_eps, batch=ligand_batch)

        pred_ligand_node = preds_cond['pred_ligand_node'] + preds_uncond['pred_ligand_node']
        pred_ligand_halfedge = preds_cond['pred_ligand_halfedge'] + preds_uncond['pred_ligand_halfedge']

        return pred_ligand_pos, pred_ligand_node, pred_ligand_halfedge

    @torch.no_grad()
    def sample(
        self, n_graphs,
        protein_node, protein_pos, protein_batch,
        ligand_batch, halfedge_index, halfedge_batch,
        batch_lab=None, gui_strength=None,
        bond_predictor=None, guidance=None,
        physical_guidance_config=None,
    ):
        device = ligand_batch.device
        # # 1. get the init values (position, node and edge types)
        n_nodes_all = len(ligand_batch)
        n_halfedges_all = len(halfedge_batch)
        
        node_init = self.node_transition.sample_init(n_nodes_all)
        halfedge_init = self.edge_transition.sample_init(n_halfedges_all)
        if self.categorical_space == 'discrete':
            _, ligand_node_h_init, log_node_type = node_init
            _, ligand_halfedge_h_init, log_halfedge_type = halfedge_init
        else:
            ligand_node_h_init = node_init
            ligand_halfedge_h_init = halfedge_init
            
        pocket_center_pos = scatter_mean(protein_pos, protein_batch, dim=0)
        ligand_center_pos = pocket_center_pos[ligand_batch]
        ligand_pos_init = ligand_center_pos + torch.randn_like(ligand_center_pos)
        protein_pos, ligand_pos_init, offset = center_pos(protein_pos, ligand_pos_init, protein_batch, ligand_batch, self.center_pos_mode)

        # # 1.1 log init
        ligand_node_traj = torch.zeros([self.num_timesteps + 1, n_nodes_all, ligand_node_h_init.shape[-1]],
                                dtype=ligand_node_h_init.dtype).to(device)
        ligand_pos_traj = torch.zeros([self.num_timesteps + 1, n_nodes_all, 3], dtype=ligand_pos_init.dtype).to(device)
        ligand_halfedge_traj = torch.zeros([self.num_timesteps + 1, n_halfedges_all, ligand_halfedge_h_init.shape[-1]],
                                    dtype=ligand_halfedge_h_init.dtype).to(device)
        ligand_node_traj[0] = ligand_node_h_init
        ligand_pos_traj[0] = ligand_pos_init + offset[ligand_batch]
        ligand_halfedge_traj[0] = ligand_halfedge_h_init

        # ==============================================================
        #  Physical guidance - what this block does, end to end
        # ==============================================================
        #
        # DiffGui denoises ligand coordinates, atom types and bond types
        # together. This block adds a physical correction on top: at selected
        # late denoising steps it computes a MACE-based force and pushes the
        # ligand coordinates along it. Atom and bond identities are never
        # touched - only the coordinate trajectory is modified.
        #
        # Per guidance step, for each molecule:
        #
        #   1. Rebuild a chemistry-aware proxy of the predicted ligand
        #      (project_to_mace_proxy). The raw predicted state is not
        #      directly evaluable by MACE, so a valid heavy-atom topology is
        #      reconstructed first. If one cannot be built, this molecule is
        #      skipped for this step.
        #
        #   2. Select the protein environment: residues with any atom within
        #      5 A of the ligand. Chosen ONCE per molecule, at its first
        #      successful step, then held fixed for the rest of that
        #      molecule's trajectory - so the signal does not jump around as
        #      the ligand moves.
        #
        #   3. Evaluate MACE twice (mace_pocket_force_with_h):
        #        ligand + pocket  ->  F_complex , E_total
        #        ligand alone     ->  F_intra       , E_lig
        #      The difference is the interaction term:
        #        F_inter = F_complex - F_intra
        #        E_int = E_total     - E_lig
        #
        #   4. Turn those forces into a coordinate displacement. TWO modes are
        #      available and they are mutually exclusive - pick one in the
        #      YAML under sample.physical_guidance, see the branch below:
        #        complex_force      uses F_complex, weighted per atom
        #        interaction_aware  mixes F_complex and F_intra
        #
        #   5. Attenuate the step by proxy confidence, clamp it if any bond
        #      would stretch past max_stretch, then add it to the coordinates.
        #      The next denoising step proceeds as normal.
        #
        # All of this is inference-time only: nothing is trained and DiffGui's
        # own weights are never modified.
        # ==============================================================
        #
        # # Per-molecule setup.
        #    Clear run-level pocket cache so EACH molecule re-selects its own
        #    5Å pocket at its first guidance step (then fixed within that molecule).
        #    Also clear last_pocket — otherwise a molecule that never achieves a
        #    finite MACE force would wrongly inherit the previous molecule's pocket.
        if physical_guidance_config is not None and physical_guidance_config.get('enabled', False):
            physical_guidance_config.pop('pocket_fixed', None)
            physical_guidance_config.pop('pocket_atomic', None)
            physical_guidance_config.pop('pocket_resids_fixed', None)
            physical_guidance_config.pop('pocket_resnames_fixed', None)
            physical_guidance_config.pop('pocket_atom_indices_fixed', None)
            physical_guidance_config.pop('pocket_fixed_step', None)
            physical_guidance_config.pop('last_pocket', None)
            pg0 = physical_guidance_config   # short alias for this setup section,
                                         # read-only from here on
            guidance_log = []  # per-step stats accumulated without GPU→CPU sync
            # Trajectory-level guidance statistics.
            #   attempt      — every guidance event × molecule
            #   precheck_fail — Stage-1 too_bad (overvalence/badgeom/clash)
            #   repair_none / postcheck_fail / repair_ratio_fail — repair-stage failures (separated)
            #   rdkit_*       — RDKit stage failures (classified)
            #   mace_fail / nonfinite — MACE numerical issues
            #   mace_ready    — proxy constructed + finite MACE force obtained
            #   applied       — delta_phys actually written back to diffusion
            physical_stats = {'attempt': 0,
                              'precheck_fail': 0,
                              'repair_none': 0,      # minimal repair itself returned None (budget/N gate/aromatic-only)
                              'postcheck_fail': 0,   # repair done but post-check failed (overvalence/badgeom/disconnect/clash)
                              'repair_ratio_fail': 0, # repair done+clean but changed >40% topology
                              'kekulize_fail': 0, 'aromatic_fail': 0,
                              'valence_fail': 0, 'rdkit_other': 0,
                              'mace_fail': 0, 'mace_nonfinite': 0,
                              'mace_ready': 0, 'delta_nonfinite': 0,
                              'applied': 0, 'guidance_exception': 0,
                              'first_mace_step': None,
                              # proxy projection stats (single deterministic proxy)
                              'proxy_direct': 0, 'proxy_fail': 0, 'proxy_repaired': 0,
                              'proxy_added_bonds': 0, 'proxy_removed_bonds': 0,
                              'proxy_order_changes': 0, 'proxy_charge_adjustments': 0,
                              'proxy_aromatic_ambiguous': 0, 'proxy_aromatic_fallback': 0,
                              'proxy_tautomer_sum': 0.0, 'proxy_tautomer_n': 0,
                              # trajectory-level proxy confidence (weighted branch)
                              'proxy_conf_sum': 0.0, 'proxy_conf_min': None,
                              'proxy_conf_n': 0, 'effective_scale_sum': 0.0}
            if pg0.get('adaptive_start', False):
                guidance_active = False
                prev_atom_argmax = None
                prev_bond_argmax = None
                stable_steps = 0
                stable_threshold = pg0.get('stable_threshold', 0.95)
                stable_required = pg0.get('stable_steps', 10)
                min_step = pg0.get('min_step', 400)  # guidance forbidden before this step
            else:
                guidance_active = True  # fixed start_step mode
                min_step = pg0.get('start_step', 200)

        # # 2. sample loop
        ligand_node_h_pert = ligand_node_h_init
        ligand_pos_pert = ligand_pos_init
        ligand_halfedge_h_pert = ligand_halfedge_h_init
        ligand_edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
        ligand_edge_batch = torch.cat([halfedge_batch, halfedge_batch], dim=0)
        for i, step in tqdm(enumerate(range(self.num_timesteps)[::-1]), total=self.num_timesteps):
            time_step = torch.full(size=(n_graphs,), fill_value=step, dtype=torch.long).to(device)
            ligand_edge_h_pert = torch.cat([ligand_halfedge_h_pert, ligand_halfedge_h_pert], dim=0)
            
            # # 2.1 inference
            if self.config.train_mode in ('ori', 'no_bond'):
                pred_ligand_pos, pred_ligand_node, pred_ligand_halfedge = self.classifier_free(
                    protein_node, protein_pos, protein_batch,
                    ligand_node_h_pert, ligand_pos_pert, ligand_batch, 
                    ligand_edge_h_pert, ligand_edge_index, ligand_edge_batch, 
                    gui_strength, time_step, batch_lab
                )
            elif self.config.train_mode in ('no_lab', 'no_both'):
                preds = self(
                    protein_node, protein_pos, protein_batch,
                    ligand_node_h_pert, ligand_pos_pert, ligand_batch,
                    ligand_edge_h_pert, ligand_edge_index, ligand_edge_batch, 
                    time_step, batch_lab
                )
                pred_ligand_pos, pred_ligand_node, pred_ligand_halfedge = preds['pred_ligand_pos'], preds['pred_ligand_node'], preds['pred_ligand_halfedge']

            # # Adaptive stability tracking (before guidance trigger)
            #    Checked every 5 steps to avoid per-step GPU→CPU sync overhead.
            #    Tracking only begins once step <= min_step: in the noise-dominated
            #    regime (step > min_step) argmax is trivially stable (near-uniform
            #    predictions) and must not trigger guidance.
            if physical_guidance_config is not None and physical_guidance_config.get('enabled', False) \
                    and physical_guidance_config.get('adaptive_start', False) and not guidance_active \
                    and step <= min_step and (i % 5 == 0):
                cur_atom_argmax = pred_ligand_node.argmax(dim=-1)
                cur_bond_argmax = pred_ligand_halfedge.argmax(dim=-1)
                if prev_atom_argmax is not None:
                    stable_frac_atom = (cur_atom_argmax == prev_atom_argmax).float().mean().item()
                    stable_frac_bond = (cur_bond_argmax == prev_bond_argmax).float().mean().item()
                    if stable_frac_atom >= stable_threshold and stable_frac_bond >= stable_threshold:
                        stable_steps += 1
                    else:
                        stable_steps = 0
                prev_atom_argmax = cur_atom_argmax.clone()
                prev_bond_argmax = cur_bond_argmax.clone()
                if stable_steps >= stable_required:
                    guidance_active = True
                    print(f"  [adaptive] guidance activated at step={step} "
                          f"(types stable for {stable_steps} steps)")

            # # 2.2 get the t - 1 state
            # pos
            ligand_pos_prev = self.pos_transition.get_prev_from_recon(
                x_t=ligand_pos_pert, x_recon=pred_ligand_pos, t=time_step, batch=ligand_batch
            )
            if self.categorical_space == 'discrete':
                # node types
                log_node_recon = F.log_softmax(pred_ligand_node, dim=-1)
                log_node_type = self.node_transition.q_v_posterior(log_node_recon, log_node_type, time_step, ligand_batch, v0_prob=True)
                ligand_node_type_prev = log_sample_categorical(log_node_type)
                ligand_node_h_prev = self.node_transition.onehot_encode(ligand_node_type_prev)
                
                # halfedge types
                log_edge_recon = F.log_softmax(pred_ligand_halfedge, dim=-1)
                log_halfedge_type = self.edge_transition.q_v_posterior(log_edge_recon, log_halfedge_type, time_step, halfedge_batch, v0_prob=True)
                ligand_halfedge_type_prev = log_sample_categorical(log_halfedge_type)
                ligand_halfedge_h_prev = self.edge_transition.onehot_encode(ligand_halfedge_type_prev)
                
            else:
                ligand_node_h_prev = self.node_transition.get_prev_from_recon(
                    x_t=ligand_node_h_pert, x_recon=pred_ligand_node, t=time_step, batch=ligand_batch)
                ligand_halfedge_h_prev = self.edge_transition.get_prev_from_recon(
                    x_t=ligand_halfedge_h_pert, x_recon=pred_ligand_halfedge, t=time_step, batch=halfedge_batch)

            # # 2.3 use guidance to modify pos
            if self.config.train_mode not in ('no_bond', 'no_both'):
                if guidance is not None:
                    gui_type, gui_scale = guidance
                    if (gui_scale > 0):
                        with torch.enable_grad():
                            ligand_node_h_in = ligand_node_h_pert.detach()
                            ligand_pos_in = ligand_pos_pert.detach().requires_grad_(True)
                            pred_bondpredictor = bond_predictor(
                                protein_node, protein_pos, protein_batch,
                                ligand_node_h_in, ligand_pos_in, ligand_batch,
                                ligand_edge_index, ligand_edge_batch, time_step)
                            delta = self.bond_guidance(gui_type, gui_scale, pred_bondpredictor, ligand_pos_in, ligand_halfedge_type_prev, log_halfedge_type)
                        ligand_pos_prev = ligand_pos_prev + delta

            # # 2.3b physical guidance (batch-safe: supports any batch_size)
            if physical_guidance_config is not None and physical_guidance_config.get('enabled', False):
                pg = physical_guidance_config
                # step > 0: the final x̂0 prediction is returned as-is after the loop;
                # guidance at step 0 would modify ligand_pos_prev but never affect the
                # returned pred_ligand_pos, so it is wasted computation.
                if pg.get('adaptive_start', False):
                    trigger = step > 0 and guidance_active and (step % pg['interval'] == 0)
                else:
                    trigger = step > 0 and step <= pg['start_step'] \
                        and (pg['start_step'] - step) % pg['interval'] == 0
                if trigger:
                    pg_mode = pg.get('mode', 'mace_only')
                    try:
                        physical_stats['attempt'] += n_graphs  # per molecule in this step
                        n_mols = int(ligand_batch.max().item()) + 1
                        atom_counts = torch.bincount(ligand_batch)
                        atom_offsets = torch.cat([torch.tensor([0], device=device),
                                                  torch.cumsum(atom_counts[:-1], dim=0)])

                        # Global predictions — bulk GPU→CPU transfers (3 tensor copies
                        # instead of dozens of per-item .item() synchronizations)
                        atom_idx_cpu = (pred_ligand_node.argmax(dim=-1).detach().cpu().tolist())
                        all_elements = [AROMATIC_INDEX_TO_ATOMIC_NUMBER.get(a, 6) for a in atom_idx_cpu]
                        atom_softmax_g = pred_ligand_node.softmax(dim=-1)
                        atom_conf_g = atom_softmax_g.max(dim=-1).values  # stays on GPU for stats
                        atom_conf_cpu = atom_conf_g.detach().cpu().numpy()  # CPU copy for proxy confidence

                        halfedge_types = pred_ligand_halfedge.argmax(dim=-1)
                        # FULL posterior for EVERY halfedge (incl. no-bond edges):
                        # P(no), P(single), P(double), P(triple), P(aromatic).
                        # Consumed by project_to_mace_proxy to rebuild the unique
                        # MACE proxy — never written back to the diffusion state.
                        bond_softmax_g = pred_ligand_halfedge.softmax(dim=-1)
                        halfedge_pairs_cpu = halfedge_index.detach().cpu().numpy()
                        halfedge_probs_cpu = bond_softmax_g.detach().cpu().numpy()

                        # Extract per-molecule data (pure CPU loop)
                        mol_data = []
                        for mol_i in range(n_mols):
                            s = int(atom_offsets[mol_i].item())
                            n = int(atom_counts[mol_i].item())
                            pos_i = (pred_ligand_pos[s:s+n] + offset[mol_i]).detach().cpu().numpy()
                            elem_i = all_elements[s:s+n]
                            atom_type_idx_i = atom_idx_cpu[s:s+n]
                            bonds_i = []
                            bond_probs_i = {}
                            bond_type_probs_i = {}
                            for k in range(halfedge_pairs_cpu.shape[1]):
                                a = int(halfedge_pairs_cpu[0, k])
                                b = int(halfedge_pairs_cpu[1, k])
                                if s <= a < s + n and s <= b < s + n:
                                    key = (min(a - s, b - s), max(a - s, b - s))
                                    probs5 = halfedge_probs_cpu[k]
                                    bond_type_probs_i[key] = probs5
                                    t = int(probs5.argmax())
                                    if t > 0:
                                        bonds_i.append((key[0], key[1], t))
                                        bond_probs_i[key] = float(probs5[t])
                            mol_data.append({'pos': pos_i, 'elem': elem_i, 'bonds': bonds_i,
                                             'bond_probs': bond_probs_i,
                                             'bond_type_probs': bond_type_probs_i,
                                             'atom_type_idx': atom_type_idx_i,
                                             'atom_conf': atom_conf_cpu[s:s+n]})

                        # Compute MACE forces per molecule
                        # With readiness gating: fast joint check → minimal repair → RDKit+MACE.
                        # The pocket is selected lazily: only after a molecule passes
                        # readiness and achieves a finite MACE force is the pocket
                        # cached for the rest of ITS trajectory (not the whole run).
                        per_mol_mace_f = []
                        per_mol_F_pocket = []
                        per_mol_F_total = []
                        per_mol_E_lig = []   # ligand-only MACE energy (eV), for guidance energy trajectory
                        per_mol_E_int = []   # interaction energy E_total - E_lig (eV)
                        readiness = pg.get('readiness', False)
                        stats_counts = {'ok': 0, 'skipped_too_bad': 0,
                                        'skipped_repair_none': 0,
                                        'skipped_postcheck': 0,
                                        'skipped_ratio': 0,
                                        'rdkit_failed': 0,
                                        'rdkit_valence': 0, 'rdkit_kekulize': 0,
                                        'rdkit_aromatic': 0, 'rdkit_other': 0,
                                        'mace_failed': 0, 'mace_nonfinite': 0,
                                        'delta_nonfinite': 0,
                                        # proxy projection stats (single deterministic proxy)
                                        'proxy_direct': 0, 'proxy_fail': 0,
                                        'proxy_repaired': 0,
                                        'proxy_added_bonds': 0, 'proxy_removed_bonds': 0,
                                        'proxy_order_changes': 0, 'proxy_charge_adjustments': 0,
                                        'proxy_aromatic_ambiguous': 0, 'proxy_aromatic_fallback': 0}
                        for mol_i in range(n_mols):
                            md = mol_data[mol_i]

                            if readiness:
                                # Deterministic single proxy projection: predicted-clean
                                # state → ONE valid, connected, valence-valid,
                                # RDKit-sanitizable heavy-atom proxy (never calls MACE,
                                # never written back to the diffusion state). Replaces
                                # the legacy hard-skip gates (fast_joint_check /
                                # fast_minimal_repair / strict postcheck / repair_ratio);
                                # those functions are kept for diagnostics only.
                                proxy = project_to_mace_proxy(
                                    md['pos'], md['elem'], md['bonds'],
                                    bond_type_probs=md.get('bond_type_probs'),
                                    atom_type_idx=md.get('atom_type_idx'),
                                    atom_conf=md.get('atom_conf'))
                                if proxy is None:
                                    stats_counts['proxy_fail'] += 1
                                    physical_stats['proxy_fail'] += 1
                                    continue
                                p_bonds, p_charges, pstats = proxy
                                md['bonds'] = p_bonds
                                md['formal_charges'] = p_charges
                                md['proxy_confidence'] = pstats['proxy_confidence']
                                stats_counts['proxy_direct'] += pstats['proxy_direct']
                                stats_counts['proxy_repaired'] += pstats['proxy_repaired']
                                stats_counts['proxy_added_bonds'] += pstats['proxy_added_bonds']
                                stats_counts['proxy_removed_bonds'] += pstats['proxy_removed_bonds']
                                stats_counts['proxy_order_changes'] += pstats['proxy_order_changes']
                                stats_counts['proxy_charge_adjustments'] += pstats['proxy_charge_adjustments']
                                physical_stats['proxy_direct'] += pstats['proxy_direct']
                                physical_stats['proxy_repaired'] += pstats['proxy_repaired']
                                physical_stats['proxy_added_bonds'] += pstats['proxy_added_bonds']
                                physical_stats['proxy_removed_bonds'] += pstats['proxy_removed_bonds']
                                physical_stats['proxy_order_changes'] += pstats['proxy_order_changes']
                                physical_stats['proxy_charge_adjustments'] += pstats['proxy_charge_adjustments']
                                physical_stats['proxy_aromatic_ambiguous'] += pstats['proxy_aromatic_ambiguous']
                                physical_stats['proxy_aromatic_fallback'] += pstats['proxy_aromatic_fallback']
                                physical_stats['proxy_tautomer_sum'] += pstats.get('proxy_tautomer_conf', 1.0)
                                physical_stats['proxy_tautomer_n'] += 1
                            else:
                                md['formal_charges'] = None
                                md['proxy_confidence'] = 1.0

                                # ---- This molecule's physical evaluation ----
                                # Rebuild the ligand proxy (if readiness gating is on), pick the
                                # 5 A protein environment once, then evaluate MACE twice:
                                #     ligand + pocket -> F_complex, E_total
                                #     ligand alone    -> F_intra,       E_lig
                                # F_inter = F_complex - F_intra is the interaction term. Which of
                                # these the guidance actually uses is decided by the mode branch
                                # further down.
                            with torch.enable_grad():
                                try:
                                    if pg_mode == 'mace_only':
                                        # Lazy pocket selection: only after this molecule
                                        # passed readiness. Failed intermediates must not
                                        # decide the trajectory's pocket.
                                        if 'pocket_fixed' in pg:
                                            pkt_c = pg['pocket_fixed']
                                            pkt_e = pg['pocket_atomic']
                                            pkt_r = pg.get('pocket_resids_fixed')
                                            pkt_rn = pg.get('pocket_resnames_fixed')
                                            pkt_idx = pg.get('pocket_atom_indices_fixed')
                                        else:
                                            pkt_np = pg['protein_tensors']['pocket_coords']  # numpy already
                                            pkt_elems = pg['protein_tensors']['pocket_elements']
                                            pkt_resids = pg['protein_tensors']['pocket_resids']
                                            pkt_resnames = pg['protein_tensors'].get('pocket_resnames',
                                                np.array(['UNK'] * len(pkt_resids)))
                                            min_dists = np.min(np.linalg.norm(
                                                pkt_np[:, None, :] - md['pos'][None, :, :], axis=2), axis=1)
                                            nearby = set(np.unique(pkt_resids[min_dists < 5.0]))
                                            if len(nearby) == 0:
                                                physical_stats['precheck_fail'] += 1
                                                continue
                                            mask_p = np.array([r in nearby for r in pkt_resids], dtype=bool)
                                            pkt_c = pkt_np[mask_p]
                                            pkt_e = [int(pkt_elems[j]) for j in range(len(pkt_elems)) if mask_p[j]]
                                            pkt_r = np.array(pkt_resids)[mask_p] if not isinstance(pkt_resids, np.ndarray) else pkt_resids[mask_p]
                                            pkt_rn = np.array(pkt_resnames)[mask_p] if not isinstance(pkt_resnames, np.ndarray) else pkt_resnames[mask_p]
                                            pkt_idx = np.where(mask_p)[0]  # indices into parsed pocket atom arrays

                                        # Single MACE complex call returns everything:
                                        # (F_inter, E_inter, F_complex, F_intra, E_total)
                                        Fp_i, E_i, Ft_i, Fl_i, Et_i = mace_pocket_force_with_h(
                                            md['pos'], md['elem'], md['bonds'],
                                            pkt_c, pkt_e,
                                            formal_charges=md.get('formal_charges'))
                                    else:
                                        raise UnsupportedGuidanceMode(
                                            f"physical_guidance.mode is '{pg_mode}'. Only "
                                            f"'mace_only' is implemented — the AMBER path was "
                                            f"removed and per-residue force decomposition was "
                                            f"abandoned. sample.py rejects other modes when it "
                                            f"loads the config, so reaching this point means "
                                            f"the model was driven directly.")
                                except UnsupportedGuidanceMode:
                                    raise
                                except Exception as e:
                                    # Classify failure: RDKit vs MACE vs other
                                    emsg = str(e)
                                    if 'rdkit' in emsg.lower():
                                        stats_counts['rdkit_failed'] += 1
                                        if 'valence' in emsg.lower():
                                            stats_counts['rdkit_valence'] += 1
                                            physical_stats['valence_fail'] += 1
                                        elif 'kekul' in emsg.lower():
                                            stats_counts['rdkit_kekulize'] += 1
                                            physical_stats['kekulize_fail'] += 1
                                        elif 'aromatic' in emsg.lower():
                                            stats_counts['rdkit_aromatic'] += 1
                                            physical_stats['aromatic_fail'] += 1
                                        else:
                                            stats_counts['rdkit_other'] += 1
                                            physical_stats['rdkit_other'] += 1
                                    elif 'mace' in emsg.lower():
                                        stats_counts['mace_failed'] += 1
                                        physical_stats['mace_fail'] += 1
                                    else:
                                        stats_counts['rdkit_other'] += 1
                                        physical_stats['rdkit_other'] += 1
                                    continue

                                # NaN/Inf gate — MACE may return non-finite forces
                                if not (torch.isfinite(Fl_i).all()
                                        and torch.isfinite(Fp_i).all()
                                        and torch.isfinite(Ft_i).all()):
                                    stats_counts['mace_nonfinite'] += 1
                                    physical_stats['mace_nonfinite'] += 1
                                    continue

                                # MACE succeeded and is finite → NOW cache the pocket
                                # for the rest of THIS molecule's trajectory.
                                if pg_mode == 'mace_only' and 'pocket_fixed' not in pg:
                                    pg['pocket_fixed'] = pkt_c
                                    pg['pocket_atomic'] = pkt_e
                                    pg['pocket_resids_fixed'] = pkt_r
                                    pg['pocket_resnames_fixed'] = pkt_rn
                                    pg['pocket_atom_indices_fixed'] = pkt_idx
                                    pg['pocket_fixed_step'] = int(step)
                                    physical_stats['first_mace_step'] = int(step)
                                    print(f"  Pocket fixed: {len(pkt_e)} atoms at step={step} (per-molecule)")

                                per_mol_mace_f.append(Fl_i.float().to(device))
                                if pg_mode == 'mace_only':
                                    per_mol_F_pocket.append(Fp_i.float().to(device))
                                    per_mol_F_total.append(Ft_i.float().to(device))
                                    # E_lig = E_total - E_inter (algebraic identity from the
                                    # interaction-energy decomposition in mace_pocket_force_*)
                                    per_mol_E_lig.append(float(Et_i) - float(E_i))
                                    per_mol_E_int.append(float(E_i))
                                stats_counts['ok'] += 1
                                physical_stats['mace_ready'] += 1

                        # Reassemble forces
                        if not per_mol_mace_f:
                            # No molecule passed readiness this step — skip guidance entirely
                            if readiness and n_mols > 0:
                                if step == pg['start_step'] or step % (pg['interval'] * 5) == 0:
                                    print(f"  [readiness] step={step:4d} all skipped: "
                                          f"{stats_counts}")
                            skip_guidance = True
                        else:
                            # Forces exist → always assemble (readiness only gates WHO
                            # gets in above, never the assembly below)
                            skip_guidance = False
                            mace_f = torch.cat(per_mol_mace_f, dim=0)
                            if pg_mode == 'mace_only':
                                F_pocket = torch.cat(per_mol_F_pocket, dim=0)
                                F_total_lig = torch.cat(per_mol_F_total, dim=0)

                            # Readiness diagnostics (logging only, no control flow)
                            if readiness:
                                if step == pg['start_step'] or step % (pg['interval'] * 5) == 0:
                                    try:
                                        ac = atom_conf_g.detach().cpu().numpy()
                                        print(f"  [readiness] step={step:4d} atom_conf: "
                                              f"mean={ac.mean():.3f} min={ac.min():.3f} "
                                              f"<0.6:{(ac < 0.6).mean():.2f} "
                                              f"ok={stats_counts['ok']} "
                                              f"precheck={stats_counts['skipped_too_bad']} "
                                              f"repair_none={stats_counts['skipped_repair_none']} "
                                              f"postcheck={stats_counts['skipped_postcheck']} "
                                              f"ratio={stats_counts['skipped_ratio']} "
                                              f"rdkit={stats_counts['rdkit_failed']}"
                                              f"(v:{stats_counts['rdkit_valence']} "
                                              f"k:{stats_counts['rdkit_kekulize']} "
                                              f"a:{stats_counts['rdkit_aromatic']} "
                                              f"o:{stats_counts['rdkit_other']})")
                                    except Exception:
                                        pass

                                                # ---- Turn the forces into a coordinate step ----
                        # Exactly one of the modes below must be enabled in
                        # the YAML. They are mutually exclusive and a config
                        # with none of them raises rather than silently
                        # running with no guidance at all.
                        #   complex_force      -> F_complex, per-atom weighted
                        #   interaction_aware  -> mixes F_complex and F_intra
                        if pg_mode == 'mace_only' and not skip_guidance:
                            if pg.get('complex_force', False):
                                # ── complex_force: direction-only, per-atom force magnitudes kept ──
                                #
                                # Formula:  Δx_i = dir_scale × m_t × w_i × F̂_i
                                #
                                #   m_t  = (1/N) Σ_j ||x̃_{t-1,j} − x_{t,j}||     global mean model step
                                #          (inherits DiffGui's natural annealing as t→0)
                                #
                                #   w_i  = min( g_i / mean(g_j),  w_max )           per-atom weight, mean≈1
                                #   g_i  = c × tanh(f_i / (s_F × c))                 soft-compressed force
                                #   s_F  = (1/N) Σ_j min(f_j, Q90({f_j}))            robust "typical force" scale
                                #   f_i  = ||F_i||                                   raw MACE force magnitude
                                #   F̂_i  = F_i / (f_i + ε)                           unit direction
                                #
                                # Parameters (in YAML: sample.physical_guidance):
                                #   complex_force: true               # enable
                                #   dir_scale: 6.0                    # global strength (same as v1)
                                #   tanh_c: 3.0                       # trust ceiling: c=∞ (linear),
                                #                                     #   c=2 (conservative), sweep {2, 3, 5, inf}
                                #   w_max: 3.0                        # optional hard safety cap (default 3)
                                #   quantile: 0.90                    # fraction for robust scale (default 0.90)
                                #
                                # Control: set w_i = 1.0 to recover v1 baseline

                                # 1. Force direction (per-atom unit vector)
                                F_mag = F_total_lig.norm(dim=-1, keepdim=True)          # (N, 1)
                                F_dir = F_total_lig / (F_mag + 1e-8)                    # (N, 3)

                                # 2. Global model step (scalar, not per-atom)
                                model_disp = ligand_pos_prev - ligand_pos_pert           # (N, 3)
                                m_t = model_disp.norm(dim=-1).mean()                     # scalar

                                # 3. Robust force scale — 90th percentile cap + mean
                                #    Shields s_F from spurious outlier atoms
                                qp = pg.get('quantile', 0.90)
                                q_val = torch.quantile(F_mag, qp)                        # scalar
                                s_F = torch.clamp(F_mag, max=q_val).mean()               # scalar

                                # 4. Relative force → soft compression (tanh) → mean-normalize
                                #    Small forces (r_i << c):  g_i ≈ r_i  (linear, faithful)
                                #    Large forces (r_i >> c):  g_i → c    (saturated, safe)
                                r_i = F_mag / (s_F + 1e-8)                               # (N, 1)
                                c = pg.get('tanh_c', 3.0)
                                if c is None or c == float('inf'):
                                    g_i = r_i                                           # linear, no compression
                                    c_str = 'inf'
                                else:
                                    g_i = c * torch.tanh(r_i / c)                       # (N, 1), soft-clamped to c
                                    c_str = str(c)
                                w_raw = g_i / (g_i.mean() + 1e-8)                      # (N, 1), mean ≈ 1.0

                                # 5. Hard safety cap — record whether it actually triggers
                                w_max = pg.get('w_max', 3.0)
                                clipped_flag = (w_raw > w_max).any()
                                w_i = torch.clamp(w_raw, max=w_max)

                                # 6. Final displacement
                                delta_phys = pg.get('dir_scale', 0.1) * m_t * w_i * F_dir

                                # 6b. Proxy-confidence scaling: the force is
                                # trusted as much as the ORIGINAL prediction
                                # supports the proxy (not how many edits were
                                # made). alpha_proxy (YAML:
                                # sample.physical_guidance.alpha_proxy, default
                                # 1.0) is a global knob; alpha_min (default
                                # 0.2) floors the per-mol factor so an
                                # uncertain proxy still gets a gentle push:
                                #   alpha_eff = alpha_proxy * (alpha_min + (1-alpha_min)*C_proxy)
                                # Direct proxies (C=1) with alpha_proxy=1 give
                                # alpha_eff=1, reproducing the old behaviour.
                                alpha_proxy = pg.get('alpha_proxy', 1.0)
                                alpha_min = pg.get('alpha_min', 0.2)
                                conf_parts = []
                                for mol_i in range(n_mols):
                                    c = mol_data[mol_i].get('proxy_confidence', 1.0)
                                    conf_parts.append(torch.full(
                                        (int(atom_counts[mol_i].item()), 1), c, device=device))
                                proxy_conf_t = torch.cat(conf_parts, dim=0)
                                alpha_eff_t = alpha_proxy * (alpha_min + (1.0 - alpha_min) * proxy_conf_t)
                                delta_phys = delta_phys * alpha_eff_t
                                conf_mean = proxy_conf_t.mean().item()
                                alpha_eff_mean = alpha_eff_t.mean().item()
                                # trajectory-level confidence stats for CSV
                                physical_stats['proxy_conf_sum'] += conf_mean
                                physical_stats['proxy_conf_n'] += 1
                                if physical_stats['proxy_conf_min'] is None \
                                        or conf_mean < physical_stats['proxy_conf_min']:
                                    physical_stats['proxy_conf_min'] = conf_mean
                                physical_stats['effective_scale_sum'] += alpha_eff_mean

                                # Post-push safety clamp — aligned with the split
                                # branch: if any bond would stretch beyond
                                # max_stretch (A), scale the whole step down
                                # (trust-region clamp).
                                worst = 0.0
                                max_stretch = float(pg.get('max_stretch', 0.3))
                                stretch_clamped = False
                                if pg.get('clamp_stretch', True):
                                    pos_cur = ligand_pos_prev
                                    pos_new = pos_cur + delta_phys
                                    for mol_i in range(n_mols):
                                        md = mol_data[mol_i]
                                        s = int(atom_offsets[mol_i].item())
                                        n = int(atom_counts[mol_i].item())
                                        for (a, b, o) in md['bonds']:
                                            d0 = (pos_cur[s+a] - pos_cur[s+b]).norm().item()
                                            d1 = (pos_new[s+a] - pos_new[s+b]).norm().item()
                                            st_ = abs(d1 - d0)
                                            if st_ > worst:
                                                worst = st_
                                    if worst > max_stretch:
                                        delta_phys = delta_phys * (max_stretch / worst)
                                        stretch_clamped = True
                                    else:
                                        stretch_clamped = False

                                # Per-step log accumulation (no sync here)
                                guidance_log.append({
                                    'step': step,
                                    'm_t': m_t.detach(),
                                    'dx_mean': delta_phys.norm(dim=-1).mean().detach(),
                                    'w_min': w_i.min().detach(),
                                    'w_max': w_i.max().detach(),
                                    'proxy_conf': conf_mean,
                                    'effective_scale': alpha_eff_mean,
                                    'stretch_clamped': int(stretch_clamped),
                                })


                            elif pg.get('interaction_aware', False):
                                # ── interaction_aware: magnitude-preserving mix of F_intra and F_inter ──
                                # Mix at the FORCE level, NOT direction voting:
                                #   F_eff,i = (1-alpha)*F_intra,i + alpha*F_inter,i
                                # alpha weights the two physical force components
                                # while each keeps its real magnitude; which one
                                # dominates depends on ||F_intra||, ||F_inter||, cosθ.
                                # (Old bug: both forces were normalized to unit
                                #  vectors first — direction voting erased the
                                #  natural size relation and let E_lig be flipped.)
                                # alpha = F_inter amplification factor:
                                #   F_eff = F_intra + alpha*F_inter
                                # F_intra keeps its full magnitude (E_lig is never
                                # weakened by alpha); alpha=0 -> pure F_intra,
                                # alpha=1 -> natural resultant F_complex.
                                F_int = F_total_lig - mace_f
                                alpha = pg.get('alpha', 5.0)
                                F_mixed = mace_f + alpha * F_int

                                F_mag = F_mixed.norm(dim=-1, keepdim=True)
                                mace_dir = F_mixed / (F_mag + 1e-8)

                                model_disp = ligand_pos_prev - ligand_pos_pert
                                m_t = model_disp.norm(dim=-1).mean()
                                qp = pg.get('quantile', 0.90)
                                q_val = torch.quantile(F_mag, qp)
                                s_F = torch.clamp(F_mag, max=q_val).mean()
                                r_i = F_mag / (s_F + 1e-8)
                                c = pg.get('tanh_c', 3.0)
                                if c is None or c == float('inf') or (isinstance(c, str) and c.lower() == 'inf'):
                                    g_i = r_i; c_str = 'inf'
                                else:
                                    c = float(c); g_i = c * torch.tanh(r_i / c)
                                    c_str = str(int(c)) if c == int(c) else str(c)
                                w_raw = g_i / (g_i.mean() + 1e-8)
                                w_max = pg.get('w_max', 3.0)
                                clipped_flag = (w_raw > w_max).any()
                                w_i = torch.clamp(w_raw, max=w_max)

                                delta_phys = pg.get('dir_scale', 6.0) * m_t * w_i * mace_dir

                                # Direction analysis: how much of this step's
                                # displacement actually goes toward F_inter.
                                F_int_dir = F_int / (F_int.norm(dim=-1, keepdim=True) + 1e-8)
                                F_total_dir = F_total_lig / (F_total_lig.norm(dim=-1, keepdim=True) + 1e-8)
                                dir_analysis = {
                                    'cos_g_inter': float((mace_dir * F_int_dir).sum(dim=-1).mean().item()),
                                    'cos_t_inter': float((F_total_dir * F_int_dir).sum(dim=-1).mean().item()),
                                    'proj_g': float((delta_phys * F_int_dir).sum(dim=-1).mean().item()),
                                    'proj_t': float(((delta_phys.norm(dim=-1, keepdim=True) * F_total_dir) * F_int_dir).sum(dim=-1).mean().item()),
                                }

                                # Proxy-confidence scaling (same as the weighted
                                # branch): an unreliable proxy gets a weaker push.
                                alpha_proxy = pg.get('alpha_proxy', 1.0)
                                alpha_min = pg.get('alpha_min', 0.2)
                                conf_parts = []
                                for mol_i in range(n_mols):
                                    c = mol_data[mol_i].get('proxy_confidence', 1.0)
                                    conf_parts.append(torch.full(
                                        (int(atom_counts[mol_i].item()), 1), c, device=device))
                                proxy_conf_t = torch.cat(conf_parts, dim=0)
                                alpha_eff_t = alpha_proxy * (alpha_min + (1.0 - alpha_min) * proxy_conf_t)
                                delta_phys = delta_phys * alpha_eff_t
                                conf_mean = proxy_conf_t.mean().item()
                                alpha_eff_mean = alpha_eff_t.mean().item()
                                # trajectory-level confidence stats for CSV
                                physical_stats['proxy_conf_sum'] += conf_mean
                                physical_stats['proxy_conf_n'] += 1
                                if physical_stats['proxy_conf_min'] is None \
                                        or conf_mean < physical_stats['proxy_conf_min']:
                                    physical_stats['proxy_conf_min'] = conf_mean
                                physical_stats['effective_scale_sum'] += alpha_eff_mean

                                # Per-atom diagnostics: model's remaining
                                # displacement |x0_pred - x_t| vs guidance push.
                                # + per-atom direction: cos(Δx_i, F_inter,i) — how much
                                # of this atom's push actually points along the
                                # interaction force (paper evidence for split).
                                if pg.get('guidance_atom_log'):
                                    amd = (pred_ligand_pos - ligand_pos_pert).norm(dim=-1).detach().cpu().numpy()
                                    agd = delta_phys.norm(dim=-1).detach().cpu().numpy()
                                    d_np = delta_phys.detach().cpu().numpy()
                                    fi_np = F_int.detach().cpu().numpy()
                                    fi_n = np.linalg.norm(fi_np, axis=-1) + 1e-8
                                    d_n = np.linalg.norm(d_np, axis=-1) + 1e-8
                                    cos_gi = (d_np * fi_np).sum(axis=-1) / (d_n * fi_n)
                                    # per-atom force magnitudes: |F_intra|, |F_inter| and
                                    # |F_complex| (needed to reconstruct the total-mode
                                    # per-atom weight w_i if the paper wants the v2
                                    # displacement; |F_intra| is not derivable from the
                                    # other two because the F_intra-F_inter angle is unknown)
                                    fl_np = mace_f.detach().cpu().numpy()
                                    fl_n = np.linalg.norm(fl_np, axis=-1)
                                    ft_np = F_total_lig.detach().cpu().numpy()
                                    ft_n = np.linalg.norm(ft_np, axis=-1)
                                    for mol_i in range(n_mols):
                                        s = int(atom_offsets[mol_i].item())
                                        n = int(atom_counts[mol_i].item())
                                        for ai in range(n):
                                            pg['atom_log_data'].append(
                                                (step, mol_i, ai, float(amd[s+ai]), float(agd[s+ai]),
                                                 float(cos_gi[s+ai]), float(fl_n[s+ai]),
                                                 float(fi_n[s+ai]), float(ft_n[s+ai])))


                                # Post-push safety: proxy confidence protects the
                                # INPUT structure; this clamps the displacement so
                                # the OUTPUT structure stays intact. If any bond
                                # would stretch beyond max_stretch (A), scale the
                                # whole step down (trust-region clamp).
                                # Initialized outside so the guidance log and the
                                # diagnostic print can read them even when
                                # clamp_stretch is disabled.
                                worst = 0.0
                                max_stretch = float(pg.get('max_stretch', 0.3))
                                stretch_clamped = False
                                if pg.get('clamp_stretch', True):
                                    pos_cur = ligand_pos_prev
                                    pos_new = pos_cur + delta_phys
                                    worst = 0.0
                                    for mol_i in range(n_mols):
                                        md = mol_data[mol_i]
                                        s = int(atom_offsets[mol_i].item())
                                        n = int(atom_counts[mol_i].item())
                                        for (a, b, o) in md['bonds']:
                                            d0 = (pos_cur[s+a] - pos_cur[s+b]).norm().item()
                                            d1 = (pos_new[s+a] - pos_new[s+b]).norm().item()
                                            st_ = abs(d1 - d0)
                                            if st_ > worst:
                                                worst = st_
                                    if worst > max_stretch:
                                        delta_phys = delta_phys * (max_stretch / worst)
                                        stretch_clamped = True
                                    else:
                                        stretch_clamped = False

                                # Per-step guidance log (accumulated without GPU→CPU
                                # sync beyond the per-step tensors, dumped after each
                                # molecule). These columns are the paper evidence for
                                # "split is necessary": direction agreement with F_inter
                                # (cos_g_inter vs cos_t_inter), the force-magnitude ratio R
                                # that motivates alpha, and the MACE energy trajectory.
                                dx_mean = delta_phys.norm(dim=-1).mean().item()
                                f_lig_mag = mace_f.norm(dim=-1).mean().item()
                                f_int_mag = F_int.norm(dim=-1).mean().item()
                                # F_complex_mag: the un-mixed resultant (what the
                                # total/v2 mode would push with) — lets the paper
                                # reconstruct the total-mode displacement, not just
                                # its direction (cos_t_inter).
                                f_total_mag = F_total_lig.norm(dim=-1).mean().item()
                                # F_eff_mag: the actual mixed force |F_intra + α·F_inter|
                                f_eff_mag = F_mag.mean().item()
                                guidance_log.append({
                                    'step': step,
                                    'm_t': m_t.detach(),
                                    'dx_mean': dx_mean,
                                    'F_intra_mag': f_lig_mag,
                                    'F_inter_mag': f_int_mag,
                                    'F_complex_mag': f_total_mag,
                                    'F_eff_mag': f_eff_mag,
                                    'R': f_int_mag / (f_lig_mag + 1e-8),
                                    's_F': s_F.item(),
                                    'w_min': w_i.min().detach(),
                                    'w_max': w_i.max().detach(),
                                    'clipped': int(clipped_flag.item()),
                                    'proxy_conf': conf_mean,
                                    'effective_scale': alpha_eff_mean,
                                    'cos_g_inter': dir_analysis['cos_g_inter'],
                                    'cos_t_inter': dir_analysis['cos_t_inter'],
                                    'proj_g': dir_analysis['proj_g'],
                                    'proj_t': dir_analysis['proj_t'],
                                    'stretch_clamped': int(stretch_clamped),
                                    'E_intra': float(np.mean(per_mol_E_lig)) if per_mol_E_lig else float('nan'),
                                    'E_inter': float(np.mean(per_mol_E_int)) if per_mol_E_int else float('nan'),
                                })


                            else:
                                raise UnsupportedGuidanceMode(
                                    f"physical_guidance enabled but no known mode is set. "
                                    f"Set 'complex_force: true' or 'interaction_aware: true'. "
                                    f"(Other modes existed during development and were removed.)")
                        else:
                            # AMBER mode removed (2026-08): all guidance runs in
                            # mace_only mode; non-mace_only configs skip guidance.
                            skip_guidance = True

                        if not skip_guidance:
                            # Final finite gate on the assembled displacement itself
                            if not torch.isfinite(delta_phys).all():
                                stats_counts['delta_nonfinite'] += 1
                                physical_stats['delta_nonfinite'] += 1
                            else:
                                ligand_pos_prev = ligand_pos_prev + delta_phys * pg.get('overall_scale', 1.0)
                                physical_stats['applied'] += 1
                            # NOTE: no torch.cuda.empty_cache() on the success path —
                            # it would be freed and immediately re-allocated next
                            # guidance step, adding allocator overhead.
                    except Exception as e:
                        physical_stats['guidance_exception'] += 1
                        print(f"  [WARN] Physical guidance failed at step {step}: {e}")
                        torch.cuda.empty_cache()
                        pass

            # 2.4 log update
            ligand_node_traj[i+1] = ligand_node_h_prev
            ligand_pos_traj[i+1] = ligand_pos_prev + offset[ligand_batch]
            ligand_halfedge_traj[i+1] = ligand_halfedge_h_prev

            # # 2.5 update t-1
            ligand_pos_pert = ligand_pos_prev
            ligand_node_h_pert = ligand_node_h_prev
            ligand_halfedge_h_pert = ligand_halfedge_h_prev

        pred_ligand_pos = pred_ligand_pos + offset[ligand_batch]
        # # 3. dump per-step guidance log (one-time sync, after sampling)
        if physical_guidance_config is not None and physical_guidance_config.get('enabled', False) \
                and physical_guidance_config.get('guidance_log') and guidance_log:
            import os
            log_path = physical_guidance_config['guidance_log']
            header_needed = (not os.path.exists(log_path)) or os.path.getsize(log_path) == 0
            def _f(x, spec):
                # Missing keys (v2/total branch has no cos/proj/magnitude fields)
                # must dump as empty, never format a str with a numeric spec.
                if x is None or x == '':
                    return ''
                try:
                    return f'{x:{spec}}'
                except (ValueError, TypeError):
                    return f'{x}'

            with open(log_path, 'a') as gf:
                if header_needed:
                    gf.write('step,m_t,dx_mean,F_intra_mag,F_inter_mag,F_complex_mag,F_eff_mag,'
                             'R,s_F,w_min,w_max,clipped,proxy_conf,effective_scale,'
                             'cos_g_inter,cos_t_inter,proj_g,proj_t,stretch_clamped,E_intra,E_inter\n')
                for e in guidance_log:
                    gf.write(f"{e['step']},{float(e['m_t']):.6f},{float(e['dx_mean']):.6f},"
                             f"{_f(e.get('F_intra_mag'), '.3f')},{_f(e.get('F_inter_mag'), '.3f')},"
                             f"{_f(e.get('F_complex_mag'), '.3f')},{_f(e.get('F_eff_mag'), '.3f')},"
                             f"{_f(e.get('R'), '.3f')},{_f(e.get('s_F'), '.3f')},"
                             f"{float(e['w_min']):.4f},{float(e['w_max']):.4f},"
                             f"{_f(e.get('clipped'), 'd')},"
                             f"{_f(e.get('proxy_conf'), '.4f')},{_f(e.get('effective_scale'), '.4f')},"
                             f"{_f(e.get('cos_g_inter'), '.4f')},{_f(e.get('cos_t_inter'), '.4f')},"
                             f"{_f(e.get('proj_g'), '.4f')},{_f(e.get('proj_t'), '.4f')},"
                             f"{_f(e.get('stretch_clamped'), 'd')},{_f(e.get('E_intra'), '.3f')},{_f(e.get('E_inter'), '.3f')}\n")

            # per-atom displacement log (model remaining vs guidance push;
            # split branch appends extra columns: cos(Δx_i, F_inter,i),
            # |F_inter,i|, |F_complex,i|; v2 branch keeps 5 columns)
            if physical_guidance_config.get('guidance_atom_log'):
                al = physical_guidance_config.get('atom_log_data')
                if al:
                    with open(physical_guidance_config['guidance_atom_log'], 'a') as af:
                        for rec in al:
                            st_, mi, ai, md, gd = rec[:5]
                            extra = ','.join(f"{v:.4f}" for v in rec[5:])
                            af.write(f"{st_},{mi},{ai},{md:.6f},{gd:.6f}"
                                     + (f",{extra}" if extra else "") + "\n")
                    al.clear()

        # # 3b. trajectory-level guidance success summary
        if physical_guidance_config is not None and physical_guidance_config.get('enabled', False):
            physical_guidance_config['last_stats'] = dict(physical_stats)  # for sample.py CSV

            # Snapshot this molecule's fixed pocket (deep copies — pg is cleared
            # at the start of the next molecule).
            if 'pocket_fixed' in physical_guidance_config:
                physical_guidance_config['last_pocket'] = {
                    'coords': np.asarray(physical_guidance_config['pocket_fixed']).copy(),
                    'elements': np.asarray(physical_guidance_config['pocket_atomic']).copy(),
                    'resids': np.asarray(physical_guidance_config['pocket_resids_fixed']).copy(),
                    'resnames': np.asarray(physical_guidance_config['pocket_resnames_fixed']).copy(),
                    'atom_indices': np.asarray(physical_guidance_config['pocket_atom_indices_fixed']).copy(),
                    'fixed_step': physical_guidance_config.get('pocket_fixed_step'),
                }
            else:
                physical_guidance_config['last_pocket'] = None
            n_attempt = max(physical_stats['attempt'], 1)
            r_ready = physical_stats['mace_ready'] / n_attempt
            r_applied = physical_stats['applied'] / n_attempt
            print(f"  [guidance-stat] attempt={physical_stats['attempt']} "
                  f"mace_ready={physical_stats['mace_ready']} "
                  f"applied={physical_stats['applied']} "
                  f"R_ready={r_ready:.2f} R_applied={r_applied:.2f} "
                  f"precheck={physical_stats['precheck_fail']} "
                  f"repair_none={physical_stats['repair_none']} "
                  f"postcheck={physical_stats['postcheck_fail']} "
                  f"ratio={physical_stats['repair_ratio_fail']} "
                  f"kekulize={physical_stats['kekulize_fail']} "
                  f"aromatic={physical_stats['aromatic_fail']} "
                  f"valence={physical_stats['valence_fail']} "
                  f"rdkit_other={physical_stats['rdkit_other']} "
                  f"mace_fail={physical_stats['mace_fail']} "
                  f"mace_nf={physical_stats['mace_nonfinite']} "
                  f"delta_nf={physical_stats['delta_nonfinite']} "
                  f"exc={physical_stats['guidance_exception']}")
        # # get the final positions
        return {
            'pred': [pred_ligand_node, pred_ligand_pos, pred_ligand_halfedge],
            'traj': [ligand_node_traj, ligand_pos_traj, ligand_halfedge_traj]
        }

    @torch.no_grad()
    def sample_frag(
        self, n_graphs,
        protein_node, protein_pos, protein_batch,
        frag_node, frag_pos, frag_batch,
        frag_halfedge_type, frag_halfedge_index, frag_halfedge_batch,
        ligand_batch, halfedge_index, halfedge_batch,
        batch_lab=None, gui_strength=None,
        bond_predictor=None, guidance=None, gen_mode=None,
        physical_guidance_config=None,
    ):
        device = ligand_batch.device
        # # 1. get the init values (position, node and edge types)
        n_nodes_ligand = len(ligand_batch)
        n_halfedges_ligand = len(halfedge_batch)        
        node_init = self.node_transition.sample_init(n_nodes_ligand)
        halfedge_init = self.edge_transition.sample_init(n_halfedges_ligand)
        if self.categorical_space == 'discrete':
            _, ligand_node_h_init, log_node_type = node_init
            _, ligand_halfedge_h_init, log_halfedge_type = halfedge_init
        else:
            ligand_node_h_init = node_init
            ligand_halfedge_h_init = halfedge_init

        frag_node_mask = get_fragment_mask(ligand_batch, frag_batch)
        frag_halfedge_mask = get_fragment_mask(halfedge_batch, frag_halfedge_batch)
        pocket_center_pos = scatter_mean(protein_pos, protein_batch, dim=0)
        ligand_center_pos = pocket_center_pos[ligand_batch]
        ligand_pos_init = ligand_center_pos + torch.randn_like(ligand_center_pos)
        protein_pos, ligand_pos_init, offset = center_pos(protein_pos, ligand_pos_init, protein_batch, ligand_batch, self.center_pos_mode)
        frag_pos = frag_pos - offset[frag_batch]

        # # 1.1 init trajectory
        ligand_node_traj = torch.zeros([self.num_timesteps + 1, n_nodes_ligand, ligand_node_h_init.shape[-1]],
                                dtype=ligand_node_h_init.dtype).to(device)
        ligand_pos_traj = torch.zeros([self.num_timesteps + 1, n_nodes_ligand, 3], dtype=ligand_pos_init.dtype).to(device)
        ligand_halfedge_traj = torch.zeros([self.num_timesteps + 1, n_halfedges_ligand, ligand_halfedge_h_init.shape[-1]],
                                    dtype=ligand_halfedge_h_init.dtype).to(device)
        ligand_node_traj[0] = ligand_node_h_init
        ligand_pos_traj[0] = ligand_pos_init + offset[ligand_batch]
        ligand_halfedge_traj[0] = ligand_halfedge_h_init

        # # 2. sample loop
        ligand_node_h_pert = ligand_node_h_init
        ligand_pos_pert = ligand_pos_init
        ligand_halfedge_h_pert = ligand_halfedge_h_init
        ligand_edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
        ligand_edge_batch = torch.cat([halfedge_batch, halfedge_batch], dim=0)
        for i, step in tqdm(enumerate(range(self.num_timesteps)[::-1]), total=self.num_timesteps):
            time_step = torch.full(size=(n_graphs,), fill_value=step, dtype=torch.long).to(device)

            # # 2.1 get the init values for fragment
            if gen_mode == 'frag_cond':
                frag_pos_pert = frag_pos
                log_frag_node_type = index_to_log_onehot(frag_node, self.ligand_node_types)
                frag_node_pert = F.one_hot(log_sample_categorical(index_to_log_onehot(frag_node, self.ligand_node_types)), self.ligand_node_types).float()
                log_frag_halfedge_type = index_to_log_onehot(frag_halfedge_type, self.num_edge_types)
                frag_halfedge_pert = F.one_hot(log_sample_categorical(index_to_log_onehot(frag_halfedge_type, self.num_edge_types)), self.num_edge_types).float()
            elif gen_mode == 'frag_diff':
                pos_pert = self.pos_transition.add_noise(frag_pos, time_step, frag_batch)
                node_pert = self.node_transition.add_noise(frag_node, time_step, frag_batch)
                halfedge_pert = self.edge_transition.add_noise(frag_halfedge_type, time_step, frag_halfedge_batch)
                
                if self.categorical_space == 'discrete':
                    frag_node_pert, log_frag_node_type, _ = node_pert
                    frag_halfedge_pert, log_frag_halfedge_type, _ = halfedge_pert
                else:
                    frag_node_pert, _ = node_pert
                    frag_halfedge_pert, _ = halfedge_pert
                frag_pos_pert = pos_pert

            # # 2.2 combine fragment and ligand
            ligand_pos_pert[frag_node_mask] = frag_pos_pert
            ligand_node_h_pert[frag_node_mask] = frag_node_pert
            ligand_halfedge_h_pert[frag_halfedge_mask] = frag_halfedge_pert
            log_node_type[frag_node_mask] = log_frag_node_type
            log_halfedge_type[frag_halfedge_mask] = log_frag_halfedge_type
            ligand_edge_h_pert = torch.cat([ligand_halfedge_h_pert, ligand_halfedge_h_pert], dim=0)
            
            # # 2.3 inference
            if self.config.train_mode in ('ori', 'no_bond'):
                pred_ligand_pos, pred_ligand_node, pred_ligand_halfedge = self.classifier_free(
                    protein_node, protein_pos, protein_batch,
                    ligand_node_h_pert, ligand_pos_pert, ligand_batch, 
                    ligand_edge_h_pert, ligand_edge_index, ligand_edge_batch, 
                    gui_strength, time_step, batch_lab
                )
            elif self.config.train_mode in ('no_lab', 'no_both'):
                preds = self(
                    protein_node, protein_pos, protein_batch,
                    ligand_node_h_pert, ligand_pos_pert, ligand_batch,
                    ligand_edge_h_pert, ligand_edge_index, ligand_edge_batch, 
                    time_step, batch_lab
                )
                pred_ligand_pos, pred_ligand_node, pred_ligand_halfedge = preds['pred_ligand_pos'], preds['pred_ligand_node'], preds['pred_ligand_halfedge']

            # # 2.4 get the t - 1 state
            # pos 
            ligand_pos_prev = self.pos_transition.get_prev_from_recon(
                x_t=ligand_pos_pert, x_recon=pred_ligand_pos, t=time_step, batch=ligand_batch
            )
            if self.categorical_space == 'discrete':
                # node types
                log_node_recon = F.log_softmax(pred_ligand_node, dim=-1)
                log_node_type = self.node_transition.q_v_posterior(log_node_recon, log_node_type, time_step, ligand_batch, v0_prob=True)
                ligand_node_type_prev = log_sample_categorical(log_node_type)
                ligand_node_h_prev = self.node_transition.onehot_encode(ligand_node_type_prev)
                
                # halfedge types
                log_edge_recon = F.log_softmax(pred_ligand_halfedge, dim=-1)
                log_halfedge_type = self.edge_transition.q_v_posterior(log_edge_recon, log_halfedge_type, time_step, halfedge_batch, v0_prob=True)
                ligand_halfedge_type_prev = log_sample_categorical(log_halfedge_type)
                ligand_halfedge_h_prev = self.edge_transition.onehot_encode(ligand_halfedge_type_prev)
                
            else:
                ligand_node_h_prev = self.node_transition.get_prev_from_recon(
                    x_t=ligand_node_h_pert, x_recon=pred_ligand_node, t=time_step, batch=ligand_batch)
                ligand_halfedge_h_prev = self.edge_transition.get_prev_from_recon(
                    x_t=ligand_halfedge_h_pert, x_recon=pred_ligand_halfedge, t=time_step, batch=halfedge_batch)

            # # 2.5 use guidance to modify pos
            if self.config.train_mode not in ('no_bond', 'no_both'):
                if guidance is not None:
                    gui_type, gui_scale = guidance
                    if (gui_scale > 0):
                        with torch.enable_grad():
                            ligand_node_h_in = ligand_node_h_pert.detach()
                            ligand_pos_in = ligand_pos_pert.detach().requires_grad_(True)
                            pred_bondpredictor = bond_predictor(
                                protein_node, protein_pos, protein_batch,
                                ligand_node_h_in, ligand_pos_in, ligand_batch,
                                ligand_edge_index, ligand_edge_batch, time_step)
                            delta = self.bond_guidance(gui_type, gui_scale, pred_bondpredictor, ligand_pos_in, ligand_halfedge_type_prev, log_halfedge_type)
                        ligand_pos_prev = ligand_pos_prev + delta

            # 2.6 update trajectory
            ligand_node_traj[i+1] = ligand_node_h_prev
            ligand_pos_traj[i+1] = ligand_pos_prev + offset[ligand_batch]
            ligand_halfedge_traj[i+1] = ligand_halfedge_h_prev

            # # 2.7 update t-1
            ligand_pos_pert = ligand_pos_prev
            ligand_node_h_pert = ligand_node_h_prev
            ligand_halfedge_h_pert = ligand_halfedge_h_prev

        pred_ligand_pos = pred_ligand_pos + offset[ligand_batch]

        # # 3. get the final positions
        return {
            'pred': [pred_ligand_node, pred_ligand_pos, pred_ligand_halfedge],
            'traj': [ligand_node_traj, ligand_pos_traj, ligand_halfedge_traj]
        }

    def bond_guidance(self, gui_type, gui_scale, pred_bondpredictor, ligand_pos_in, halfedge_type_prev, log_halfedge_type):
        if gui_type == 'entropy':
            prob_halfedge = torch.softmax(pred_bondpredictor, dim=-1)
            entropy = - torch.sum(prob_halfedge * torch.log(prob_halfedge + 1e-12), dim=-1)
            entropy = entropy.log().sum()
            delta = - torch.autograd.grad(entropy, ligand_pos_in)[0] * gui_scale
        elif gui_type == 'uncertainty':
            uncertainty = torch.sigmoid( -torch.logsumexp(pred_bondpredictor, dim=-1))
            uncertainty = uncertainty.log().sum()
            delta = - torch.autograd.grad(uncertainty, ligand_pos_in)[0] * gui_scale
        elif gui_type == 'uncertainty_bond':  # only for the predicted real bond (not no bond)
            prob = torch.softmax(pred_bondpredictor, dim=-1)
            uncertainty = torch.sigmoid( -torch.logsumexp(pred_bondpredictor, dim=-1))
            uncertainty = uncertainty.log()
            uncertainty = (uncertainty * prob[:, 1:].detach().sum(dim=-1)).sum()
            delta = - torch.autograd.grad(uncertainty, ligand_pos_in)[0] * gui_scale
        elif gui_type == 'entropy_bond':
            prob_halfedge = torch.softmax(pred_bondpredictor, dim=-1)
            entropy = - torch.sum(prob_halfedge * torch.log(prob_halfedge + 1e-12), dim=-1)
            entropy = entropy.log()
            entropy = (entropy * prob_halfedge[:, 1:].detach().sum(dim=-1)).sum()
            delta = - torch.autograd.grad(entropy, ligand_pos_in)[0] * gui_scale
        elif gui_type == 'logit_bond':
            ind_real_bond = ((halfedge_type_prev >= 1) & (halfedge_type_prev <= 4))
            idx_real_bond = ind_real_bond.nonzero().squeeze(-1)
            pred_real_bond = pred_bondpredictor[idx_real_bond, halfedge_type_prev[idx_real_bond]]
            pred = pred_real_bond.sum()
            delta = + torch.autograd.grad(pred, ligand_pos_in)[0] * gui_scale
        elif gui_type == 'logit':
            ind_bond_notmask = (halfedge_type_prev <= 4)
            idx_real_bond = ind_bond_notmask.nonzero().squeeze(-1)
            pred_real_bond = pred_bondpredictor[idx_real_bond, halfedge_type_prev[idx_real_bond]]
            pred = pred_real_bond.sum()
            delta = + torch.autograd.grad(pred, ligand_pos_in)[0] * gui_scale
        elif gui_type == 'crossent':
            prob_halfedge_type = log_halfedge_type.exp()[:, :-1]  # the last one is masked bond (not used in predictor)
            entropy = F.cross_entropy(pred_bondpredictor, prob_halfedge_type, reduction='none')
            entropy = entropy.log().sum()
            delta = - torch.autograd.grad(entropy, ligand_pos_in)[0] * gui_scale
        elif gui_type == 'crossent_bond':
            prob_halfedge_type = log_halfedge_type.exp()[:, 1:-1]  # the last one is masked bond. first one is no bond
            entropy = F.cross_entropy(pred_bondpredictor[:, 1:], prob_halfedge_type, reduction='none')
            entropy = entropy.log().sum()
            delta = - torch.autograd.grad(entropy, ligand_pos_in)[0] * gui_scale
        else:
            raise NotImplementedError(f'Guidance type {gui_type} is not implemented')
        
        return delta

