from typing import Optional, Callable

import torch
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn import SchNet
from torch_geometric.typing import OptTensor


class Flexible_latent_SchNet(SchNet):
    def __init__(
            self,
            latent_size: int = 128,
            hidden_channels: int = 128,
            num_filters: int = 128,
            num_interactions: int = 6,
            num_gaussians: int = 50,
            cutoff: float = 10.0,
            max_num_neighbors: int = 32,
            readout: str = 'add',
            dipole: bool = False,
            mean: Optional[float] = None,
            std: Optional[float] = None,
            atomref: OptTensor = None,
            interaction_graph: Optional[Callable] = None,
    ):
        super().__init__(hidden_channels, num_filters, num_interactions, num_gaussians, cutoff, interaction_graph, max_num_neighbors, readout, dipole, mean, std, atomref)
        self.latent_size = latent_size
        self.lin1 = Linear(hidden_channels, (hidden_channels+latent_size) // 2)
        self.lin2 = Linear((hidden_channels+latent_size) // 2, latent_size)
    def forward(self, z: Tensor, pos: Tensor,
                batch: OptTensor = None) -> Tensor:
        r"""
        Args:
            z (torch.Tensor): Atomic number of each atom with shape
                :obj:`[num_atoms]`.
            pos (torch.Tensor): Coordinates of each atom with shape
                :obj:`[num_atoms, 3]`.
            batch (torch.Tensor, optional): Batch indices assigning each atom
                to a separate molecule with shape :obj:`[num_atoms]`.
                (default: :obj:`None`)
        """
        batch = torch.zeros_like(z) if batch is None else batch

        h = self.embedding(z)
        edge_index, edge_weight = self.interaction_graph(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)

        for num,interaction in enumerate(self.interactions):
            h = h + interaction(h, edge_index, edge_weight, edge_attr)


        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        if self.dipole:
            # Get center of mass.
            mass = self.atomic_mass[z].view(-1, 1)
            M = self.sum_aggr(mass, batch, dim=0)
            c = self.sum_aggr(mass * pos, batch, dim=0) / M
            h = h * (pos - c.index_select(0, batch))

        if not self.dipole and self.mean is not None and self.std is not None:
            h = h * self.std + self.mean

        if not self.dipole and self.atomref is not None:
            h = h + self.atomref(z)

        out = self.readout(h, batch, dim=0)

        if self.dipole:
            out = torch.norm(out, dim=-1, keepdim=True)

        if self.scale is not None:
            out = self.scale * out

        return out
