import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord

import unfoldNd


@register('liif')
class LIIF(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True, device='cuda'):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        self.device = device

        self.encoder = models.make(encoder_spec)

        if imnet_spec is not None:
            imnet_in_dim = self.encoder.out_dim
            if self.feat_unfold:
                imnet_in_dim *= 27
            imnet_in_dim += 3 # attach coord
            if self.cell_decode:
                imnet_in_dim += 3
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        else:
            self.imnet = None

    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None, half=False):
        feat = self.feat

        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, 0, :] \
                .permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            feat = unfoldNd.unfoldNd(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 27, feat.shape[2], feat.shape[3], feat.shape[4])

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            vz_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, vz_lst, eps_shift = [0], [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-3] / 2
        ry = 2 / feat.shape[-2] / 2
        rz = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-3:], flatten=False).to(self.device) \
            .permute(3, 0, 1, 2) \
            .unsqueeze(0).expand(feat.shape[0], 3, *feat.shape[-3:])

        if half:
            feat_coord=feat_coord.half()

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                for vz in vz_lst:
                    coord_ = coord.clone()
                    coord_[:, :, 0] += vx * rx + eps_shift
                    coord_[:, :, 1] += vy * ry + eps_shift
                    coord_[:, :, 2] += vz * rz + eps_shift
                    coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                    q_feat = F.grid_sample(
                        feat, coord_.flip(-1).unsqueeze(1).unsqueeze(1),
                        mode='nearest', align_corners=False)[:, :, 0, 0, :] \
                        .permute(0, 2, 1)
                    q_coord = F.grid_sample(
                        feat_coord, coord_.flip(-1).unsqueeze(1).unsqueeze(1),
                        mode='nearest', align_corners=False)[:, :, 0, 0, :] \
                        .permute(0, 2, 1)
                    rel_coord = coord - q_coord
                    rel_coord[:, :, 0] *= feat.shape[-3]
                    rel_coord[:, :, 1] *= feat.shape[-2]
                    rel_coord[:, :, 2] *= feat.shape[-1]
                    inp = torch.cat([q_feat, rel_coord], dim=-1)

                    if self.cell_decode:
                        rel_cell = cell.clone()
                        rel_cell[:, :, 0] *= feat.shape[-3]
                        rel_cell[:, :, 1] *= feat.shape[-2]
                        rel_cell[:, :, 2] *= feat.shape[-1]
                        inp = torch.cat([inp, rel_cell], dim=-1)

                    bs, q = coord.shape[:2]
                    pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                    preds.append(pred)

                    rel_coord=rel_coord.float()
                    area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1] * rel_coord[:, :, 2])
                    areas.append(area + (1e-6 if half else 1e-9))

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            if half:
                ret = ret + pred * ((area / tot_area).half()).unsqueeze(-1)
            else:
                ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell, inp.dtype == torch.float16)
