import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord

import unfoldNd


@register('metasr')
class MetaSR(nn.Module):

    def __init__(self, encoder_spec, feat_unfold=True, device='cuda'):
        super().__init__()

        self.encoder = models.make(encoder_spec)
        if feat_unfold:
            imnet_spec = {
                'name': 'mlp',
                'args': {
                    'in_dim': 4,
                    'out_dim': self.encoder.out_dim * 27 * 1,
                    'hidden_list': [256, 256, 256, 256]
                }
            }
        else:
            imnet_spec = {
                'name': 'mlp',
                'args': {
                    'in_dim': 4,
                    'out_dim': self.encoder.out_dim * 1,
                    'hidden_list': [256, 256, 256, 256]
                }
            }
        self.imnet = models.make(imnet_spec)
        self.device=device
        self.feat_unfold = feat_unfold

    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None, half=False):
        feat = self.feat
        if self.feat_unfold:
            feat = unfoldNd.unfoldNd(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 27, feat.shape[2], feat.shape[3], feat.shape[4])

        feat_coord = make_coord(feat.shape[-3:], flatten=False).to(self.device)
        feat_coord[:, :, 0] -= (2 / feat.shape[-3]) / 2
        feat_coord[:, :, 1] -= (2 / feat.shape[-2]) / 2
        feat_coord[:, :, 3] -= (2 / feat.shape[-1]) / 2
        feat_coord = feat_coord.permute(3, 0, 1, 2) \
            .unsqueeze(0).expand(feat.shape[0], 3, *feat.shape[-3:])

        if half:
            feat_coord=feat_coord.half()

        coord_ = coord.clone()
        coord_[:, :, 0] -= cell[:, :, 0] / 2
        coord_[:, :, 1] -= cell[:, :, 1] / 2
        coord_[:, :, 2] -= cell[:, :, 2] / 2
        coord_q = (coord_ + 1e-6).clamp(-1 + 1e-6, 1 - 1e-6)
        q_feat = F.grid_sample(
            feat, coord_q.flip(-1).unsqueeze(1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, 0, :] \
            .permute(0, 2, 1)
        q_coord = F.grid_sample(
            feat_coord, coord_.flip(-1).unsqueeze(1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, 0, :] \
            .permute(0, 2, 1)

        rel_coord = coord_ - q_coord
        rel_coord[:, :, 0] *= feat.shape[-3]
        rel_coord[:, :, 1] *= feat.shape[-2]
        rel_coord[:, :, 2] *= feat.shape[-1]

        r_rev = cell[:, :, 0] * (feat.shape[-2] / 2)
        inp = torch.cat([rel_coord, r_rev.unsqueeze(-1)], dim=-1)

        bs, q = coord.shape[:2]
        pred = self.imnet(inp.view(bs * q, -1)).view(bs * q, feat.shape[1], 1)
        pred = torch.bmm(q_feat.contiguous().view(bs * q, 1, -1), pred)
        pred = pred.view(bs, q, 1)
        return pred

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)
