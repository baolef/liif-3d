import argparse
import os
from PIL import Image

import torch
from torchvision import transforms

import models
from utils import make_coord
from test import batched_predict

import ants


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.png')
    parser.add_argument('--model')
    parser.add_argument('--resolution')
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # img = transforms.ToTensor()(Image.open(args.input).convert('RGB'))

    img = ants.image_read(args.input).iMath_normalize()
    inp=torch.Tensor(img.numpy(True).transpose(3, 0, 1, 2))

    inp=inp[:,:64,:64,:64]

    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()

    h, w, d = list(map(int, args.resolution.split(',')))
    coord = make_coord((h, w, d)).cuda()
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    cell[:, 2] *= 2 / d
    pred = batched_predict(model, ((inp - 0.5) / 0.5).cuda().unsqueeze(0),
        coord.unsqueeze(0), cell.unsqueeze(0), bsize=500000)[0]

    pred = (pred * 0.5 + 0.5).view(h, w, d).cpu().numpy()
    print(pred.min(),pred.max())
    # transforms.ToPILImage()(pred).save(args.output)
    pred_img = ants.from_numpy(pred,origin=img.origin, spacing=tuple(img.spacing), direction=img.direction)
    print(pred_img.min(),pred_img.max())
    # pred_img.iMath_normalize()
    ants.image_write(pred_img,args.output)
