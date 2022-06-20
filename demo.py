import argparse
import os
from PIL import Image

import torch
from torchvision import transforms

import models
from utils import make_coord
from test import batched_predict, block_predict

import ants
from tqdm import tqdm


def generate(path, save_path):
    img = ants.image_read(path).iMath_normalize()
    inp = torch.Tensor(img.numpy(True).transpose(3, 0, 1, 2))

    h, w, d = list(map(int, args.resolution.split(',')))
    if args.ratio == 1:
        coord = make_coord((h, w, d)).cuda()
        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / h
        cell[:, 1] *= 2 / w
        cell[:, 2] *= 2 / d
        pred = batched_predict(model, ((inp - 0.5) / 0.5).cuda().unsqueeze(0),
                               coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
    else:
        scale = [h / img.shape[0], w / img.shape[1], d / img.shape[2]]
        pred = block_predict(model, ((inp - 0.5) / 0.5).cuda().unsqueeze(0), 30000, args.ratio, scale)

    pred = (pred * 0.5 + 0.5).view(h, w, d).cpu().numpy()
    pred_img = ants.from_numpy(pred, origin=img.origin, spacing=tuple(img.spacing), direction=img.direction)
    if os.path.isdir(save_path):
        save_path = os.path.join(save_path, os.path.basename(path))
    ants.image_write(pred_img, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.png')
    parser.add_argument('--model')
    parser.add_argument('--resolution')
    parser.add_argument('--output', default='')
    parser.add_argument('--ratio', type=int, default=1)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()
    model.eval()

    if os.path.isfile(args.input):
        generate(args.input, args.output)
    else:
        if args.output == '':
            output_path = os.path.join('./demo', args.model.split('/')[-2], args.model.split('/')[-1][:-len('.pth')],
                                       args.resolution.replace(',','-'))
        else:
            output_path=args.output
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        phar = tqdm(os.listdir(args.input))
        for filename in phar:
            generate(os.path.join(args.input, filename),output_path)
