import argparse
import os
import math
from functools import partial

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils

import ants


def batched_predict(model, inp, coord, cell, bsize, half=False):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :], half)
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred


def block_predict(model, inp, coord, cell, bsize, ratio, scale, gt_div, gt_sub, batch, half=False, device='cuda'):
    with torch.no_grad():
        ih, iw, id = inp.shape[-3:]
        s = round(math.pow(coord.shape[1] / (ih * iw * id), 1 / 3))
        shape = [inp.shape[0], 1, round(ih * s), round(iw * s), round(ih * s)]
        output = torch.zeros(shape).to(device)
        cnt = torch.zeros(shape).to(device)
        blk_half_shape = (int(inp.shape[2] / ratio / 2), int(inp.shape[3] / ratio / 2), int(inp.shape[4] / ratio / 2))

        for x in range(0, inp.shape[2] - blk_half_shape[0], blk_half_shape[0]):
            for y in range(0, inp.shape[3] - blk_half_shape[1], blk_half_shape[1]):
                for z in range(0, inp.shape[4] - blk_half_shape[2], blk_half_shape[2]):
                    blk = inp[:, :, x:min(x + blk_half_shape[0] * 2, inp.shape[2]),
                          y:min(y + blk_half_shape[1] * 2, inp.shape[3]),
                          z:min(z + blk_half_shape[2] * 2, inp.shape[4])]

                    # model.gen_feat(blk)

                    # prepare blk_coords and blk_cell
                    blk_coord = utils.make_coord((blk.shape[2] * scale, blk.shape[3] * scale, blk.shape[3] * scale))
                    blk_coord = blk_coord.repeat(len(blk), 1, 1).to(device)
                    blk_cell = torch.ones_like(blk_coord)
                    blk_cell[:, :, 0] *= 2 / (blk.shape[2] * scale)
                    blk_cell[:, :, 1] *= 2 / (blk.shape[3] * scale)
                    blk_cell[:, :, 2] *= 2 / (blk.shape[4] * scale)
                    blk_cell = blk_cell.repeat(len(blk), 1, 1).to(device)

                    # n = blk_coord.shape[1]
                    # ql = 0
                    # preds = []
                    # while ql < n:
                    #     qr = min(ql + bsize, n)
                    #     pred = model.query_rgb(blk_coord[:, ql: qr, :], blk_cell[:, ql: qr, :], half)
                    #     preds.append(pred)
                    #     ql = qr
                    # pred = torch.cat(preds, dim=1)

                    pred=batched_predict(model,blk,blk_coord,blk_cell,bsize,half)

                    # convert pred to 3d
                    ih, iw, id = blk.shape[-3:]
                    shape = [inp.shape[0], round(ih * scale), round(iw * scale), round(id * scale), 1]
                    pred = pred.view(*shape).permute(0, 4, 1, 2, 3).contiguous()

                    # putting pred to output
                    output[:, :, scale * x:scale * min(x + blk_half_shape[0] * 2, inp.shape[2]),
                    scale * y:scale * min(y + blk_half_shape[1] * 2, inp.shape[3]),
                    scale * z:scale * min(z + blk_half_shape[2] * 2, inp.shape[4])] += pred
                    cnt[:, :, scale * x:scale * min(x + blk_half_shape[0] * 2, inp.shape[2]),
                    scale * y:scale * min(y + blk_half_shape[1] * 2, inp.shape[3]),
                    scale * z:scale * min(z + blk_half_shape[2] * 2, inp.shape[4])] += 1

    return output / cnt


def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False, visualize=False, device='cuda', half=False, ratio=1):
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1, 1).to(device)
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1, 1).to(device)
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).to(device)
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).to(device)

    if half:
        inp_sub = inp_sub.half()
        inp_div = inp_div.half()
        gt_sub = gt_sub.half()
        gt_div = gt_div.half()

    if eval_type is None:
        metric_fn = utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    val_res = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            if k != 'path':
                if half:
                    batch[k] = v.half().to(device)
                else:
                    batch[k] = v.to(device)

        inp = (batch['inp'] - inp_sub) / inp_div
        if ratio == 1:
            if eval_bsize is None:
                with torch.no_grad():
                    pred = model(inp, batch['coord'], batch['cell'])
            else:
                pred = batched_predict(model, inp,
                                       batch['coord'], batch['cell'], eval_bsize, half)
        else:
            scale = int(eval_type.split('-')[1])
            pred = block_predict(model, inp,
                                 batch['coord'], batch['cell'], eval_bsize, ratio, scale, gt_div, gt_sub, batch, half)

        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)

        if eval_type is not None:  # reshape for shaving-eval
            ih, iw, id = batch['inp'].shape[-3:]
            s = round(math.pow(batch['coord'].shape[1] / (ih * iw * id), 1 / 3))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), round(ih * s), 1]
            pred = pred.view(*shape) \
                .permute(0, 4, 1, 2, 3).contiguous()
            batch['gt'] = batch['gt'].view(*shape) \
                .permute(0, 4, 1, 2, 3).contiguous()

        res = metric_fn(pred, batch['gt'])
        val_res.add(res.item(), inp.shape[0])

        if visualize:
            to_nii(pred, batch['path'], scale)

        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))

    return val_res.item()


def to_nii(inps, paths, scale):
    inps = inps.to(device='cpu', dtype=torch.float).numpy()
    for i in range(len(paths)):
        inp = inps[i, 0]
        path = paths[i]
        name = os.path.split(path)[-1]
        img_original = ants.image_read(path)
        spacing = list(img_original.spacing)
        for j in range(inp.ndim):
            spacing[j] /= scale
        img = ants.from_numpy(inp, origin=img_original.origin, spacing=spacing, direction=img_original.direction)
        ants.image_write(img, os.path.join(save_path, name))


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--ratio', type=int, default=1)
    parser.add_argument('--half', action='store_true')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cpu' if (args.gpu == '-1' or args.gpu == 'cpu') else 'cuda'

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'], pin_memory=True)

    model_spec = torch.load(args.model, map_location=torch.device(device))['model']
    model_spec['args']['device'] = device
    model = models.make(model_spec, load_sd=True).to(device)

    if args.half:
        model = model.half()

    save_path = os.path.join('./results', args.model.split('/')[-2], args.model.split('/')[-1][:-len('.pth')],
                             args.config.split('/')[-1][:-len('.yaml')])
    if args.ratio>1:
        save_path += '-ratio-' + str(args.ratio)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(save_path)

    res = eval_psnr(loader, model,
                    data_norm=config.get('data_norm'),
                    eval_type=config.get('eval_type'),
                    eval_bsize=config.get('eval_bsize'),
                    verbose=True,
                    visualize=True,
                    device=device,
                    half=args.half,
                    ratio=args.ratio)
    print('result: {:.4f}'.format(res))
    with open(os.path.join(save_path, 'result.txt'), 'w') as f:
        f.write('psnr: {:.4f}'.format(res))
