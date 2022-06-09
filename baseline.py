import ants
import os
import torch
import utils


if __name__ == '__main__':
    root='/data/baole/liif/oasis/valid_HR'
    save_root='results/baseline/'
    scales=[2,3,4,6,12,18,24,30]
    for scale in scales:
        save_path=os.path.join(save_root,'test-div2k-'+str(scale))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        val_res = utils.Averager()
        for file in os.listdir(root):
            img=ants.image_read(os.path.join(root, file))
            shape_hr = list(img.shape)
            shape_lr = list(img.shape)
            for j in range(len(shape_lr)):
                shape_lr[j] = int(shape_lr[j] / scale)
            img_lr=ants.resample_image(img, shape_lr, use_voxels=True)
            img_hr=ants.resample_image(img_lr, shape_hr, use_voxels=True)
            tensor_gt=torch.Tensor(img.numpy())
            tensor_hr = torch.Tensor(img_hr.numpy())

            ants.image_write(img_hr,os.path.join(save_path,file))

            res = utils.calc_psnr(tensor_gt,tensor_hr,rgb_range=255)
            val_res.add(res.item())

        print(save_path)
        with open(os.path.join(save_path, 'result.txt'), 'w') as f:
            f.write('psnr: {:.4f}'.format(val_res.item()))

