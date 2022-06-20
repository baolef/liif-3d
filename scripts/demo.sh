path="logs/demo_$(date +"%Y-%m-%d_%T").log"
python demo.py \
--input /data/baole/liif/oasis/all \
--model ./save/_train_rdnB-liif-mixed/epoch-last.pth \
--resolution 512,512,512 \
--ratio 4 \
--gpu 0 \
> $path 2>&1 &