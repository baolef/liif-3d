#files=$(find logs -maxdepth 1 -type f -printf '.')
#cnt=${#files}
#path="logs/train_$((cnt+1)).log"
#echo $path
path="logs/train_$(date +"%Y-%m-%d_%T").log"
python train_liif.py --config configs/train-div2k/train_rdn-metasr-u.yaml --gpu 1 > $path 2>&1 &