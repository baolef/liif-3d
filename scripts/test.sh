model="./save/_train_rdn-liif/epoch-best.pth"
gpu=0
path="logs/test_$(date +"%Y-%m-%d_%T").log"
nohup sh scripts/test-div2k.sh $model $gpu > $path 2>&1 &