model="./save/_train_rdnA-metasr-full/epoch-last.pth"
gpu=3
path="logs/test_$(date +"%Y-%m-%d_%T").log"
nohup sh scripts/test-div2k.sh $model $gpu > $path 2>&1 &