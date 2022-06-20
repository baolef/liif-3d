model="./save/_train_rdnA-liif-mixed/epoch-last.pth"
gpu=0
path="logs/test_$(date +"%Y-%m-%d_%T").log"
nohup sh scripts/test-div2k.sh $model $gpu > $path 2>&1 &