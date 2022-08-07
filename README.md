# LIIF-3d

The code is based on [Yinbo Chen](https://yinboc.github.io/)'s [liif](https://github.com/yinboc/liif).

This is 3D implementation of the LIIF model.

### Training
To train the model, you can run `train.sh` in the `scripts` folder
```bash
sh scripts/train.sh
```

To visualize the training results (eg. loss, training output), you can run `tensorboard.sh` 
```bash
sh tensorboard.sh
```

If `tensorboard.sh` is run on a server, you can run `local.sh` on your local machine to access the visualization
```bash
sh local.sh
```

### Testing
To test the model, you need to run `test-div2k.sh` in the `scripts` folder
```bash
sh scripts/test-div2k.sh
```

To summarize the testing results, you need to run `summary.py`
```bash
python summary.py
```

To run the baseline (nearest neighbour), you need to run `baseline.py`
```bash
python baseline.py
```

### Demo
To directly calculate the output without computing metrics, you need to run `demo.sh` in the `scripts` folder
```bash
sh scripts/demo.sh
```
