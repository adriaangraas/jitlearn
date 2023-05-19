```bash
mamba create -n jitlearn python=3.10 matplotlib transforms3d \
scikit-learn numpy scipy pyqtgraph tqdm imageio
mamba install -c pytorch -c nvidia pytorch-cuda=11.7 torchvision
```