# Rebuild R2

Use a fresh Linux-64 prefix and keep the channels explicit and strict.
The R1 environment is never modified or reused.

```bash
conda create -y --override-channels --strict-channel-priority \
  -c pytorch -c nvidia -c defaults \
  --prefix "$R2_PREFIX" \
  python=3.10.16 pip pytorch=2.4.1 torchvision=0.19.1 \
  pytorch-cuda=12.4 mkl=2023.1.0 intel-openmp=2023.1.0 \
  numpy=1.26.4 scipy=1.13.1

conda install -y --override-channels --strict-channel-priority \
  -c defaults --prefix "$R2_PREFIX" fsspec
```

Install the frozen Python dependencies with the checked-in constraints:

```bash
CUDA_VISIBLE_DEVICES='' "$R2_PREFIX/bin/python" -m pip install \
  -c environment/pip-constraints.txt \
  pycocotools==2.0.11 faster-coco-eval==1.6.7 \
  pytest==8.4.2 PyYAML==6.0.3 tensorboard==2.21.0 tqdm==4.70.0
```

Run `pip check`, compile vendor Python sources in memory, import the vendor
core, and construct the R18 model with `PResNet.pretrained=false`. Do not use
the training entry point, load checkpoints, access datasets, call forward or
backward, initialize CUDA, or install deployment packages.

The explicit Conda lock and package manifests in this directory are the
portable identity records for this R2 contract.
