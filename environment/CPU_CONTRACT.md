# R2 CPU Contract

This record covers the frozen RT-DETRv2 R18 CPU construction contract.

- Python: 3.10.16
- PyTorch: 2.4.1 with CUDA build 12.4
- torchvision: 0.19.1
- MKL: 2023.1.0
- intel-openmp: 2023.1.0
- NumPy: 1.26.4
- SciPy: 1.13.1
- CUDA visibility: hidden; available false, device count 0, initialized false
- Vendor Python sources compiled in memory: 83
- R18 configuration: `vendor/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml`
- Backbone depth: 18
- Model class: `src.zoo.rtdetr.rtdetr.RTDETR`
- Total parameters: 20184464
- Trainable parameters: 20184464
- Parameter finiteness: pass
- `libtorch_cpu.so` unresolved-symbol check: pass
- Download and checkpoint calls: none
- Forward/backward calls: none
- Dataset/DataLoader construction: none
- Data access, evaluation, training, and speed measurement: none

The model construction used an in-memory `PResNet.pretrained=false` override.
This prevents the upstream configuration's optional pretrained-weight path from
performing a network request while preserving the frozen R18 architecture.
