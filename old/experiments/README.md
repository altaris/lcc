# How to launch an experiment

ezpz

```sh
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python3.10 experiments/script.py
```

`CUDA_VISIBLE_DEVICES` must be between `0` and `NB_OF_GPUS - 1`. To use
multiple GPUs, simply separate the indices by commas:
`CUDA_VISIBLE_DEVICES=0,1,2`. To hide all GPUs, use `CUDA_VISIBLE_DEVICES=`.

Experiments that involve clustering work better when only one GPU is available
for some reason...

I had problems with OpenBLAS in the past and setting `OMP_NUM_THREADS=1`
helped. IDK if it's always necessary.