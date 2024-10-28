# [`timm/resnet18.a3_in1k`](https://huggingface.co/timm/resnet18.a3_in1k)

```
conv1 -> Conv2d
bn1 -> BatchNorm2d
layer1 -> Sequential
|------0 -> BasicBlock
|      |-conv1 -> Conv2d
|      |-bn1 -> BatchNorm2d
|      |-conv2 -> Conv2d
|      |-bn2 -> BatchNorm2d
|------1 -> BasicBlock
|      |-conv1 -> Conv2d
|      |-bn1 -> BatchNorm2d
|      |-conv2 -> Conv2d
|      |-bn2 -> BatchNorm2d
layer2 -> Sequential
|------0 -> BasicBlock
|      |-conv1 -> Conv2d
|      |-bn1 -> BatchNorm2d
|      |-conv2 -> Conv2d
|      |-bn2 -> BatchNorm2d
|      |-downsample -> Sequential
|      | |----------0 -> Conv2d
|      | |----------1 -> BatchNorm2d
|------1 -> BasicBlock
|      |-conv1 -> Conv2d
|      |-bn1 -> BatchNorm2d
|      |-conv2 -> Conv2d
|      |-bn2 -> BatchNorm2d
layer3 -> Sequential
|------0 -> BasicBlock
|      |-conv1 -> Conv2d
|      |-bn1 -> BatchNorm2d
|      |-conv2 -> Conv2d
|      |-bn2 -> BatchNorm2d
|      |-downsample -> Sequential
|      | |----------0 -> Conv2d
|      | |----------1 -> BatchNorm2d
|------1 -> BasicBlock
|      |-conv1 -> Conv2d
|      |-bn1 -> BatchNorm2d
|      |-conv2 -> Conv2d
|      |-bn2 -> BatchNorm2d
layer4 -> Sequential
|------0 -> BasicBlock
|      |-conv1 -> Conv2d
|      |-bn1 -> BatchNorm2d
|      |-conv2 -> Conv2d
|      |-bn2 -> BatchNorm2d
|      |-downsample -> Sequential
|      | |----------0 -> Conv2d
|      | |----------1 -> BatchNorm2d
|------1 -> BasicBlock
|      |-conv1 -> Conv2d
|      |-bn1 -> BatchNorm2d
|      |-conv2 -> Conv2d
|      |-bn2 -> BatchNorm2d
fc -> Linear
```
