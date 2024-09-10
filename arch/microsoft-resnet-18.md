# [`microsoft/resnet-18`](https://huggingface.co/microsoft/resnet-18)

```
resnet -> ResNetModel
|------embedder -> ResNetEmbeddings
|      |--------embedder -> ResNetConvLayer
|      |        |--------convolution -> Conv2d
|      |        |--------normalization -> BatchNorm2d
|------encoder -> ResNetEncoder
|      |-------stages -> ModuleList
|      |       |------0 -> ResNetStage
|      |       |      |-layers -> Sequential
|      |       |      | |------0 -> ResNetBasicLayer
|      |       |      | |      |-layer -> Sequential
|      |       |      | |      | |-----0 -> ResNetConvLayer
|      |       |      | |      | |     |-convolution -> Conv2d
|      |       |      | |      | |     |-normalization -> BatchNorm2d
|      |       |      | |      | |-----1 -> ResNetConvLayer
|      |       |      | |      | |     |-convolution -> Conv2d
|      |       |      | |      | |     |-normalization -> BatchNorm2d
|      |       |      | |------1 -> ResNetBasicLayer
|      |       |      | |      |-layer -> Sequential
|      |       |      | |      | |-----0 -> ResNetConvLayer
|      |       |      | |      | |     |-convolution -> Conv2d
|      |       |      | |      | |     |-normalization -> BatchNorm2d
|      |       |      | |      | |-----1 -> ResNetConvLayer
|      |       |      | |      | |     |-convolution -> Conv2d
|      |       |      | |      | |     |-normalization -> BatchNorm2d
|      |       |------1 -> ResNetStage
|      |       |      |-layers -> Sequential
|      |       |      | |------0 -> ResNetBasicLayer
|      |       |      | |      |-shortcut -> ResNetShortCut
|      |       |      | |      | |--------convolution -> Conv2d
|      |       |      | |      | |--------normalization -> BatchNorm2d
|      |       |      | |      |-layer -> Sequential
|      |       |      | |      | |-----0 -> ResNetConvLayer
|      |       |      | |      | |     |-convolution -> Conv2d
|      |       |      | |      | |     |-normalization -> BatchNorm2d
|      |       |      | |      | |-----1 -> ResNetConvLayer
|      |       |      | |      | |     |-convolution -> Conv2d
|      |       |      | |      | |     |-normalization -> BatchNorm2d
|      |       |      | |------1 -> ResNetBasicLayer
|      |       |      | |      |-layer -> Sequential
|      |       |      | |      | |-----0 -> ResNetConvLayer
|      |       |      | |      | |     |-convolution -> Conv2d
|      |       |      | |      | |     |-normalization -> BatchNorm2d
|      |       |      | |      | |-----1 -> ResNetConvLayer
|      |       |      | |      | |     |-convolution -> Conv2d
|      |       |      | |      | |     |-normalization -> BatchNorm2d
|      |       |------2 -> ResNetStage
|      |       |      |-layers -> Sequential
|      |       |      | |------0 -> ResNetBasicLayer
|      |       |      | |      |-shortcut -> ResNetShortCut
|      |       |      | |      | |--------convolution -> Conv2d
|      |       |      | |      | |--------normalization -> BatchNorm2d
|      |       |      | |      |-layer -> Sequential
|      |       |      | |      | |-----0 -> ResNetConvLayer
|      |       |      | |      | |     |-convolution -> Conv2d
|      |       |      | |      | |     |-normalization -> BatchNorm2d
|      |       |      | |      | |-----1 -> ResNetConvLayer
|      |       |      | |      | |     |-convolution -> Conv2d
|      |       |      | |      | |     |-normalization -> BatchNorm2d
|      |       |      | |------1 -> ResNetBasicLayer
|      |       |      | |      |-layer -> Sequential
|      |       |      | |      | |-----0 -> ResNetConvLayer
|      |       |      | |      | |     |-convolution -> Conv2d
|      |       |      | |      | |     |-normalization -> BatchNorm2d
|      |       |      | |      | |-----1 -> ResNetConvLayer
|      |       |      | |      | |     |-convolution -> Conv2d
|      |       |      | |      | |     |-normalization -> BatchNorm2d
|      |       |------3 -> ResNetStage
|      |       |      |-layers -> Sequential
|      |       |      | |------0 -> ResNetBasicLayer
|      |       |      | |      |-shortcut -> ResNetShortCut
|      |       |      | |      | |--------convolution -> Conv2d
|      |       |      | |      | |--------normalization -> BatchNorm2d
|      |       |      | |      |-layer -> Sequential
|      |       |      | |      | |-----0 -> ResNetConvLayer
|      |       |      | |      | |     |-convolution -> Conv2d
|      |       |      | |      | |     |-normalization -> BatchNorm2d
|      |       |      | |      | |-----1 -> ResNetConvLayer
|      |       |      | |      | |     |-convolution -> Conv2d
|      |       |      | |      | |     |-normalization -> BatchNorm2d
|      |       |      | |------1 -> ResNetBasicLayer
|      |       |      | |      |-layer -> Sequential
|      |       |      | |      | |-----0 -> ResNetConvLayer
|      |       |      | |      | |     |-convolution -> Conv2d
|      |       |      | |      | |     |-normalization -> BatchNorm2d
|      |       |      | |      | |-----1 -> ResNetConvLayer
|      |       |      | |      | |     |-convolution -> Conv2d
|      |       |      | |      | |     |-normalization -> BatchNorm2d
classifier -> Sequential
|----------1 -> Linear
```
