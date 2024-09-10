# [`google/mobilenet_v2_1.0_224`](https://huggingface.co/google/mobilenet_v2_1.0_224)

```
mobilenet_v2 -> MobileNetV2Model
|------------conv_stem -> MobileNetV2Stem
|            |---------first_conv -> MobileNetV2ConvLayer
|            |         |----------convolution -> Conv2d
|            |         |----------normalization -> BatchNorm2d
|            |---------conv_3x3 -> MobileNetV2ConvLayer
|            |         |--------convolution -> Conv2d
|            |         |--------normalization -> BatchNorm2d
|            |---------reduce_1x1 -> MobileNetV2ConvLayer
|            |         |----------convolution -> Conv2d
|            |         |----------normalization -> BatchNorm2d
|------------layer -> ModuleList
|            |-----0 -> MobileNetV2InvertedResidual
|            |     |-expand_1x1 -> MobileNetV2ConvLayer
|            |     | |----------convolution -> Conv2d
|            |     | |----------normalization -> BatchNorm2d
|            |     |-conv_3x3 -> MobileNetV2ConvLayer
|            |     | |--------convolution -> Conv2d
|            |     | |--------normalization -> BatchNorm2d
|            |     |-reduce_1x1 -> MobileNetV2ConvLayer
|            |     | |----------convolution -> Conv2d
|            |     | |----------normalization -> BatchNorm2d
|            |-----1 -> MobileNetV2InvertedResidual
|            |     |-expand_1x1 -> MobileNetV2ConvLayer
|            |     | |----------convolution -> Conv2d
|            |     | |----------normalization -> BatchNorm2d
|            |     |-conv_3x3 -> MobileNetV2ConvLayer
|            |     | |--------convolution -> Conv2d
|            |     | |--------normalization -> BatchNorm2d
|            |     |-reduce_1x1 -> MobileNetV2ConvLayer
|            |     | |----------convolution -> Conv2d
|            |     | |----------normalization -> BatchNorm2d
|            |-----2 -> MobileNetV2InvertedResidual
|            |     |-expand_1x1 -> MobileNetV2ConvLayer
|            |     | |----------convolution -> Conv2d
|            |     | |----------normalization -> BatchNorm2d
|            |     |-conv_3x3 -> MobileNetV2ConvLayer
|            |     | |--------convolution -> Conv2d
|            |     | |--------normalization -> BatchNorm2d
|            |     |-reduce_1x1 -> MobileNetV2ConvLayer
|            |     | |----------convolution -> Conv2d
|            |     | |----------normalization -> BatchNorm2d
|            |-----3 -> MobileNetV2InvertedResidual
|            |     |-expand_1x1 -> MobileNetV2ConvLayer
|            |     | |----------convolution -> Conv2d
|            |     | |----------normalization -> BatchNorm2d
|            |     |-conv_3x3 -> MobileNetV2ConvLayer
|            |     | |--------convolution -> Conv2d
|            |     | |--------normalization -> BatchNorm2d
|            |     |-reduce_1x1 -> MobileNetV2ConvLayer
|            |     | |----------convolution -> Conv2d
|            |     | |----------normalization -> BatchNorm2d
|            |-----4 -> MobileNetV2InvertedResidual
|            |     |-expand_1x1 -> MobileNetV2ConvLayer
|            |     | |----------convolution -> Conv2d
|            |     | |----------normalization -> BatchNorm2d
|            |     |-conv_3x3 -> MobileNetV2ConvLayer
|            |     | |--------convolution -> Conv2d
|            |     | |--------normalization -> BatchNorm2d
|            |     |-reduce_1x1 -> MobileNetV2ConvLayer
|            |     | |----------convolution -> Conv2d
|            |     | |----------normalization -> BatchNorm2d
|            |-----5 -> MobileNetV2InvertedResidual
|            |     |-expand_1x1 -> MobileNetV2ConvLayer
|            |     | |----------convolution -> Conv2d
|            |     | |----------normalization -> BatchNorm2d
|            |     |-conv_3x3 -> MobileNetV2ConvLayer
|            |     | |--------convolution -> Conv2d
|            |     | |--------normalization -> BatchNorm2d
|            |     |-reduce_1x1 -> MobileNetV2ConvLayer
|            |     | |----------convolution -> Conv2d
|            |     | |----------normalization -> BatchNorm2d
|            |-----6 -> MobileNetV2InvertedResidual
|            |     |-expand_1x1 -> MobileNetV2ConvLayer
|            |     | |----------convolution -> Conv2d
|            |     | |----------normalization -> BatchNorm2d
|            |     |-conv_3x3 -> MobileNetV2ConvLayer
|            |     | |--------convolution -> Conv2d
|            |     | |--------normalization -> BatchNorm2d
|            |     |-reduce_1x1 -> MobileNetV2ConvLayer
|            |     | |----------convolution -> Conv2d
|            |     | |----------normalization -> BatchNorm2d
|            |-----7 -> MobileNetV2InvertedResidual
|            |     |-expand_1x1 -> MobileNetV2ConvLayer
|            |     | |----------convolution -> Conv2d
|            |     | |----------normalization -> BatchNorm2d
|            |     |-conv_3x3 -> MobileNetV2ConvLayer
|            |     | |--------convolution -> Conv2d
|            |     | |--------normalization -> BatchNorm2d
|            |     |-reduce_1x1 -> MobileNetV2ConvLayer
|            |     | |----------convolution -> Conv2d
|            |     | |----------normalization -> BatchNorm2d
|            |-----8 -> MobileNetV2InvertedResidual
|            |     |-expand_1x1 -> MobileNetV2ConvLayer
|            |     | |----------convolution -> Conv2d
|            |     | |----------normalization -> BatchNorm2d
|            |     |-conv_3x3 -> MobileNetV2ConvLayer
|            |     | |--------convolution -> Conv2d
|            |     | |--------normalization -> BatchNorm2d
|            |     |-reduce_1x1 -> MobileNetV2ConvLayer
|            |     | |----------convolution -> Conv2d
|            |     | |----------normalization -> BatchNorm2d
|            |-----9 -> MobileNetV2InvertedResidual
|            |     |-expand_1x1 -> MobileNetV2ConvLayer
|            |     | |----------convolution -> Conv2d
|            |     | |----------normalization -> BatchNorm2d
|            |     |-conv_3x3 -> MobileNetV2ConvLayer
|            |     | |--------convolution -> Conv2d
|            |     | |--------normalization -> BatchNorm2d
|            |     |-reduce_1x1 -> MobileNetV2ConvLayer
|            |     | |----------convolution -> Conv2d
|            |     | |----------normalization -> BatchNorm2d
|            |-----10 -> MobileNetV2InvertedResidual
|            |     |--expand_1x1 -> MobileNetV2ConvLayer
|            |     |  |----------convolution -> Conv2d
|            |     |  |----------normalization -> BatchNorm2d
|            |     |--conv_3x3 -> MobileNetV2ConvLayer
|            |     |  |--------convolution -> Conv2d
|            |     |  |--------normalization -> BatchNorm2d
|            |     |--reduce_1x1 -> MobileNetV2ConvLayer
|            |     |  |----------convolution -> Conv2d
|            |     |  |----------normalization -> BatchNorm2d
|            |-----11 -> MobileNetV2InvertedResidual
|            |     |--expand_1x1 -> MobileNetV2ConvLayer
|            |     |  |----------convolution -> Conv2d
|            |     |  |----------normalization -> BatchNorm2d
|            |     |--conv_3x3 -> MobileNetV2ConvLayer
|            |     |  |--------convolution -> Conv2d
|            |     |  |--------normalization -> BatchNorm2d
|            |     |--reduce_1x1 -> MobileNetV2ConvLayer
|            |     |  |----------convolution -> Conv2d
|            |     |  |----------normalization -> BatchNorm2d
|            |-----12 -> MobileNetV2InvertedResidual
|            |     |--expand_1x1 -> MobileNetV2ConvLayer
|            |     |  |----------convolution -> Conv2d
|            |     |  |----------normalization -> BatchNorm2d
|            |     |--conv_3x3 -> MobileNetV2ConvLayer
|            |     |  |--------convolution -> Conv2d
|            |     |  |--------normalization -> BatchNorm2d
|            |     |--reduce_1x1 -> MobileNetV2ConvLayer
|            |     |  |----------convolution -> Conv2d
|            |     |  |----------normalization -> BatchNorm2d
|            |-----13 -> MobileNetV2InvertedResidual
|            |     |--expand_1x1 -> MobileNetV2ConvLayer
|            |     |  |----------convolution -> Conv2d
|            |     |  |----------normalization -> BatchNorm2d
|            |     |--conv_3x3 -> MobileNetV2ConvLayer
|            |     |  |--------convolution -> Conv2d
|            |     |  |--------normalization -> BatchNorm2d
|            |     |--reduce_1x1 -> MobileNetV2ConvLayer
|            |     |  |----------convolution -> Conv2d
|            |     |  |----------normalization -> BatchNorm2d
|            |-----14 -> MobileNetV2InvertedResidual
|            |     |--expand_1x1 -> MobileNetV2ConvLayer
|            |     |  |----------convolution -> Conv2d
|            |     |  |----------normalization -> BatchNorm2d
|            |     |--conv_3x3 -> MobileNetV2ConvLayer
|            |     |  |--------convolution -> Conv2d
|            |     |  |--------normalization -> BatchNorm2d
|            |     |--reduce_1x1 -> MobileNetV2ConvLayer
|            |     |  |----------convolution -> Conv2d
|            |     |  |----------normalization -> BatchNorm2d
|            |-----15 -> MobileNetV2InvertedResidual
|            |     |--expand_1x1 -> MobileNetV2ConvLayer
|            |     |  |----------convolution -> Conv2d
|            |     |  |----------normalization -> BatchNorm2d
|            |     |--conv_3x3 -> MobileNetV2ConvLayer
|            |     |  |--------convolution -> Conv2d
|            |     |  |--------normalization -> BatchNorm2d
|            |     |--reduce_1x1 -> MobileNetV2ConvLayer
|            |     |  |----------convolution -> Conv2d
|            |     |  |----------normalization -> BatchNorm2d
|------------conv_1x1 -> MobileNetV2ConvLayer
|            |--------convolution -> Conv2d
|            |--------normalization -> BatchNorm2d
classifier -> Linear
```
