# [`timm/mobilenetv3_small_050.lamb_in1k`](https://huggingface.co/timm/mobilenetv3_small_050.lamb_in1k)

```
conv_stem -> Conv2d
bn1 -> BatchNormAct2d
blocks -> Sequential
|------0 -> Sequential
|      |-0 -> DepthwiseSeparableConv
|      | |-conv_dw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|------1 -> Sequential
|      |-0 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-1 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|------2 -> Sequential
|      |-0 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-1 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-2 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|------3 -> Sequential
|      |-0 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-1 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|------4 -> Sequential
|      |-0 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-1 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-2 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|------5 -> Sequential
|      |-0 -> ConvBnAct
|      | |-conv -> Conv2d
|      | |-bn1 -> BatchNormAct2d
conv_head -> Conv2d
classifier -> Linear
```
