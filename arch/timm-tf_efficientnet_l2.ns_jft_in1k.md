# [`timm/tf_efficientnet_l2.ns_jft_in1k`](https://huggingface.co/timm/tf_efficientnet_l2.ns_jft_in1k)

```
conv_stem -> Conv2dSame
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
|      |-1 -> DepthwiseSeparableConv
|      | |-conv_dw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      |-2 -> DepthwiseSeparableConv
|      | |-conv_dw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      |-3 -> DepthwiseSeparableConv
|      | |-conv_dw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      |-4 -> DepthwiseSeparableConv
|      | |-conv_dw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      |-5 -> DepthwiseSeparableConv
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
|      | |-conv_dw -> Conv2dSame
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
|      |-3 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-4 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-5 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-6 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-7 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-8 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-9 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-10 -> InvertedResidual
|      | |--conv_pw -> Conv2d
|      | |--bn1 -> BatchNormAct2d
|      | |--conv_dw -> Conv2d
|      | |--bn2 -> BatchNormAct2d
|      | |--se -> SqueezeExcite
|      | |  |--conv_reduce -> Conv2d
|      | |  |--conv_expand -> Conv2d
|      | |--conv_pwl -> Conv2d
|      | |--bn3 -> BatchNormAct2d
|------2 -> Sequential
|      |-0 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2dSame
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
|      |-3 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-4 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-5 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-6 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-7 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-8 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-9 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-10 -> InvertedResidual
|      | |--conv_pw -> Conv2d
|      | |--bn1 -> BatchNormAct2d
|      | |--conv_dw -> Conv2d
|      | |--bn2 -> BatchNormAct2d
|      | |--se -> SqueezeExcite
|      | |  |--conv_reduce -> Conv2d
|      | |  |--conv_expand -> Conv2d
|      | |--conv_pwl -> Conv2d
|      | |--bn3 -> BatchNormAct2d
|------3 -> Sequential
|      |-0 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2dSame
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
|      |-3 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-4 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-5 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-6 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-7 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-8 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-9 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-10 -> InvertedResidual
|      | |--conv_pw -> Conv2d
|      | |--bn1 -> BatchNormAct2d
|      | |--conv_dw -> Conv2d
|      | |--bn2 -> BatchNormAct2d
|      | |--se -> SqueezeExcite
|      | |  |--conv_reduce -> Conv2d
|      | |  |--conv_expand -> Conv2d
|      | |--conv_pwl -> Conv2d
|      | |--bn3 -> BatchNormAct2d
|      |-11 -> InvertedResidual
|      | |--conv_pw -> Conv2d
|      | |--bn1 -> BatchNormAct2d
|      | |--conv_dw -> Conv2d
|      | |--bn2 -> BatchNormAct2d
|      | |--se -> SqueezeExcite
|      | |  |--conv_reduce -> Conv2d
|      | |  |--conv_expand -> Conv2d
|      | |--conv_pwl -> Conv2d
|      | |--bn3 -> BatchNormAct2d
|      |-12 -> InvertedResidual
|      | |--conv_pw -> Conv2d
|      | |--bn1 -> BatchNormAct2d
|      | |--conv_dw -> Conv2d
|      | |--bn2 -> BatchNormAct2d
|      | |--se -> SqueezeExcite
|      | |  |--conv_reduce -> Conv2d
|      | |  |--conv_expand -> Conv2d
|      | |--conv_pwl -> Conv2d
|      | |--bn3 -> BatchNormAct2d
|      |-13 -> InvertedResidual
|      | |--conv_pw -> Conv2d
|      | |--bn1 -> BatchNormAct2d
|      | |--conv_dw -> Conv2d
|      | |--bn2 -> BatchNormAct2d
|      | |--se -> SqueezeExcite
|      | |  |--conv_reduce -> Conv2d
|      | |  |--conv_expand -> Conv2d
|      | |--conv_pwl -> Conv2d
|      | |--bn3 -> BatchNormAct2d
|      |-14 -> InvertedResidual
|      | |--conv_pw -> Conv2d
|      | |--bn1 -> BatchNormAct2d
|      | |--conv_dw -> Conv2d
|      | |--bn2 -> BatchNormAct2d
|      | |--se -> SqueezeExcite
|      | |  |--conv_reduce -> Conv2d
|      | |  |--conv_expand -> Conv2d
|      | |--conv_pwl -> Conv2d
|      | |--bn3 -> BatchNormAct2d
|      |-15 -> InvertedResidual
|      | |--conv_pw -> Conv2d
|      | |--bn1 -> BatchNormAct2d
|      | |--conv_dw -> Conv2d
|      | |--bn2 -> BatchNormAct2d
|      | |--se -> SqueezeExcite
|      | |  |--conv_reduce -> Conv2d
|      | |  |--conv_expand -> Conv2d
|      | |--conv_pwl -> Conv2d
|      | |--bn3 -> BatchNormAct2d
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
|      |-3 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-4 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-5 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-6 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-7 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-8 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-9 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-10 -> InvertedResidual
|      | |--conv_pw -> Conv2d
|      | |--bn1 -> BatchNormAct2d
|      | |--conv_dw -> Conv2d
|      | |--bn2 -> BatchNormAct2d
|      | |--se -> SqueezeExcite
|      | |  |--conv_reduce -> Conv2d
|      | |  |--conv_expand -> Conv2d
|      | |--conv_pwl -> Conv2d
|      | |--bn3 -> BatchNormAct2d
|      |-11 -> InvertedResidual
|      | |--conv_pw -> Conv2d
|      | |--bn1 -> BatchNormAct2d
|      | |--conv_dw -> Conv2d
|      | |--bn2 -> BatchNormAct2d
|      | |--se -> SqueezeExcite
|      | |  |--conv_reduce -> Conv2d
|      | |  |--conv_expand -> Conv2d
|      | |--conv_pwl -> Conv2d
|      | |--bn3 -> BatchNormAct2d
|      |-12 -> InvertedResidual
|      | |--conv_pw -> Conv2d
|      | |--bn1 -> BatchNormAct2d
|      | |--conv_dw -> Conv2d
|      | |--bn2 -> BatchNormAct2d
|      | |--se -> SqueezeExcite
|      | |  |--conv_reduce -> Conv2d
|      | |  |--conv_expand -> Conv2d
|      | |--conv_pwl -> Conv2d
|      | |--bn3 -> BatchNormAct2d
|      |-13 -> InvertedResidual
|      | |--conv_pw -> Conv2d
|      | |--bn1 -> BatchNormAct2d
|      | |--conv_dw -> Conv2d
|      | |--bn2 -> BatchNormAct2d
|      | |--se -> SqueezeExcite
|      | |  |--conv_reduce -> Conv2d
|      | |  |--conv_expand -> Conv2d
|      | |--conv_pwl -> Conv2d
|      | |--bn3 -> BatchNormAct2d
|      |-14 -> InvertedResidual
|      | |--conv_pw -> Conv2d
|      | |--bn1 -> BatchNormAct2d
|      | |--conv_dw -> Conv2d
|      | |--bn2 -> BatchNormAct2d
|      | |--se -> SqueezeExcite
|      | |  |--conv_reduce -> Conv2d
|      | |  |--conv_expand -> Conv2d
|      | |--conv_pwl -> Conv2d
|      | |--bn3 -> BatchNormAct2d
|      |-15 -> InvertedResidual
|      | |--conv_pw -> Conv2d
|      | |--bn1 -> BatchNormAct2d
|      | |--conv_dw -> Conv2d
|      | |--bn2 -> BatchNormAct2d
|      | |--se -> SqueezeExcite
|      | |  |--conv_reduce -> Conv2d
|      | |  |--conv_expand -> Conv2d
|      | |--conv_pwl -> Conv2d
|      | |--bn3 -> BatchNormAct2d
|------5 -> Sequential
|      |-0 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2dSame
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
|      |-3 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-4 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-5 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-6 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-7 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-8 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-9 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-10 -> InvertedResidual
|      | |--conv_pw -> Conv2d
|      | |--bn1 -> BatchNormAct2d
|      | |--conv_dw -> Conv2d
|      | |--bn2 -> BatchNormAct2d
|      | |--se -> SqueezeExcite
|      | |  |--conv_reduce -> Conv2d
|      | |  |--conv_expand -> Conv2d
|      | |--conv_pwl -> Conv2d
|      | |--bn3 -> BatchNormAct2d
|      |-11 -> InvertedResidual
|      | |--conv_pw -> Conv2d
|      | |--bn1 -> BatchNormAct2d
|      | |--conv_dw -> Conv2d
|      | |--bn2 -> BatchNormAct2d
|      | |--se -> SqueezeExcite
|      | |  |--conv_reduce -> Conv2d
|      | |  |--conv_expand -> Conv2d
|      | |--conv_pwl -> Conv2d
|      | |--bn3 -> BatchNormAct2d
|      |-12 -> InvertedResidual
|      | |--conv_pw -> Conv2d
|      | |--bn1 -> BatchNormAct2d
|      | |--conv_dw -> Conv2d
|      | |--bn2 -> BatchNormAct2d
|      | |--se -> SqueezeExcite
|      | |  |--conv_reduce -> Conv2d
|      | |  |--conv_expand -> Conv2d
|      | |--conv_pwl -> Conv2d
|      | |--bn3 -> BatchNormAct2d
|      |-13 -> InvertedResidual
|      | |--conv_pw -> Conv2d
|      | |--bn1 -> BatchNormAct2d
|      | |--conv_dw -> Conv2d
|      | |--bn2 -> BatchNormAct2d
|      | |--se -> SqueezeExcite
|      | |  |--conv_reduce -> Conv2d
|      | |  |--conv_expand -> Conv2d
|      | |--conv_pwl -> Conv2d
|      | |--bn3 -> BatchNormAct2d
|      |-14 -> InvertedResidual
|      | |--conv_pw -> Conv2d
|      | |--bn1 -> BatchNormAct2d
|      | |--conv_dw -> Conv2d
|      | |--bn2 -> BatchNormAct2d
|      | |--se -> SqueezeExcite
|      | |  |--conv_reduce -> Conv2d
|      | |  |--conv_expand -> Conv2d
|      | |--conv_pwl -> Conv2d
|      | |--bn3 -> BatchNormAct2d
|      |-15 -> InvertedResidual
|      | |--conv_pw -> Conv2d
|      | |--bn1 -> BatchNormAct2d
|      | |--conv_dw -> Conv2d
|      | |--bn2 -> BatchNormAct2d
|      | |--se -> SqueezeExcite
|      | |  |--conv_reduce -> Conv2d
|      | |  |--conv_expand -> Conv2d
|      | |--conv_pwl -> Conv2d
|      | |--bn3 -> BatchNormAct2d
|      |-16 -> InvertedResidual
|      | |--conv_pw -> Conv2d
|      | |--bn1 -> BatchNormAct2d
|      | |--conv_dw -> Conv2d
|      | |--bn2 -> BatchNormAct2d
|      | |--se -> SqueezeExcite
|      | |  |--conv_reduce -> Conv2d
|      | |  |--conv_expand -> Conv2d
|      | |--conv_pwl -> Conv2d
|      | |--bn3 -> BatchNormAct2d
|      |-17 -> InvertedResidual
|      | |--conv_pw -> Conv2d
|      | |--bn1 -> BatchNormAct2d
|      | |--conv_dw -> Conv2d
|      | |--bn2 -> BatchNormAct2d
|      | |--se -> SqueezeExcite
|      | |  |--conv_reduce -> Conv2d
|      | |  |--conv_expand -> Conv2d
|      | |--conv_pwl -> Conv2d
|      | |--bn3 -> BatchNormAct2d
|      |-18 -> InvertedResidual
|      | |--conv_pw -> Conv2d
|      | |--bn1 -> BatchNormAct2d
|      | |--conv_dw -> Conv2d
|      | |--bn2 -> BatchNormAct2d
|      | |--se -> SqueezeExcite
|      | |  |--conv_reduce -> Conv2d
|      | |  |--conv_expand -> Conv2d
|      | |--conv_pwl -> Conv2d
|      | |--bn3 -> BatchNormAct2d
|      |-19 -> InvertedResidual
|      | |--conv_pw -> Conv2d
|      | |--bn1 -> BatchNormAct2d
|      | |--conv_dw -> Conv2d
|      | |--bn2 -> BatchNormAct2d
|      | |--se -> SqueezeExcite
|      | |  |--conv_reduce -> Conv2d
|      | |  |--conv_expand -> Conv2d
|      | |--conv_pwl -> Conv2d
|      | |--bn3 -> BatchNormAct2d
|      |-20 -> InvertedResidual
|      | |--conv_pw -> Conv2d
|      | |--bn1 -> BatchNormAct2d
|      | |--conv_dw -> Conv2d
|      | |--bn2 -> BatchNormAct2d
|      | |--se -> SqueezeExcite
|      | |  |--conv_reduce -> Conv2d
|      | |  |--conv_expand -> Conv2d
|      | |--conv_pwl -> Conv2d
|      | |--bn3 -> BatchNormAct2d
|      |-21 -> InvertedResidual
|      | |--conv_pw -> Conv2d
|      | |--bn1 -> BatchNormAct2d
|      | |--conv_dw -> Conv2d
|      | |--bn2 -> BatchNormAct2d
|      | |--se -> SqueezeExcite
|      | |  |--conv_reduce -> Conv2d
|      | |  |--conv_expand -> Conv2d
|      | |--conv_pwl -> Conv2d
|      | |--bn3 -> BatchNormAct2d
|------6 -> Sequential
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
|      |-3 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-4 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
|      |-5 -> InvertedResidual
|      | |-conv_pw -> Conv2d
|      | |-bn1 -> BatchNormAct2d
|      | |-conv_dw -> Conv2d
|      | |-bn2 -> BatchNormAct2d
|      | |-se -> SqueezeExcite
|      | | |--conv_reduce -> Conv2d
|      | | |--conv_expand -> Conv2d
|      | |-conv_pwl -> Conv2d
|      | |-bn3 -> BatchNormAct2d
conv_head -> Conv2d
bn2 -> BatchNormAct2d
classifier -> Linear
```
