# [`timm/convnext_small.in12k`](https://huggingface.co/timm/convnext_small.in12k)

```
stem -> Sequential
|----0 -> Conv2d
|----1 -> LayerNorm2d
stages -> Sequential
|------0 -> ConvNeXtStage
|      |-blocks -> Sequential
|      | |------0 -> ConvNeXtBlock
|      | |      |-conv_dw -> Conv2d
|      | |      |-norm -> LayerNorm
|      | |      |-mlp -> Mlp
|      | |      | |---fc1 -> Linear
|      | |      | |---fc2 -> Linear
|      | |------1 -> ConvNeXtBlock
|      | |      |-conv_dw -> Conv2d
|      | |      |-norm -> LayerNorm
|      | |      |-mlp -> Mlp
|      | |      | |---fc1 -> Linear
|      | |      | |---fc2 -> Linear
|      | |------2 -> ConvNeXtBlock
|      | |      |-conv_dw -> Conv2d
|      | |      |-norm -> LayerNorm
|      | |      |-mlp -> Mlp
|      | |      | |---fc1 -> Linear
|      | |      | |---fc2 -> Linear
|------1 -> ConvNeXtStage
|      |-downsample -> Sequential
|      | |----------0 -> LayerNorm2d
|      | |----------1 -> Conv2d
|      |-blocks -> Sequential
|      | |------0 -> ConvNeXtBlock
|      | |      |-conv_dw -> Conv2d
|      | |      |-norm -> LayerNorm
|      | |      |-mlp -> Mlp
|      | |      | |---fc1 -> Linear
|      | |      | |---fc2 -> Linear
|      | |------1 -> ConvNeXtBlock
|      | |      |-conv_dw -> Conv2d
|      | |      |-norm -> LayerNorm
|      | |      |-mlp -> Mlp
|      | |      | |---fc1 -> Linear
|      | |      | |---fc2 -> Linear
|      | |------2 -> ConvNeXtBlock
|      | |      |-conv_dw -> Conv2d
|      | |      |-norm -> LayerNorm
|      | |      |-mlp -> Mlp
|      | |      | |---fc1 -> Linear
|      | |      | |---fc2 -> Linear
|------2 -> ConvNeXtStage
|      |-downsample -> Sequential
|      | |----------0 -> LayerNorm2d
|      | |----------1 -> Conv2d
|      |-blocks -> Sequential
|      | |------0 -> ConvNeXtBlock
|      | |      |-conv_dw -> Conv2d
|      | |      |-norm -> LayerNorm
|      | |      |-mlp -> Mlp
|      | |      | |---fc1 -> Linear
|      | |      | |---fc2 -> Linear
|      | |------1 -> ConvNeXtBlock
|      | |      |-conv_dw -> Conv2d
|      | |      |-norm -> LayerNorm
|      | |      |-mlp -> Mlp
|      | |      | |---fc1 -> Linear
|      | |      | |---fc2 -> Linear
|      | |------2 -> ConvNeXtBlock
|      | |      |-conv_dw -> Conv2d
|      | |      |-norm -> LayerNorm
|      | |      |-mlp -> Mlp
|      | |      | |---fc1 -> Linear
|      | |      | |---fc2 -> Linear
|      | |------3 -> ConvNeXtBlock
|      | |      |-conv_dw -> Conv2d
|      | |      |-norm -> LayerNorm
|      | |      |-mlp -> Mlp
|      | |      | |---fc1 -> Linear
|      | |      | |---fc2 -> Linear
|      | |------4 -> ConvNeXtBlock
|      | |      |-conv_dw -> Conv2d
|      | |      |-norm -> LayerNorm
|      | |      |-mlp -> Mlp
|      | |      | |---fc1 -> Linear
|      | |      | |---fc2 -> Linear
|      | |------5 -> ConvNeXtBlock
|      | |      |-conv_dw -> Conv2d
|      | |      |-norm -> LayerNorm
|      | |      |-mlp -> Mlp
|      | |      | |---fc1 -> Linear
|      | |      | |---fc2 -> Linear
|      | |------6 -> ConvNeXtBlock
|      | |      |-conv_dw -> Conv2d
|      | |      |-norm -> LayerNorm
|      | |      |-mlp -> Mlp
|      | |      | |---fc1 -> Linear
|      | |      | |---fc2 -> Linear
|      | |------7 -> ConvNeXtBlock
|      | |      |-conv_dw -> Conv2d
|      | |      |-norm -> LayerNorm
|      | |      |-mlp -> Mlp
|      | |      | |---fc1 -> Linear
|      | |      | |---fc2 -> Linear
|      | |------8 -> ConvNeXtBlock
|      | |      |-conv_dw -> Conv2d
|      | |      |-norm -> LayerNorm
|      | |      |-mlp -> Mlp
|      | |      | |---fc1 -> Linear
|      | |      | |---fc2 -> Linear
|      | |------9 -> ConvNeXtBlock
|      | |      |-conv_dw -> Conv2d
|      | |      |-norm -> LayerNorm
|      | |      |-mlp -> Mlp
|      | |      | |---fc1 -> Linear
|      | |      | |---fc2 -> Linear
|      | |------10 -> ConvNeXtBlock
|      | |      |--conv_dw -> Conv2d
|      | |      |--norm -> LayerNorm
|      | |      |--mlp -> Mlp
|      | |      |  |---fc1 -> Linear
|      | |      |  |---fc2 -> Linear
|      | |------11 -> ConvNeXtBlock
|      | |      |--conv_dw -> Conv2d
|      | |      |--norm -> LayerNorm
|      | |      |--mlp -> Mlp
|      | |      |  |---fc1 -> Linear
|      | |      |  |---fc2 -> Linear
|      | |------12 -> ConvNeXtBlock
|      | |      |--conv_dw -> Conv2d
|      | |      |--norm -> LayerNorm
|      | |      |--mlp -> Mlp
|      | |      |  |---fc1 -> Linear
|      | |      |  |---fc2 -> Linear
|      | |------13 -> ConvNeXtBlock
|      | |      |--conv_dw -> Conv2d
|      | |      |--norm -> LayerNorm
|      | |      |--mlp -> Mlp
|      | |      |  |---fc1 -> Linear
|      | |      |  |---fc2 -> Linear
|      | |------14 -> ConvNeXtBlock
|      | |      |--conv_dw -> Conv2d
|      | |      |--norm -> LayerNorm
|      | |      |--mlp -> Mlp
|      | |      |  |---fc1 -> Linear
|      | |      |  |---fc2 -> Linear
|      | |------15 -> ConvNeXtBlock
|      | |      |--conv_dw -> Conv2d
|      | |      |--norm -> LayerNorm
|      | |      |--mlp -> Mlp
|      | |      |  |---fc1 -> Linear
|      | |      |  |---fc2 -> Linear
|      | |------16 -> ConvNeXtBlock
|      | |      |--conv_dw -> Conv2d
|      | |      |--norm -> LayerNorm
|      | |      |--mlp -> Mlp
|      | |      |  |---fc1 -> Linear
|      | |      |  |---fc2 -> Linear
|      | |------17 -> ConvNeXtBlock
|      | |      |--conv_dw -> Conv2d
|      | |      |--norm -> LayerNorm
|      | |      |--mlp -> Mlp
|      | |      |  |---fc1 -> Linear
|      | |      |  |---fc2 -> Linear
|      | |------18 -> ConvNeXtBlock
|      | |      |--conv_dw -> Conv2d
|      | |      |--norm -> LayerNorm
|      | |      |--mlp -> Mlp
|      | |      |  |---fc1 -> Linear
|      | |      |  |---fc2 -> Linear
|      | |------19 -> ConvNeXtBlock
|      | |      |--conv_dw -> Conv2d
|      | |      |--norm -> LayerNorm
|      | |      |--mlp -> Mlp
|      | |      |  |---fc1 -> Linear
|      | |      |  |---fc2 -> Linear
|      | |------20 -> ConvNeXtBlock
|      | |      |--conv_dw -> Conv2d
|      | |      |--norm -> LayerNorm
|      | |      |--mlp -> Mlp
|      | |      |  |---fc1 -> Linear
|      | |      |  |---fc2 -> Linear
|      | |------21 -> ConvNeXtBlock
|      | |      |--conv_dw -> Conv2d
|      | |      |--norm -> LayerNorm
|      | |      |--mlp -> Mlp
|      | |      |  |---fc1 -> Linear
|      | |      |  |---fc2 -> Linear
|      | |------22 -> ConvNeXtBlock
|      | |      |--conv_dw -> Conv2d
|      | |      |--norm -> LayerNorm
|      | |      |--mlp -> Mlp
|      | |      |  |---fc1 -> Linear
|      | |      |  |---fc2 -> Linear
|      | |------23 -> ConvNeXtBlock
|      | |      |--conv_dw -> Conv2d
|      | |      |--norm -> LayerNorm
|      | |      |--mlp -> Mlp
|      | |      |  |---fc1 -> Linear
|      | |      |  |---fc2 -> Linear
|      | |------24 -> ConvNeXtBlock
|      | |      |--conv_dw -> Conv2d
|      | |      |--norm -> LayerNorm
|      | |      |--mlp -> Mlp
|      | |      |  |---fc1 -> Linear
|      | |      |  |---fc2 -> Linear
|      | |------25 -> ConvNeXtBlock
|      | |      |--conv_dw -> Conv2d
|      | |      |--norm -> LayerNorm
|      | |      |--mlp -> Mlp
|      | |      |  |---fc1 -> Linear
|      | |      |  |---fc2 -> Linear
|      | |------26 -> ConvNeXtBlock
|      | |      |--conv_dw -> Conv2d
|      | |      |--norm -> LayerNorm
|      | |      |--mlp -> Mlp
|      | |      |  |---fc1 -> Linear
|      | |      |  |---fc2 -> Linear
|------3 -> ConvNeXtStage
|      |-downsample -> Sequential
|      | |----------0 -> LayerNorm2d
|      | |----------1 -> Conv2d
|      |-blocks -> Sequential
|      | |------0 -> ConvNeXtBlock
|      | |      |-conv_dw -> Conv2d
|      | |      |-norm -> LayerNorm
|      | |      |-mlp -> Mlp
|      | |      | |---fc1 -> Linear
|      | |      | |---fc2 -> Linear
|      | |------1 -> ConvNeXtBlock
|      | |      |-conv_dw -> Conv2d
|      | |      |-norm -> LayerNorm
|      | |      |-mlp -> Mlp
|      | |      | |---fc1 -> Linear
|      | |      | |---fc2 -> Linear
|      | |------2 -> ConvNeXtBlock
|      | |      |-conv_dw -> Conv2d
|      | |      |-norm -> LayerNorm
|      | |      |-mlp -> Mlp
|      | |      | |---fc1 -> Linear
|      | |      | |---fc2 -> Linear
head -> NormMlpClassifierHead
|----norm -> LayerNorm2d
|----fc -> Linear
```
