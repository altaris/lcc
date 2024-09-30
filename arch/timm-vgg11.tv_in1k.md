# [`timm/vgg11.tv_in1k`](https://huggingface.co/timm/vgg11.tv_in1k)

```
features -> Sequential
|--------0 -> Conv2d
|--------3 -> Conv2d
|--------6 -> Conv2d
|--------8 -> Conv2d
|--------11 -> Conv2d
|--------13 -> Conv2d
|--------16 -> Conv2d
|--------18 -> Conv2d
pre_logits -> ConvMlp
|----------fc1 -> Conv2d
|----------fc2 -> Conv2d
head -> ClassifierHead
|----fc -> Linear
```
