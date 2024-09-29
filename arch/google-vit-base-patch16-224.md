# [`google/vit-base-patch16-224`](https://huggingface.co/google/vit-base-patch16-224)

```
vit -> ViTModel
|---embeddings -> ViTEmbeddings
|   |----------patch_embeddings -> ViTPatchEmbeddings
|   |          |----------------projection -> Conv2d
|---encoder -> ViTEncoder
|   |-------layer -> ModuleList
|   |       |-----0 -> ViTLayer
|   |       |     |-attention -> ViTSdpaAttention
|   |       |     | |---------attention -> ViTSdpaSelfAttention
|   |       |     | |         |---------query -> Linear
|   |       |     | |         |---------key -> Linear
|   |       |     | |         |---------value -> Linear
|   |       |     | |---------output -> ViTSelfOutput
|   |       |     | |         |------dense -> Linear
|   |       |     |-intermediate -> ViTIntermediate
|   |       |     | |------------dense -> Linear
|   |       |     |-output -> ViTOutput
|   |       |     | |------dense -> Linear
|   |       |     |-layernorm_before -> LayerNorm
|   |       |     |-layernorm_after -> LayerNorm
|   |       |-----1 -> ViTLayer
|   |       |     |-attention -> ViTSdpaAttention
|   |       |     | |---------attention -> ViTSdpaSelfAttention
|   |       |     | |         |---------query -> Linear
|   |       |     | |         |---------key -> Linear
|   |       |     | |         |---------value -> Linear
|   |       |     | |---------output -> ViTSelfOutput
|   |       |     | |         |------dense -> Linear
|   |       |     |-intermediate -> ViTIntermediate
|   |       |     | |------------dense -> Linear
|   |       |     |-output -> ViTOutput
|   |       |     | |------dense -> Linear
|   |       |     |-layernorm_before -> LayerNorm
|   |       |     |-layernorm_after -> LayerNorm
|   |       |-----2 -> ViTLayer
|   |       |     |-attention -> ViTSdpaAttention
|   |       |     | |---------attention -> ViTSdpaSelfAttention
|   |       |     | |         |---------query -> Linear
|   |       |     | |         |---------key -> Linear
|   |       |     | |         |---------value -> Linear
|   |       |     | |---------output -> ViTSelfOutput
|   |       |     | |         |------dense -> Linear
|   |       |     |-intermediate -> ViTIntermediate
|   |       |     | |------------dense -> Linear
|   |       |     |-output -> ViTOutput
|   |       |     | |------dense -> Linear
|   |       |     |-layernorm_before -> LayerNorm
|   |       |     |-layernorm_after -> LayerNorm
|   |       |-----3 -> ViTLayer
|   |       |     |-attention -> ViTSdpaAttention
|   |       |     | |---------attention -> ViTSdpaSelfAttention
|   |       |     | |         |---------query -> Linear
|   |       |     | |         |---------key -> Linear
|   |       |     | |         |---------value -> Linear
|   |       |     | |---------output -> ViTSelfOutput
|   |       |     | |         |------dense -> Linear
|   |       |     |-intermediate -> ViTIntermediate
|   |       |     | |------------dense -> Linear
|   |       |     |-output -> ViTOutput
|   |       |     | |------dense -> Linear
|   |       |     |-layernorm_before -> LayerNorm
|   |       |     |-layernorm_after -> LayerNorm
|   |       |-----4 -> ViTLayer
|   |       |     |-attention -> ViTSdpaAttention
|   |       |     | |---------attention -> ViTSdpaSelfAttention
|   |       |     | |         |---------query -> Linear
|   |       |     | |         |---------key -> Linear
|   |       |     | |         |---------value -> Linear
|   |       |     | |---------output -> ViTSelfOutput
|   |       |     | |         |------dense -> Linear
|   |       |     |-intermediate -> ViTIntermediate
|   |       |     | |------------dense -> Linear
|   |       |     |-output -> ViTOutput
|   |       |     | |------dense -> Linear
|   |       |     |-layernorm_before -> LayerNorm
|   |       |     |-layernorm_after -> LayerNorm
|   |       |-----5 -> ViTLayer
|   |       |     |-attention -> ViTSdpaAttention
|   |       |     | |---------attention -> ViTSdpaSelfAttention
|   |       |     | |         |---------query -> Linear
|   |       |     | |         |---------key -> Linear
|   |       |     | |         |---------value -> Linear
|   |       |     | |---------output -> ViTSelfOutput
|   |       |     | |         |------dense -> Linear
|   |       |     |-intermediate -> ViTIntermediate
|   |       |     | |------------dense -> Linear
|   |       |     |-output -> ViTOutput
|   |       |     | |------dense -> Linear
|   |       |     |-layernorm_before -> LayerNorm
|   |       |     |-layernorm_after -> LayerNorm
|   |       |-----6 -> ViTLayer
|   |       |     |-attention -> ViTSdpaAttention
|   |       |     | |---------attention -> ViTSdpaSelfAttention
|   |       |     | |         |---------query -> Linear
|   |       |     | |         |---------key -> Linear
|   |       |     | |         |---------value -> Linear
|   |       |     | |---------output -> ViTSelfOutput
|   |       |     | |         |------dense -> Linear
|   |       |     |-intermediate -> ViTIntermediate
|   |       |     | |------------dense -> Linear
|   |       |     |-output -> ViTOutput
|   |       |     | |------dense -> Linear
|   |       |     |-layernorm_before -> LayerNorm
|   |       |     |-layernorm_after -> LayerNorm
|   |       |-----7 -> ViTLayer
|   |       |     |-attention -> ViTSdpaAttention
|   |       |     | |---------attention -> ViTSdpaSelfAttention
|   |       |     | |         |---------query -> Linear
|   |       |     | |         |---------key -> Linear
|   |       |     | |         |---------value -> Linear
|   |       |     | |---------output -> ViTSelfOutput
|   |       |     | |         |------dense -> Linear
|   |       |     |-intermediate -> ViTIntermediate
|   |       |     | |------------dense -> Linear
|   |       |     |-output -> ViTOutput
|   |       |     | |------dense -> Linear
|   |       |     |-layernorm_before -> LayerNorm
|   |       |     |-layernorm_after -> LayerNorm
|   |       |-----8 -> ViTLayer
|   |       |     |-attention -> ViTSdpaAttention
|   |       |     | |---------attention -> ViTSdpaSelfAttention
|   |       |     | |         |---------query -> Linear
|   |       |     | |         |---------key -> Linear
|   |       |     | |         |---------value -> Linear
|   |       |     | |---------output -> ViTSelfOutput
|   |       |     | |         |------dense -> Linear
|   |       |     |-intermediate -> ViTIntermediate
|   |       |     | |------------dense -> Linear
|   |       |     |-output -> ViTOutput
|   |       |     | |------dense -> Linear
|   |       |     |-layernorm_before -> LayerNorm
|   |       |     |-layernorm_after -> LayerNorm
|   |       |-----9 -> ViTLayer
|   |       |     |-attention -> ViTSdpaAttention
|   |       |     | |---------attention -> ViTSdpaSelfAttention
|   |       |     | |         |---------query -> Linear
|   |       |     | |         |---------key -> Linear
|   |       |     | |         |---------value -> Linear
|   |       |     | |---------output -> ViTSelfOutput
|   |       |     | |         |------dense -> Linear
|   |       |     |-intermediate -> ViTIntermediate
|   |       |     | |------------dense -> Linear
|   |       |     |-output -> ViTOutput
|   |       |     | |------dense -> Linear
|   |       |     |-layernorm_before -> LayerNorm
|   |       |     |-layernorm_after -> LayerNorm
|   |       |-----10 -> ViTLayer
|   |       |     |--attention -> ViTSdpaAttention
|   |       |     |  |---------attention -> ViTSdpaSelfAttention
|   |       |     |  |         |---------query -> Linear
|   |       |     |  |         |---------key -> Linear
|   |       |     |  |         |---------value -> Linear
|   |       |     |  |---------output -> ViTSelfOutput
|   |       |     |  |         |------dense -> Linear
|   |       |     |--intermediate -> ViTIntermediate
|   |       |     |  |------------dense -> Linear
|   |       |     |--output -> ViTOutput
|   |       |     |  |------dense -> Linear
|   |       |     |--layernorm_before -> LayerNorm
|   |       |     |--layernorm_after -> LayerNorm
|   |       |-----11 -> ViTLayer
|   |       |     |--attention -> ViTSdpaAttention
|   |       |     |  |---------attention -> ViTSdpaSelfAttention
|   |       |     |  |         |---------query -> Linear
|   |       |     |  |         |---------key -> Linear
|   |       |     |  |         |---------value -> Linear
|   |       |     |  |---------output -> ViTSelfOutput
|   |       |     |  |         |------dense -> Linear
|   |       |     |--intermediate -> ViTIntermediate
|   |       |     |  |------------dense -> Linear
|   |       |     |--output -> ViTOutput
|   |       |     |  |------dense -> Linear
|   |       |     |--layernorm_before -> LayerNorm
|   |       |     |--layernorm_after -> LayerNorm
|---layernorm -> LayerNorm
classifier -> Linear
```
