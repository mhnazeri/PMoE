A brief overview of different experiments.

## Stage 2 of Training
In stage 2, we predict actions. We have different configurations for
predicting actions such as use MoE alone or PU-Net alone or their mixture.
Below, we are going to discuss these different combinations.

### Experiment 0: PU-Net
This experiment compares the use intermediate representation
for action prediction in PU-Net network. To achieve this, set `model.type` to `punet` for
using PU-Net + backbone or set it to 'punet_inter' to use only PU-Net as backbone with intermediate
representations.

```yaml
model:
  type: 'punet_inter' # network type. Valid values are: moe, punet, punet_inter, pmoe, pmoe+pretrained, moe_alt
  action_head: # speed encoder network, just an MLP
    dims: [ 1536, 512, 512] # MLP dimensions. First dim should always be 1536
    act: 'elu' # type of activation function. Valid values: [relu, tanh, sigmoid, elu]
    l_act: True # use activation function on the last layer or just return scores/logits?
    bn: True # use batchnorm?
    dropout: 0.0
  speed_encoder: # speed encoder network, just an MLP
    dims: [1, 512, 512] # MLP dimensions. First dim should always be 1
    act: 'relu' # type of activation function. Valid values: [relu, tanh, sigmoid, elu]
    l_act: False # use activation function on the last layer or just return scores/logits?
    bn: True # use batchnorm?
    dropout: 0.0
  command_encoder: # command encoder network, just an MLP
    dims: [ 4, 512, 512] # MLP dimensions. First dim should always be 4
    act: 'relu' # type of activation function. Valid values: [relu, tanh, sigmoid, elu]
    l_act: False # use activation function on the last layer or just return scores/logits?
    bn: True # use batchnorm?
    dropout: 0.0
  speed_prediction: # speed prediction head, just an MLP
    dims: [ 1536, 512, 512, 1] # MLP dimensions. First dim should always be 1536
    act: 'relu' # type of activation function. Valid values: [relu, tanh, sigmoid, elu]
    l_act: False # use activation function on the last layer or just return scores/logits?
    bn: True # use batchnorm?
    dropout: 0.0
  backbone: # visual feature extractor
    type: 'rgb' # valid values: rgb, segmentation
    n_frames: 4 # number of input frames
    rgb: # configs for resnet and mobilenet models
      arch: 'resnet18' # valid values: resnet18/34/50, mobilenet_v2/3_small/large
      pretrained: False
      gamma: 2
      b: 1
    segmentation:
      gamma: 2
      b: 1
      inter_repr: True # always true for this stage
      model_dir: "../checkpoint/unet-e3-swa.pth"
  punet:
    past_frames: 4 # please note that data is captured at 2hz
    future_frames: 4
    in_features: 3
    num_classes: 23 # number of segmentation classes, for carla 0.9.6+ it is 23
    gamma: 2 # gamma parameter of eca module
    b: 1 # b parameter of eca module
    inter_repr: False # whether PU-Net returns intermediate representation or not (not intended for training)
    unet_inter_repr: False # whether U-Net returns intermediate representation or not (set to True if future_frames is 0)
    model_name: "unet-swa"
    model_path: "../checkpoint/unet-e2-swa-best.pth"
```


### Experiment 1: MoE
In the first experiment we are going to train MoE experts
only. The only parameters that are changed in this
experiment is `backbone`. The type of model corresponds
whether to extract features from RGB images or use
from segmentations. For extracting segmentations
currently only U-Net is implemented and past frames
are given different weights dynamically using
cross channel co-efficients. `n_experts` corresponds to
the number of agents to have:

```yaml
model:
  n_experts: 3
  backbone: # visual feature extractor
    type: 'rgb' # valid values: rgb, segmentation
    n_frames: 4 # number of input frames
    rgb: # configs for resnet and mobilenet models
      arch: 'resnet18' # valid values: resnet18/34/50, mobilenet_v2/3_small/large
      pretrained: False
      gamma: 2
      b: 1
    segmentation:
      gamma: 2
      b: 1
      inter_repr: True
      model_dir: "../checkpoint/unet-e3-swa.pth"
```

### Experiment 2: PMoE
To train PMoE model, there is two possibility. Either you can train PMoE with PU-Net pretrained on action
prediction or not. To do so, change the model type to 'pmoe+pretrained' to use pretrained PU-Net and also you should
provide the PU-Net model directory in `model.pmoe.punet_dir`. Using `pmoe` alone as `model.type` will not use
PU-Net pretrained weights.