[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

PyTorch implementation of PMoE.

## Training
To run different training stages, first define the training and model parameters via the respective config file in `conf` directory, then, use `run.sh stage_number` to run training where `stage_number` can be one of `stage0`, `stage1`, `stage2`, `stage3`. To see more options and their description execute `./run.sh -h`

## Evaluation
To evaluate the models there are two options, Carla leaderboard evaluation or benchmark evaluation which can be run via
`./run.sh leaderboard` or `./run.sh benchmark` respectively. You can configure the model and benchmark parameters through
the `conf/benchmark.yaml` config file.
# License
License
This repo is released under the MIT License (please refer to the LICENSE file for details). Part of the PythonAPI and the map rendering code is borrowed from the official [CARLA repo](https://github.com/carla-simulator/carla), which is under MIT license. The image augmentation code is borrowed from [Coiltraine](https://github.com/felipecode/coiltraine) which is released under MIT license.

