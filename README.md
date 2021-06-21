[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Planning Mixture of Experts
This repository contains PyTorch implementation of Planning Mixture of Experts (PMoE).

## Dependencies
This repo uses PyTorch and CARLA simulator (0.9.10.1). Please make sure that all dependencies are installed by running:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Please specify the CARLA root directory at the top of `run.sh` file.
```bash
export CARLA_ROOT=~/Projects/nazeri/CARLA_0.9.10.1          # change to where you installed CARLA
```
You need at least a middle range GPU to run the experiments with minimum possible batch size.

## Training
To run different training stages, please see [experiments doc](docs/experiments.md) first. You can define the 
training and model parameters via the respective config file 
in `conf` directory, then, use `run.sh stage_number` to run training where `stage_number` can be one of 
`stage0`, `stage1`, `stage2`, `stage3`. To see more options, and their description execute `./run.sh -h`

## Evaluation
To evaluate the models there are two options, Carla leaderboard evaluation or nocrash evaluation which can 
be run via `./run.sh benchmark` or `./run.sh nocrash` respectively. You can configure the model and benchmark 
parameters through the `conf/benchmark.yaml` config file.

## Dataset
The dataset was collect using [this repo](https://github.com/mhnazeri/carla_data_collector) which is fork of [LBC leaderboard submission](https://github.com/bradyz/2020_CARLA_challenge).

## Acknowledgements
Thanks for [World on Rails](https://github.com/dotchen/WorldOnRails), [Learning by Cheating](https://github.com/dotchen/LearningByCheating),
[CARLA leaderboard](https://github.com/carla-simulator/leaderboard.git),
[CARLA scenario_runner](https://github.com/carla-simulator/scenario_runner.git),
[Coiltraine](https://github.com/felipecode/coiltraine), repos for making their code public.

## License
This repo is released under the MIT License (please refer to the LICENSE file for details). Part of the PythonAPI and the map rendering code is borrowed from the official [CARLA repo](https://github.com/carla-simulator/carla), which is under MIT license. 
The image augmentation code is borrowed from [Coiltraine](https://github.com/felipecode/coiltraine) which is released under MIT license.
The evaluation code is borrowed from [World on Rails](https://github.com/dotchen/WorldOnRails), and 
[Learning by Cheating](https://github.com/dotchen/LearningByCheating) which also released under the MIT License.

