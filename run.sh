#!/bin/bash
export CARLA_ROOT=~/Projects/nazeri/CARLA_0.9.10.1          # change to where you installed CARLA
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/agents
export PYTHONPATH=$PYTHONPATH:PMoE
export PYTHONPATH=$PYTHONPATH:PMoE/leaderboard
export PYTHONPATH=$PYTHONPATH:PMoE/scenario_runner
export COMET_LOGGING_CONSOLE=info

Help()
{
   # Display Help
   echo 
   echo "Facilitates running different stages of training and evaluation (Carla root should be defined at the beginning of this file)."
   echo 
   echo "options:"
   echo "stage0                      Starts training for stage 0, trains U-Net to predict image segmentation maps."
   echo "stage1                      Starts training for stage 1, trains U-Net to predict future segmentation maps."
   echo "stage2 conf_dir             Starts training for stage 2, trains different models to output control commands."
   echo "stage3                      Starts training for stage 3, refine models on driving task itself. This stage requires Carla."
   echo "benchmark                   Benchmark agent against CORL 2017 benchmarks"
   echo "nocrash town weather        Benchmark agent against NoCrash benchmark. Where town is {Town01, Town02}, weather {test, train}"
   echo "view_benchmark              Print benchmark results."   
   echo "leaderboard                 Benchmark agent against Carla leaderboard."
   echo
}

run () {
  case $1 in
    stage0)
      python PMoE/trainer/train_0.py
      ;;
    stage1)
      python PMoE/trainer/train_1.py
      ;;
    stage2)
      python PMoE/trainer/train_2.py $2
      ;;
    stage3)
      python PMoE/trainer/train_3.py
      ;;
    benchmark)
      python PMoE/eval/evaluate.py
      ;;
    nocrash)
      python PMoE/eval/evaluate_nocrash.py --town $2 --weather $3
      ;;
    view_benchmark)
      python PMoE/eval/view_benchmark_results.py
      ;;
#    leaderboard)
#      python3 leaderboard/leaderboard/leaderboard_evaluator.py
#      ;;
    -h) # display Help
      Help
      exit
      ;;
    *)
      echo "Unknown '$1' argument. Please run with '-h' argument to see more details."
      # Help
      exit
      ;;
  esac
}

run $1 $2 $3

echo "Done."
