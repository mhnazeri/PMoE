#!/bin/bash
export CARLA_ROOT=~/Projects/nazeri/CARLA_0.9.10.1          # change to where you installed CARLA
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/agents
export PYTHONPATH=$PYTHONPATH:PMoE
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner


export PORT=2000                                                    # change to port that CARLA is running
export ROUTES=leaderboard/data/routes_training/route_00.xml         # change to desired route
export TEAM_AGENT=auto_pilot.py                                     # no need to change
export TEAM_CONFIG=data                                             # change path to save data


if [ -d "$TEAM_CONFIG" ]; then
        CHECKPOINT_ENDPOINT="$TEAM_CONFIG/$(basename $ROUTES .xml).txt"
    else
            CHECKPOINT_ENDPOINT="$(dirname $TEAM_CONFIG)/$(basename $ROUTES .xml).txt"
fi

Help()
{
   # Display Help
   echo 
   echo "Facilitates running different stages of training and evaluation (Carla root should be defined at the beginning of this file)."
   echo 
   echo "options:"
   echo "stage0         Starts training for stage 0, trains U-Net to predict image segmentation maps."
   echo "stage1         Starts training for stage 1, trains U-Net to predict future segmentation maps."
   echo "stage2         Starts training for stage 2, trains different models to output control commands."
   echo "stage3         Starts training for stage 3, refine models on driving task itself. This stage requires Carla."
   echo "benchmark      Benchmark agent against CORL 2017 and NoCrash benchmarks."
   echo "view_benchmark Print benchmark results."   
   echo "leaderboard    Benchmark agent against Carla leaderboard."
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
      python PMoE/eval/benchmark_agent.py
      ;;
    view_benchmark)
      python PMoE/eval/view_benchmark_results.py
      ;;
    leaderboard)
      python3 leaderboard/leaderboard/leaderboard_evaluator.py \
              --track=SENSORS \
              --scenarios=leaderboard/data/all_towns_traffic_scenarios_public.json  \
              --agent=${TEAM_AGENT} \
              --agent-config=${TEAM_CONFIG} \
              --routes=${ROUTES} \
              --checkpoint=${CHECKPOINT_ENDPOINT} \
              --port=${PORT}
      ;;
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

run $1 $2

echo "Done."
