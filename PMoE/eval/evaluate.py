import sys
from pathlib import Path
try:
    sys.path.append(str(Path("../").resolve()))
except:
    raise RuntimeError("Can't append root directory of the project the path")

from runners import ChallengeRunner


def main(args):
    scenario = 'PMoE/assets/all_towns_traffic_scenarios.json'
    # scenario = 'PMoE/assets/no_scenarios.json'
    # route = 'PMoE/assets/routes_dev.xml'
    route = 'PMoE/assets/routes_training/route_10.xml'

    args.agent = 'PMoE/autoagents/image_agent'
    args.agent_config = args.agent_config

    port = args.port
    tm_port = port + 2
    runner = ChallengeRunner(args, scenario, route, port=port, tm_port=tm_port)
    runner.run()



if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--agent-config', default='PMoE/conf/benchmark.yaml')
    parser.add_argument('--trafficManagerSeed', default='0',
                        help='Seed used by the TrafficManager (default: 0)')
    parser.add_argument('--timeout', default="60.0",
                        help='Set the CARLA client timeout value in seconds')

    parser.add_argument('--port', type=int, default=2000)

    parser.add_argument('--repetitions',
                        type=int,
                        default=1,
                        help='Number of repetitions per route.')
    parser.add_argument("--track", type=str, default='SENSORS', help="Participation track: SENSORS, MAP")
    parser.add_argument('--resume', type=bool, default=False, help='Resume execution from last checkpoint?')
    parser.add_argument("--checkpoint", type=str,
                        default='PMoE/benchmark_results/simulation_results.json',
                        help="Path to checkpoint used for saving statistics and resuming")

    args = parser.parse_args()
    main(args)
