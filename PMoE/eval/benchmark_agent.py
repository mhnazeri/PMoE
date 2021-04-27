"""source: https://github.com/dotchen/LearningByCheating"""
import time

from pathlib import Path
import glob
import sys

try:
    sys.path.append(glob.glob("../utils/PythonAPI")[0])
    sys.path.append(glob.glob("../")[0])
except IndexError as e:
    pass

from utils.benchmark import make_suite, get_suites, ALL_SUITES
from utils.benchmark.run_benchmark import run_benchmark
from utils.utility import get_conf


def _agent_factory_hack(config):
    """
    These imports before carla.Client() cause seg faults...
    """
    from model.roaming import RoamingAgentMine

    if config.env.autopilot:
        return RoamingAgentMine

    import torch

    from model.moe import get_model
    from model.rl_agent import Actor

    if config.model.type == "rl_agent":
        model = Actor(config.model)
    else:
        model = get_model(config.model)
        state_dict = torch.load(config.model.model_dir)
        model.load_state_dict(state_dict["model"])

    model.eval()

    return model


def run(config):
    log_dir = Path(config.logger.log_dir)
    total_time = 0.0

    for suite_name in get_suites(config.env.suite):
        tick = time.time()

        benchmark_dir = (
            log_dir / "benchmark" / ("%s_seed%d" % (suite_name, config.env.seed))
        )
        benchmark_dir.mkdir(parents=True, exist_ok=True)

        with make_suite(
            suite_name, port=config.env.port, big_cam=config.env.big_cam
        ) as env:
            agent_maker = _agent_factory_hack(config)

            run_benchmark(
                agent_maker,
                env,
                benchmark_dir,
                config.env.seed,
                config.env.autopilot,
                config.env.resume,
                max_run=config.env.max_run,
                show=config.env.show,
            )

        elapsed = time.time() - tick
        total_time += elapsed

        print("%s: %.3f hours." % (suite_name, elapsed / 3600))

    print("Total time: %.3f hours." % (total_time / 3600))


if __name__ == "__main__":
    config = get_conf("../conf/benchmark.yaml")
    run(config)
