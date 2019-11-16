"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from playground import maybe_initialize_process
from copy import deepcopy
import multiprocessing as m


def launch_local(
    baseline,
    variant,
    env_class,
    env_kwargs=None,
    observation_key="observation",
    num_seeds=2
):
    # initialize tensorflow and the multiprocessing interface
    maybe_initialize_process()

    # if only one seed is requested, then run in the main thread
    if num_seeds == 1:
        return baseline(
            variant,
            env_class,
            env_kwargs=env_kwargs,
            observation_key=observation_key)

    # launch the experiments on the local machine
    processes = []
    for seed in range(num_seeds):
        seed_variant = deepcopy(variant)
        seed_variant["logging_dir"] += "{}/".format(seed)
        processes.append(
            m.Process(
                target=baseline,
                args=(seed_variant, env_class),
                kwargs=dict(
                    env_kwargs=env_kwargs,
                    observation_key=observation_key)))

    # wait for every experiment to finish
    for p in processes:
        p.start()
    for p in processes:
        p.join()
