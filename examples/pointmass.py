"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from playground.launch import launch_local
from playground.baselines.sac import sac, sac_variant
from playground.envs.pointmass_env import PointmassEnv


if __name__ == "__main__":

    # parameters for the learning experiment
    variant = dict(
        max_num_steps=10000,
        logging_dir="pointmass/sac/",
        hidden_size=400,
        num_hidden_layers=2,
        reward_scale=1.0,
        discount=0.99,
        initial_alpha=0.01,
        lr=0.0003,
        tau=0.05,
        batch_size=256,
        max_path_length=10,
        num_workers=2,
        num_warm_up_steps=100,
        num_steps_per_epoch=10,
        num_steps_per_eval=100,
        num_epochs_per_eval=10,
        num_epochs=10000)

    # make sure that all the right parameters are here
    assert all([x in variant.keys() for x in sac_variant.keys()])

    # launch the experiment using ray
    launch_local(
        sac,
        variant,
        PointmassEnv,
        num_seeds=2)
