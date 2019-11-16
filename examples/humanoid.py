"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from playground.launch import launch_local
from playground.baselines.sac import sac, sac_variant
from gym.envs.mujoco.humanoid import HumanoidEnv


if __name__ == "__main__":

    # parameters for the learning experiment
    variant = dict(
        max_num_steps=1000000,
        logging_dir="humanoid/sac/",
        hidden_size=400,
        num_hidden_layers=2,
        reward_scale=0.01,
        discount=0.99,
        initial_alpha=0.01,
        lr=0.0003,
        tau=0.005,
        batch_size=256,
        max_path_length=1000,
        num_workers=2,
        num_warm_up_steps=10000,
        num_steps_per_epoch=1000,
        num_steps_per_eval=10000,
        num_epochs_per_eval=10,
        num_epochs=10000)

    # make sure that all the right parameters are here
    assert all([x in variant.keys() for x in sac_variant.keys()])

    # launch the experiment using ray
    launch_local(
        sac,
        variant,
        HumanoidEnv,
        num_seeds=1)
