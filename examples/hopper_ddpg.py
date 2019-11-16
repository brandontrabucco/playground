"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from playground.launch import launch_local
from playground.baselines.ddpg import ddpg, ddpg_variant
from gym.envs.mujoco.hopper import HopperEnv


if __name__ == "__main__":

    # parameters for the learning experiment
    variant = dict(
        max_num_steps=1000000,
        logging_dir="hopper_test2/ddpg/",
        hidden_size=400,
        num_hidden_layers=2,
        reward_scale=1.0,
        discount=0.99,
        policy_learning_rate=0.0001,
        qf_learning_rate=0.001,
        tau=0.005,
        exploration_noise_std=0.2,
        batch_size=256,
        max_path_length=500,
        num_workers=2,
        num_warm_up_steps=5000,
        num_steps_per_epoch=500,
        num_steps_per_eval=5000,
        num_epochs_per_eval=10,
        num_epochs=10000)

    # make sure that all the right parameters are here
    assert all([x in variant.keys() for x in ddpg_variant.keys()])

    # launch the experiment using ray
    launch_local(
        ddpg,
        variant,
        HopperEnv,
        num_seeds=1)
