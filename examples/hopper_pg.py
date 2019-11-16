"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from playground.launch import launch_local
from playground.baselines.policy_gradient import policy_gradient, policy_gradient_variant
from gym.envs.mujoco.hopper import HopperEnv


if __name__ == "__main__":

    # parameters for the learning experiment
    variant = dict(
        max_path_length=500,
        max_num_paths=1000,
        logging_dir="hopper_test2/pg/",
        hidden_size=400,
        num_hidden_layers=2,
        reward_scale=1.0,
        discount=0.99,
        policy_learning_rate=0.0001,
        exploration_noise_std=0.5,
        num_workers=10,
        num_steps_per_epoch=5000,
        num_steps_per_eval=5000,
        num_epochs_per_eval=10,
        num_epochs=1000)

    # make sure that all the right parameters are here
    assert all([x in variant.keys() for x in policy_gradient_variant.keys()])

    # launch the experiment using ray
    launch_local(
        policy_gradient,
        variant,
        HopperEnv,
        num_seeds=1)
