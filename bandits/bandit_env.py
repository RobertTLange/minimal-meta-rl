import numpy as np
from gym import spaces


class BernoulliMAB(object):
    def __init__(self, num_arms=2, bandit_type="independent", max_steps=100):
        """ Set initial number of arms & reward func/env uncertainty """
        self.num_arms = num_arms
        self.bandit_type = bandit_type
        self.observation_space = None
        self.action_space = spaces.Discrete(self.num_arms)
        self.max_steps = max_steps

    def set_params(self, env_params):
        """ Set the parameters of the environment """
        self.bandit_type = env_params["bandit_type"]
        self.num_arms = env_params["num_arms"]
        self.max_steps = env_params["max_steps"]

    def reset(self):
        """ Set the reward function - Note that this resamples parameters! """
        self.timestep = 0
        self.done = False
        self.init_reward_function()
        return self.timestep

    def init_reward_function(self):
        """ Select a bandit type to train on/define reward function"""
        if self.bandit_type == "independent":
            self.arm_reward_means = np.random.uniform(low=0, high=1, size=self.num_arms)
        elif self.bandit_type == "dependent-uniform" and self.num_arms == 2:
            p1 = np.random.uniform(low=0, high=1, size=1)[0]
            self.arm_reward_means = np.array([p1, 1-p1])
        elif self.bandit_type == "dependent-easy" and self.num_arms == 2:
            p1 = np.random.choice([0.1, 0.9], size=1)[0]
            self.arm_reward_means = np.array([p1, 1-p1])
        elif self.bandit_type == "dependent-medium" and self.num_arms == 2:
            p1 = np.random.choice([0.25, 0.75], size=1)[0]
            self.arm_reward_means = np.array([p1, 1-p1])
        elif self.bandit_type == "dependent-hard" and self.num_arms == 2:
            p1 = np.random.choice([0.4, 0.6], size=1)[0]
            self.arm_reward_means = np.array([p1, 1-p1])
        elif self.bandit_type == "dependent-info" and self.num_arms == 11:
            a_11 = np.random.choice(np.linspace(0.1, 1, 10))
            high_rew_arm = int(a_11*10)
            arm_rew_temp = np.ones(11)
            arm_rew_temp[high_rew_arm-1] = 5
            arm_rew_temp[10] = a_11
            self.arm_reward_means = arm_rew_temp
        else:
            raise ValueError("Provide a valid Bernoulli bandit type")
        self.best_exp_arm = np.argmax(self.arm_reward_means)

    def step(self, action, return_regret=False):
        """ Perform a pull from the Bernoulli Bandit """
        if self.done:
            raise RuntimeError("Episode has finished. Call env.reset() to" +
                               "start a new episode.")
        self.timestep += 1
        if self.timestep >= self.max_steps:
            self.done = True

        reward = np.random.binomial(n=1, p=self.arm_reward_means[action])
        if return_regret:
            regret = self.arm_reward_means[self.best_exp_arm] - self.arm_reward_means[action]
            suboptimal_pull = (self.best_exp_arm != action)
            return reward, regret, suboptimal_pull, self.done, self.timestep

        return reward, self.done, self.timestep
