import torch
import numpy as np
from utils.general import one_hot_encode, normalize_time
from actor_critic.loss_fct import compute_returns
from bandits.bandit_env import BernoulliMAB


def run_bandit_episode(env, meta_net, discount_factor, device,
                       test_agent=False, use_bootstrap=True):
    """ Run episode in env + return quantities to learn from. """
    action_hist, log_probs, values, rewards, entropy = [], [], [], [], []
    cum_regret, cum_suboptimal_pulls = [], []

    # Rest resamples a new bandit from the type chosen when instantiated
    t = env.reset()
    done = False
    # Loop over individual episode timesteps
    while not done:
        if t == 0:
            hidden = meta_net.init_hidden(device, 1)
            temp_in = torch.tensor([0, normalize_time(t,
                                    env.max_steps, horizon=False)],
                                   dtype=torch.float)
            concat_in = torch.cat((temp_in, one_hot_encode(env.num_arms, 0)))
            in_rnn = concat_in.view(1, 1, -1).to(device)
        else:
            temp_in = torch.tensor([rewards[-1], normalize_time(t,
                                    env.max_steps, horizon=False)],
                                   dtype=torch.float)

            concat_in = torch.cat((temp_in, one_hot_encode(env.num_arms, action_hist[-1])))
            in_rnn = concat_in.view(1, 1, -1).to(device)

        policy_out, value_out, hidden = meta_net(in_rnn, hidden)
        action = policy_out.sample()
        reward, regret, suboptimal_pull, done, t = env.step(action.item(),
                                                            return_regret=True)
        cum_regret.append(regret)
        cum_suboptimal_pulls.append(suboptimal_pull)

        log_prob = policy_out.log_prob(action)
        entropy.append(policy_out.entropy().view(-1))
        log_probs.append(log_prob.view(-1))
        values.append(value_out.view(-1))
        rewards.append(reward)
        action_hist.append(action)

    # Use bootstrap value estimate to compute returns
    if use_bootstrap:
        temp_in = torch.tensor([rewards[-1], normalize_time(t,
                                env.max_steps, horizon=False)],
                               dtype=torch.float)
        concat_in = torch.cat((temp_in, one_hot_encode(env.num_arms,
                                                       action_hist[-1])))
        in_rnn = concat_in.view(1, 1, -1).to(device)
        _, value_next, _ = meta_net(in_rnn, hidden)
        # Compute multi-step returns with bootstrap value estimate [INFINITE]
        returns = compute_returns(rewards, value_next, gamma=discount_factor)
    else:
        # Compute multi-step returns without bootstrap value estimate [FINITE]
        returns = compute_returns(rewards, gamma=discount_factor)

    # Get tracked values ready for loss computation
    log_probs = torch.cat(log_probs)
    returns   = torch.cat(returns).view(-1).detach()
    values    = torch.cat(values)
    entropy   = torch.cat(entropy)
    advantage = returns - values

    if not test_agent:
        return (log_probs, advantage, entropy, (sum(rewards),
                np.sum(cum_regret), np.sum(cum_suboptimal_pulls)))
    else:
        return sum(rewards), np.cumsum(cum_regret), np.cumsum(cum_suboptimal_pulls)


def test_bandit_agent(meta_net, device, train_config):
    """ Test the agent for a number of trials and episodes """
    rewards, cum_regrets, cum_suboptimal_pulls = [], [], []

    for episode in range(train_config.test_num_episodes):
        # "Sample"/Init new bandit at the beginning of each episode
        lifetime = train_config.test_lifetime

        # Evaluate meta-agent on bandit
        env = BernoulliMAB(num_arms=train_config.num_arms,
                           bandit_type=train_config.test_bandit_type,
                           max_steps=lifetime)

        reward_temp, cum_regret_temp, cum_suboptimal_temp = run_bandit_episode(env,
                                        meta_net, 1, device,
                                        test_agent=True, use_bootstrap=False)
        rewards.append(reward_temp)
        cum_regrets.append(cum_regret_temp)
        cum_suboptimal_pulls.append(cum_suboptimal_temp)

    # Return the mean regret/no. of suboptimal arm pulls over different eps
    avg_reward = np.mean(rewards)
    avg_regret = np.mean(cum_regrets, axis=0)
    avg_suboptimal_pulls = np.mean(cum_suboptimal_pulls, axis=0)
    return avg_reward, avg_regret, avg_suboptimal_pulls
