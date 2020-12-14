import torch


def a2c_loss(log_probs, advantage, entropy, entropy_beta=0.02,
             value_beta=0.5, apply_mean=True, return_all=False):
    """ Actor (REINFORCE) + Critic (Bellman Con.) + Entropy (Explore) Loss. """
    a_loss = -(log_probs * advantage.detach()).mean()
    c_loss = advantage.pow(2).mean()

    if apply_mean:
        actor_loss = a_loss.mean()
        critic_loss = c_loss.mean()
        entropy_loss = entropy.mean()
    else:
        actor_loss = a_loss.sum()
        critic_loss = c_loss.sum()
        entropy_loss = entropy.sum()

    # Collect the 3 loss terms weighted by coefficients
    full_loss = actor_loss + value_beta*critic_loss - entropy_beta*entropy_loss
    if return_all:
        return full_loss, actor_loss.item(), critic_loss.item(), entropy_loss.item()
    else:
        return full_loss


def compute_returns(rewards, next_value=None, masks=None, gamma=1.):
    """
    Compute list of returns up to T - if next_value: use bootstrap from critic
    - returns[-1] = r_T + gamma+next_value
    - returns[-2] = r_T-1 + gamma*r_T + gamma**2*next_value
    ....
    - returns[0] = sum_{t=1}^T gamma*{t-1} r_t + gamma**T*next_value
    """
    # Use bootstrapping if v estimate is provided - otw. set 0 (finite horizon)
    if next_value is not None:
        R = next_value
    else:
        if type(rewards) == torch.Tensor:
            R = torch.zeros(rewards[0].size())
        else:
            R = torch.Tensor([0])
    returns = []

    for step in reversed(range(len(rewards))):
        if masks is not None:
            R = rewards[step] + gamma * R * masks[step]
        else:
            R = rewards[step] + gamma * R
        returns.insert(0, R)
    return returns
