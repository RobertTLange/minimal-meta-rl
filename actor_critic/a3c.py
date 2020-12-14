import copy
import random
import math
import time
import numpy as np

import torch
import torch.multiprocessing as mp
from torch.optim import Adam
import torch.optim as optim

from torch.nn.utils import clip_grad_value_

from actor_critic.loss_fct import a2c_loss
from utils.general import linearly_anneal, set_random_seeds
from bandits.bandit_env import BernoulliMAB
from bandits.bandit_helpers import run_bandit_episode, test_bandit_agent


def copy_model_over(from_model, to_model):
    """Copies model parameters from from_model to to_model"""
    for to_model, from_model in zip(to_model.parameters(),
                                    from_model.parameters()):
        to_model.data.copy_(from_model.data.clone())


def a3c_update_shared_model(global_net, shared_optimizer,
                            episode_number, gradient_updates_queue,
                            optimizer_lock, total_episodes):
    """ Worker - update shared model w. grads as they are put into queue. """
    # IMPORTANT: Asynchronous - apply gradients as soon as they come in!
    while True:
        with episode_number.get_lock():
            carry_on = episode_number.value < total_episodes
        if carry_on:
            # Receive gradient estimates and directly apply
            try:
                gradients = gradient_updates_queue.get()
                with optimizer_lock:
                    shared_optimizer.zero_grad()
                    for grads, params in zip(gradients, global_net.parameters()):
                        params._grad = grads #.clone()
                    shared_optimizer.step()
            except:
                break
        else:
            break


def a2c_update_shared_model(global_net, shared_optimizer, episode_number,
                            gradient_updates_queue, optimizer_lock,
                            num_workers, total_episodes):
    """ Worker - update shared model w. grads as they are put into queue. """
    # IMPORTANT: Synchronous - Collect grads until have set from each worker
    while True:
        with episode_number.get_lock():
            carry_on = episode_number.value < total_episodes
        if carry_on:
            # Receive grad estimates + weight until all workers done before apply
            try:
                gradients_seen = 0
                while gradients_seen < num_workers:
                    if gradients_seen == 0:
                        gradients = gradient_updates_queue.get()
                    else:
                        new_grads = gradient_updates_queue.get()
                        gradients = [grad + new_grad for grad,
                                     new_grad in zip(gradients, new_grads)]
                    gradients_seen += 1
                shared_optimizer.zero_grad()
                for grads, params in zip(gradients, global_net.parameters()):
                    params._grad = grads #.clone()
                shared_optimizer.step()
            except:
                break
        else:
            break


class Actor_Critic_Worker(mp.Process):
    """AC worker - collects transitions for desired no. of episodes """
    def __init__(self, worker_num, episodes_to_run, train_config, global_net,
                 local_net, shared_optimizer, global_ep_count, global_step_count,
                 optimizer_lock, results_queue, gradient_updates_queue, device):
        super(Actor_Critic_Worker, self).__init__()
        self.train_config = train_config
        self.worker_num = worker_num
        self.device = device
        set_random_seeds(self.worker_num)

        self.global_net = global_net
        self.local_net = local_net
        self.local_optimizer = Adam(self.local_net.parameters(),
                                    lr=0.0, eps=1e-4)

        self.global_ep_count = global_ep_count
        self.global_step_count = global_step_count
        self.optimizer_lock = optimizer_lock
        self.shared_optimizer = shared_optimizer

        self.episodes_to_run = episodes_to_run
        self.results_queue = results_queue
        self.episode_number = 0
        self.gradient_updates_queue = gradient_updates_queue

        # Init new bandit env at beginning of training (reset after ep)
        if self.train_config.env_name == "bernoulli-bandit":
            self.env = BernoulliMAB(num_arms=train_config.num_arms,
                              bandit_type=train_config.train_bandit_type,
                              max_steps=train_config.train_lifetime)
        else:
            raise ValueError("Provide valid env name or adjust for gym.")

    def run(self):
        """ Starts an A3C worker - each worker gets threads assigned """
        torch.set_num_threads(self.train_config.threads_per_worker)
        # Run over episodes for individual worker!
        for ep_ix in range(self.episodes_to_run):
            # At beginning of episode pull newest parameters from global net
            with self.optimizer_lock:
                copy_model_over(self.global_net, self.local_net)

            with self.global_ep_count.get_lock():
                # Get discount factor - anneal up (longer credit assignment)
                discount_factor = linearly_anneal(self.global_ep_count.value,
                                         self.train_config.min_discount_factor,
                                         self.train_config.max_discount_factor,
                                         int(self.train_config.anneal_discount_in*
                                             self.train_config.train_num_episodes))

                # Get loss weight coefficient for entropy term - anneal down
                e_beta = linearly_anneal(self.global_ep_count.value,
                                self.train_config.max_beta_entropy,
                                self.train_config.min_beta_entropy,
                                int(self.train_config.anneal_entropy_in*
                                    self.train_config.train_num_episodes))

            # Rollout a bandit/gridworld episode & unpack the results
            if self.train_config.env_name in ["bernoulli-bandit"]:
                ep_results = run_bandit_episode(self.env, self.local_net,
                            discount_factor, self.device,
                            use_bootstrap=self.train_config.use_bootstrap)

            log_probs, advantage, entropy, perf = ep_results
            # Compute A2C loss & return total, actor (PG), critic (MSBE), entropy
            total_l, actor_l, critic_l, entropy_l = a2c_loss(log_probs, advantage,
                                        entropy, entropy_beta=e_beta,
                                        value_beta=self.train_config.value_beta,
                                        return_all=True)

            self.put_gradients_in_queue(total_l, self.train_config.clip_grad_norm)
            self.episode_number += 1
            with self.global_step_count.get_lock():
                self.global_step_count.value += log_probs.size(0)

            with self.global_ep_count.get_lock():
                self.global_ep_count.value += 1
                if self.global_ep_count.value % self.train_config.log_every_episodes == 0:
                    clock_tick = [self.global_ep_count.value, self.global_step_count.value]
                    loss_stats = [total_l.item(), actor_l, critic_l, entropy_l]
                    if self.train_config.env_name in ["bernoulli-bandit"]:
                        avg_reward, avg_regret, avg_subopt_pulls = test_bandit_agent(self.local_net,
                                                                                     self.device,
                                                                                     self.train_config)
                        performance = [avg_reward, avg_regret[-1], avg_subopt_pulls[-1]]

                    stats_tick = loss_stats + performance
                    logging_data = [clock_tick, stats_tick]
                    self.results_queue.put(logging_data)

    def put_gradients_in_queue(self, total_loss, clip_grad_norm=None):
        """ Put gradients in queue for optim process to update shared model"""
        # Clean up gradients in the local network
        self.local_optimizer.zero_grad()
        total_loss.backward()
        if clip_grad_norm is not None:
            clip_grad_value_(self.local_net.parameters(), clip_grad_norm)
        gradients = [param.grad for param in self.local_net.parameters()]
        self.gradient_updates_queue.put(gradients)


class SharedAdam(torch.optim.Adam):
    """Creates an adam optimizer object that is shareable between processes. Useful for algorithms like A3C. Code
    taken from https://github.com/ikostrikov/pytorch-a3c/blob/master/my_optim.py"""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps,
                                         weight_decay=weight_decay,
                                         amsgrad=amsgrad)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                amsgrad = group['amsgrad']
                state = self.state[p]
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step'].item()
                bias_correction2 = 1 - beta2 ** state['step'].item()
                step_size = group['lr'] * math.sqrt(
                    bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)
        return loss
