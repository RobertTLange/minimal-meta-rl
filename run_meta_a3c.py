import os
import time
import argparse
import math

import numpy as np
import torch
import torch.optim as optim
import torch.multiprocessing as mp

from utils.logger import DeepLogger
from utils.general import set_random_seeds, load_config
from actor_critic.meta_rnn import MetaLSTM
from actor_critic.a3c import (Actor_Critic_Worker,
                              a3c_update_shared_model,
                              a2c_update_shared_model,
                              SharedAdam)


def train(net_config, train_config, log_config):
    """ Run the meta-training loop. """
    # Set random seeds, device, parallel workers, logger for results
    set_random_seeds(seed_id=train_config.seed_id, verbose=True)
    device = torch.device(train_config.device_name)
    num_workers = train_config.num_workers
    train_log = DeepLogger(**log_config)

    # Define the network architecture, optimizer & the logger
    global_meta_net = MetaLSTM(**net_config).float().to(device)
    shared_optimizer = SharedAdam(global_meta_net.parameters(),
                                  lr=train_config.l_rate,
                                  weight_decay=train_config.l2_decay)

    # Share network parameters/optimizer state moving averages (momentum/var)
    global_meta_net.share_memory()
    shared_optimizer.share_memory()
    global_ep_count, global_step_count, results_queue = (mp.Value('i', 0),
                                                         mp.Value('d', 0.),
                                                         mp.Queue())
    gradient_updates_queue = mp.Queue()
    optimizer_lock = mp.Lock()

    # Differentiate optimization workers A2C (sync) vs. A3C (async) - start
    if train_config.sync_grads:
        optimizer_worker = mp.Process(target=a2c_update_shared_model,
                                      args=(global_meta_net,
                                            shared_optimizer,
                                            global_ep_count,
                                            gradient_updates_queue,
                                            optimizer_lock,
                                            num_workers,
                                            train_config.train_num_episodes))
    else:
        optimizer_worker = mp.Process(target=a3c_update_shared_model,
                                      args=(global_meta_net,
                                            shared_optimizer,
                                            global_ep_count,
                                            gradient_updates_queue,
                                            optimizer_lock,
                                            train_config.train_num_episodes))
    optimizer_worker.start()

    # Initialize desired number of parallel 'experience' workers
    processes = []
    episodes_per_process = int(train_config.train_num_episodes/num_workers) + 1

    for p_id in range(num_workers):
        local_meta_net = MetaLSTM(**net_config).float().to(device)
        worker = Actor_Critic_Worker(p_id, episodes_per_process,
                                     train_config,
                                     global_meta_net, local_meta_net,
                                     shared_optimizer, global_ep_count,
                                     global_step_count, optimizer_lock,
                                     results_queue, gradient_updates_queue, device)
        worker.start()
        processes.append(worker)

    # Start and collect results from workerts
    update_logger(global_net=global_meta_net,
                  train_log=train_log,
                  episode_number=global_ep_count,
                  results_queue=results_queue,
                  total_episodes=train_config.train_num_episodes)

    # Stop workers once done with generating results and clean up
    optimizer_worker.terminate()
    for worker in processes:
        worker.terminate()


def update_logger(global_net, train_log, episode_number, results_queue,
                  total_episodes):
    """ Worker which updates log with results as they come into queue. """
    start_t = time.time()
    while True:
        with episode_number.get_lock():
            carry_on = episode_number.value < total_episodes
        if carry_on:
            # Receive results from process and log results/network chkpt
            if not results_queue.empty():
                logging_stats = results_queue.get()
                train_log.update_log(logging_stats[0], logging_stats[1])
                train_log.save_log()
                train_log.save_network(global_net)
                start_t = time.time()
        else:
            break


if __name__ == "__main__":
    # Load and unpack training configs
    config_fname = "bernoulli_a2c.json"
    configs = load_config(config_fname)
    train_config, net_config, log_config = (configs.train_config,
                                            configs.net_config,
                                            configs.log_config)

    # Add individual filename of train config for logging
    log_config.config_fname = config_fname
    log_config.experiment_dir = "experiments/"
    log_config.fname_ext = ""
    log_config.tboard_fname = ""

    # Start the training run
    start_t = time.time()
    train(net_config, train_config, log_config)
    print("Total Training Time: ", time.time() - start_t)
