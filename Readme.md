# 'Minimal' Actor-Critic Pipeline for Memory-Based Meta-Learning

Hi there! In this repo you can find a minimal pipeline for training an Actor-Critic LSTM-based agent on a Bernoulli Multi-Arm-Bandit task similar to RL^2 by Wang et al. (2016). It implements both A2C and A3C in PyTorch and aims to provide a minimal and extendable tool. We use multiprocessing and queues to process gradient streams from multiple actors.

# Structure of Repository

```
minimal-meta-rl
├── actor_critic: A2C/A3C workers, loss and Meta-LSTM architecture.
├── bandits: Bandit envs, episode rollout helpers.
├── utils: Helper functions (preparation, logging, annealing, etc.).
├── Readme.md: Documentation.
├── bernoulli_a2c.json: Training configuration file.
├── cluster_train.sh: qsub cluster job configuration file.
├── requirements.txt: Dependencies.
├── run_meta_a3c.py: Main training script for A2C.
├── .gitignore: Files to be excluded from version control.
```

# Installation & Execution

1. Create a clean virtual environment and install requirements.

```
conda create -n experiment-env python=3.6
pip install -r requirements.txt
```

2. Activate environment and execute main training routine.

```
conda activate experiment-env
python run_meta_a3c.py
```

3. Visualize the learning curves in a separate notebook.

```
jupyter notebook visualize_results.ipynb
```

4. **Extra**: Running the training on the cluster. Log into the cluster and execute the qsub bash script:

```
qsub cluster_train.sh
```
