# Notes

This code consists of two separate implementations: one discrete, one continuous. It accompanies the arXiv paper "Fine-tuning Behavioral Cloning Policies with Preference-Based Reinforcement Learning" [link](https://arxiv.org/abs/2509.26605).

## Installation
We recommend using the `uv` package manager [link to docs](https://docs.astral.sh/uv/getting-started/installation/), alternatively we recommend `conda`. Get `uv` via

Macos/Unix: 
`$ curl -LsSf https://astral.sh/uv/install.sh | sh`
or
`$ pip install uv`

Windows:

`$ powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

or

`$ pip install uv`


Then, create a virtual environment and install required packages with:

`$ uv venv bridge_venv --python 3.12`

This will create a folder in this directory called `bridge_venv/`. Activate the venv (every time you re-open the terminal)

`$ source bridge_venv/bin/activate`

Install requirements:

`$ uv pip install --no-cache-dir -r requirements.txt`

Verify code runs:

`$ python -m main_tabular -cn starmdp_debug`

`$ python -m main_continuous -cn reacher_debug`


## Running experiments
You can re-run paper experiments, run your own experiments, or re-plot experiments from the metrics we've pre-saved. 

**Main experiments**:

`main_tabular.py` runs discrete experiments, `main_continuous.py` runs continuous ones. These create the regret and |Pi_t| plots seen in Figure 2 in the paper.

To run an experiment:
First decide on the environment. We provide 2 discrete (`StarMDP`, `Gridworld`), 2 continuous MuJoCo (`Reacher-v5`, `Ant-v5`).

If you want, alter the config files in `configs/special/`. For each environment, there's 4 config files, one for each type of run, named `[ENV]_[TYPE].yaml`. The types are:
- `_paper`: re-runs paper run. results saved to `exps/[env]/paper_run_RERUN/`. (exact reproduction unlikely, see disclaimer below)
- `_replot`: replots paper run from saved metrics. results saved to `exps/[env]/paper_run`.
- `_quick`: runs an experiment with less seeds that should still be representative. results saved to `exps/[env]/[run_ID]/`. default `run_ID` is a 3-digit number counting up, you can provide a custom name by adding `--run_ID [your_id]` to the CLI call.
- `_debug`: to verify the code works: runs with low seeds, iterations, computational complexity. saves to `exps/[env]/debug/`.

Then run the chosen experiment via:
`$ python -m [main file, tabular or continuous] -cn [config name without the .yaml] [any config entries you want to change with CLI args]`

Example for tabular (starMDP):
`$ python -m main_tabular -cn starmdp_paper`

Example for MuJoCo (Reacher-v5):
`$ python -m main_continuous -cn reacher_quick --run_ID FooBarRun --embedding_name reacher_perf`

We also give you the CLI calls in .txt files in the paper experiment results folders, in `exps/[env]/paper_runs/paste_for_rerunning.txt` and `.../paste_for_replotting.txt`.

**Ablations**: You can run the ablations we ran using these scripts. It'll reproduce the data, we did selectively omit some of the runs to have a cleaner plot (less lines) so it may look a bit different.

Radius ablation:
`$ python -m continuous_radius_ablation -cn ablations/reacher_radius_ablation`

Number of offline traj's ablation: 
`$ python -m continuous_N_offline_trajs_ablation -cn ablations/reacher_noffline_ablation`

Different embeddings: this is for `Reacher-v5`. Test 3 different embeddings, with the names `avg_sa`, `reacher_perf`, `last_s`, via `--embedding_name [...]`:
`$ python -m main_continuous -cn ablations/reacher_embedding_comparison --embedding_name [embedding name] --run_ID [name you want, e.g. run_avg_sa]`

## Disclaimers:
### General
**Note 1**: Paper runs were done on many seeds. Even with multiprocessing, this is challenging on laptops. Discrete experiments are reasonably quick even on laptops, but the MuJoCo ones can take hours and we strongly recommend running them on a cluster. We've attached a SLURM script that starts a SLURM run (results save to same directories as described above), call it with 
`$ sbatch run_continuous_slurm.sh -cn [config name without the .yaml] [any config entries you want to change with CLI args]`

**Note 2**: Although we do extensively use seed setting and run everything on CPU, we noticed that on different machines, results could vary, e.g. between an M1 Macbook, an AMD CPU server, and an Intel CPU server. Due to this and some refactoring since running the paper experiments, exact reproduction of the paper plots may not be possible.

## Discrete/Continuous-specific disclaimers:

### Discrete
This version is meant to represent the theoretical version of BRIDGE as accurately as possible. It allows you to calculate all constants according to their respective formulas - the algorithm does run better in practice if some of them are adjusted to the specific setup you're running. See the configs for examples. 

This code started as an adaption from https://openreview.net/forum?id=2pJpFtdVNe which we heavily modified and expanded.

### Continuous
The first time you run a specific setup, or if you change the embedding function, number of candidate policies, anything about the behavior cloning, the number of offline trajectories, or anything that will alter the set of candidate policies, it'll need to precompute each policy's embedding. This takes a while! 

If you do want to train your own experts: we used `mps` and `cpu` training, we did not test `cuda` training. If you run into errors, ctrl+F in `experiment_runner_continuous.py` for `cuda` and comment things out.


