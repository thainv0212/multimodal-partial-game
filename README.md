# This repository contains source code and data for paper "An Effective Multimodal Approach for Blind Reinforcement Agents in Partially Observable Games"

## Folder structure:
- multimodal training: containing source code to train the proposed multimodal encoders.
- rl training: containing source code to train reinforcement learning.

## Installation
- For gmc training and policy training
```bash
    cd multimodal/gmc_code/rl
    conda env create -f env.yml
    conda activate GMC
```
- For comm training
```bash
    cd multimodal/comm
    conda env create -f env.yml
    conda
```
## Training gmc
```bash
    cd multimodal/gmc_code/rl
    python main_rl.py -f with experiment.stage="train_model" experiment.scenario=scenario experiment.model=gmcnew2
```
## Training comm
```bash
    cd multimodal/comm
    HYDRA_FULL_ERROR=1 python main_defend_the_center.py \
      +data=defend_the_center \
      +data.data_module.biased=false \
      +model="comm_weighted" \
      model.model.encoder.embed_dim=512 \
      mode="train" \
      trainer.max_epochs=100
```

## Training reinforcement learning agent
```bash
    cd rl
    python -m sf_examples.vizdoom.train_vizdoom --env=vizdoom20 --train_for_env_steps=500000000 --algo=APPO --env_frameskip=4 --use_rnn=True --wide_aspect_ratio=False --num_workers=36 --num_envs_per_worker=8 --decorrelate_envs_on_one_worker=False --train_dir=train_dir/my_way_home --encoder_type=gmc16 --gmc_model_file=path_to_model --use_sound
```