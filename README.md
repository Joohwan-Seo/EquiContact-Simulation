# EquiContact-Simulation
A repo to test G-CompACT in the simulation environment. 

Written by Joohwan Seo, Ph.D. Candidate at Mechanical Engineering, UC Berkeley. 

## Installation
```
git clone --recurse-submodules git@github.com:Joohwan-Seo/EquiContact-Simulation.git
```

### Note
``recurse-submodules`` key is required to run the package

### Install Dependencies
The robot environment is built from the ``robosuite`` package, but with the modification to add "Indy7" Robot and "Geometric Impedance Control (GIC)". 
```
your_directory/equicontact-simulation/external/robosuite
pip install -e . 
```

```
your_directory/equicontact-simulation/external/robosuite_models
pip install -e .
```

Please refer to the dependencies issue with the original repos:

source of robosuite: https://github.com/ARISE-Initiative/robosuite \
source of robosuite-models: https://github.com/ARISE-Initiative/robosuite_models

### Install DETR
Follow the installation details of ACT: https://github.com/tonyzhaozh/act

```
cd act/detr
pip install -e . 
```

The original ACT uses python 3.8, but this repo was tested with python 3.10.16.

## Running the Code

### Overview
The imitation learning procedures are composed as 
- collect expert demonstration for stack block task
- train ACT with different setup
- evaluation of trained model

### Stack Block Environment
The objective of the ``stack`` environment is to pick up the small block and stack on the large block.
For our test to G-CompACT, we only test with the first stage, which is just to pick up the small block.

### Collection of Expert Demonstration
Run
```
python collect_data.py
```

Refer to config file provided in ``config/controller/indy7_absolute_pose.json`` and ``config/train/ACT_stack_08.yaml``

In the config file, you can define the types of the observation and actions, which will be also used in the evaluation.

This will collect 50 successful demos by default. This code run rule-based policy implemented in ``scripted_policy/policy_player_stack.py``.

### Train the ACT
Run
```
source train.sh
```
which will automatically run ``preprocess_data.py`` and ``act/train_act.py``, with the predefined config file.

**NOTE**  Make sure to use the config file that you want.
**NOTE** Of course, you can put the parameters related to the neural network to the config file, but you may need to change several lines in the ``detr/main.py``, etc. (This is due to original implementation of ACT.)

### Evalute the trained model
Run 
```
source eval.sh
```

Make sure to use the same parameters for the neural network in the ``train.sh`` and ``eval.sh``.

