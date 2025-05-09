# Enhancing End-to-End Autonomous Parking with Correction of Timed Automaton


This repository contains the code for the paper 
[Enhancing End-to-End Autonomous Parking with Correction of Timed Automaton](https://arxiv.org/abs/2504.10812).

This work is based on previous works [E2E Parking: Autonomous Parking by the End-to-end Neural Network on the CARLA Simulator](https://github.com/qintonguav/e2e-parking-carla) and [E2E Parking Dataset: An Open Benchmark for End-to-End Autonomous Parking](https://github.com/KejiaGao/e2e-parking-carla-dataset). By integrating rule-based correction with timed automaton into end-to-end Transformer model, the hybrid framework achieves a state-of-the-art performance. The closed-loop evaluation is conducted on CARLA simulator 0.9.11. Utilizing the same original model architecture, the model trained on E2E Parking dataset with timed automaton achieves an overall Target Success Rate (TSR) of 97.66%, which significantly exceeds the results reported by the original authors.


## Setup

Clone the repo, setup CARLA 0.9.11, and build the conda environment:

```Shell
git clone https://github.com/KejiaGao/e2e-parking-carla-timed-automaton.git
cd e2e-parking-carla-dataset/
conda env create -f environment.yml
conda activate E2EParking
chmod +x setup_carla.sh
./setup_carla.sh
```
CUDA 11.7 is used as default. The original authors also validate the compatibility of CUDA 10.2 and 11.3.

## Evaluation (Inference with pre-trained model)
For inference, the original authors prepare a [pre-trained model](https://drive.google.com/file/d/1XOlzBAb9W91R6WOB-srgdY8AZH3fXlML/view?usp=sharing). This pre-trained model has an overall success rate of around 75%. We have tried to train models on our dataset based on this pre-trained model and the Target Success Rate drops so we recommend training the model without the pre-trained model.


The first step is to launch a CARLA server:

```Shell
./carla/CarlaUE4.sh -opengl
```

In a separate terminal, use the script below for trained model evaluation:
```Shell
python3 carla_parking_eva.py
```

The main variables to set for this script:
```
--model_path        -> path to model.ckpt
--eva_epochs        -> number of eva epochs (default: 4')
--eva_task_nums     -> number of evaluation task (default: 16')
--eva_parking_nums  -> number of parking nums for every slot (default: 6')
--eva_result_path   -> path to save evaluation result csv file
--shuffle_veh       -> shuffle static vehicles between tasks (default: True)
--shuffle_weather   -> shuffle weather between tasks (default: False)
--random_seed       -> random seed to initialize env (default: 0)
```
When the evaluation is completed, metrics will be saved to csv files located at '--eva_result_path'.

## Dataset and Training
### E2E Parking Dataset
Please refer to [e2e-parking-carla-dataset](https://github.com/KejiaGao/e2e-parking-carla-dataset).

### Training Data Generation
The original authors have provided the tools for manual parking data generation, which is the way I create the dataset. 
The first step is to launch a CARLA server:

```Shell
./carla/CarlaUE4.sh -opengl
```

In a separate terminal, use the script below for generating training data:
```Shell
python3 carla_data_gen.py
```

The main variables to set for this script:
```
--save_path         -> path to save sensor data (default: ./e2e_parking/)
--task_num          -> number of parking task (default: 16)
--shuffle_veh       -> shuffle static vehicles between tasks (default: True)
--shuffle_weather   -> shuffle weather between tasks (default: False)
--random_seed       -> random seed to initialize env; if sets to 0, use current timestamp as seed (default: 0)
```

Keyboard Control designed by original authors (not recommended):
```
w/a/s/d:    throttle/left_steer/right_steer/hand_brake
space:      brake
q:          reverse gear
BackSpace:  reset the current task
TAB:        switch camera view
```

Functional mapping of XBOX controller (recommended):
```
LS, Left Stick:     steering
RT, Right Trigger:  throttle
LT, Left Trigger:   brake
LB, Left Bumper:    handbrake
Button A:           switch first / reverse gear:
Button Y:           reset the current task
RB, Right Bumper:   switch camera view
```

Conditions for successful parking:
```
position: vehicle center to slot center < 0.5 meter
orientation: rotation error < 0.5 degree
duration: satisfy the above two conditions for 60 frames
```
The target parking slot is marked with a red 'T'. 

The program automatically switches to the next task when the current one is completed.

Any collision will reset the task.

### Training script

The code for training is provided in [pl_train.py](./pl_train.py) \
Run the script in the terminal to start training:
```Shell
python pl_train.py 
```
To configure the training parameters, please refer to [training.yaml](./config/training.yaml), including training data path, number of epochs and checkpoint path.

To select the GPU(s) for training, modify the setting in [pl_train.py](./pl_train.py). The parameter ```num_gpus``` can be automatically detected and doesn't need changing anymore.

For instance, 4 GPU parallel training:
```
line 2: os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
```



## Bibtex
If this work is helpful for your research, please consider citing our paper with the following BibTeX information.

```
@misc{gao2025e2eparkingdatasetopen,
      title={E2E Parking Dataset: An Open Benchmark for End-to-End Autonomous Parking}, 
      author={Kejia Gao and Liguo Zhou and Mingjun Liu and Alois Knoll},
      year={2025},
      eprint={2504.10812},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2504.10812}, 
}
```
