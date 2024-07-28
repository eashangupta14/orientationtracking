# Orientation Tracking and Paronama Generation

This is the code for ECE 276A project 1.

## Installation

Follow the steps for setting up the environment  required to run the code on Windows.

```bash
conda create -n <Env Name> python=3.8
conda activate <Env Name>
pip install -r requirements.txt
pip install "jax[cpu]===0.3.14" -f https://whls.blob.core.windows.net/unstable/index.html --use-deprecated legacy-resolver

```

## Folder Structure

This is the folder structure that need to be followed while running the code.

```bash
├───code
│   └───__pycache__
│   │
│   │─── main.py
│   │─── utils.py
│   
│───data
│   ├───testset
│   │   ├───cam
│   │   │   └───cam10.p
│   │   └───imu
│   │       └───imuRaw10.p
│   └───trainset
│       ├───cam
│       │   └───cam1.p
│       ├───imu
│       │   └───imuRaw1.p
│       └───vicon
│           └───viconRot1.p
```

## To Run

Follow the commands to run the program. There are 2 ways to run the code.

1. To use first 250 points for imu calibration.
2. For the user to select the range of points for imu calibration.(This method was used for results shown in report).

The results for both of them are similar.

Also use data_num and dataset flags to select the dataset number(1 or 2 or 3 ...) and dataset type(train or test).

### Using first 250 points for imu calibration

```bash
cd code
python main.py --data_num <dataset number> --dataset <train or test> 
```

For example to run dataset 10 in test set use:

```bash
cd code
python main.py --data_num 10 --dataset test 
```

### Letting User Select the Points

```bash
cd code
python main.py --data_num <dataset number> --dataset <train or test> --user
```

In this a window will open with imu values, select the two points on the graph between which you want the code to use values for IMU calibration.

## Results

Results will be shown in form of plots as the code progresses. Close the graphs to continue the code.
Also graphs will be saved in `plots` folder in the root directory.
