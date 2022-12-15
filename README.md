# Neural networks course project

# Age Estimator

Neural networks course project which focuses on implementing a system for estimating a person's age based on their
portrait.

## Project Organization

    ├── datasets                <- Folder that contains all raw datasets
    │   ├── UTKFace             <- All images
    │   ├── UTKFace_grouped     <- Images grouped by age
    ├── docs                    <- Project documentation
    ├── gui                     <- Folder that contains gui
    ├── models                  <- Models used in this project
    ├── performance             <- Folder that contains summary of model performance
    ├── states                  <- Folder that contains model states
    ├── util                    <- File that contains utility methods
    ├── README.md               <- Top level README.md for developers using this project
    ├── requirements.txt        <- The requirements file for reproducing the environment, e.g.
    │                               generated with `pip freeze > requirements.txt`
    ├── train.py                <- Training interface
    ├── predict.py              <- Predicting interface
    ├── main.py                 <- Entry point of the project

## Getting started

Create venv:

```bash
python3 -m venv venv
```

Creating venv is required only when running for the first time.

Activate venv:

```bash
source venv/bin/activate
```

Install requirements:

```bash
python3 -m pip install -r requirements.txt
```

## Running locally

+ Interactive GUI experience run `main.py` file.
    + GUI uses trained model which can be configured. The list of trained models ready to be used "out of the box" is
      located in the `states` directory.
+ Training:
    + for training existing models with different hyperparameters run modified `train.py` file.
    + For training new models first create a custom model file in the `models` directory.
+ Predicting:
    + In `predict.py` file load trained model and dataset.