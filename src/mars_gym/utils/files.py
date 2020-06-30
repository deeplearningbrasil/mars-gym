import os
import json
from typing import Type


OUTPUT_PATH = os.environ["OUTPUT_PATH"] if "OUTPUT_PATH" in os.environ else "output"


def get_params_path(task_dir: str) -> str:
    return os.path.join(task_dir, "params.json")


def get_params(task_dir: str) -> dict:
    with open(get_params_path(task_dir), "r") as params_file:
        return json.load(params_file)


def get_weights_path(task_dir: str) -> str:
    return os.path.join(task_dir, "weights.pt")


def get_history_path(task_dir: str) -> str:
    return os.path.join(task_dir, "history.csv")


def get_tensorboard_logdir(task_id: str) -> str:
    return os.path.join(OUTPUT_PATH, "tensorboard_logs", task_id)


def get_task_dir(model_cls: Type, task_id: str):
    return os.path.join(OUTPUT_PATH, "models", model_cls.__name__, "results", task_id)


def get_interaction_dir(model_cls: Type, task_id: str):
    return os.path.join(
        OUTPUT_PATH, "interaction", model_cls.__name__, "results", task_id
    )


def get_simulator_datalog_path(task_dir: str) -> str:
    return os.path.join(task_dir, "sim-datalog.csv")


def get_interator_datalog_path(task_dir: str) -> str:
    return os.path.join(task_dir, "all-datalog.csv")


def get_ground_truth_datalog_path(task_dir: str) -> str:
    return os.path.join(task_dir, "gt-datalog.csv")


def get_test_set_predictions_path(task_dir: str) -> str:
    return os.path.join(task_dir, "test_set_predictions.csv")


def get_index_mapping_path(task_dir: str) -> str:
    return os.path.join(task_dir, "index_mapping.pkl")
