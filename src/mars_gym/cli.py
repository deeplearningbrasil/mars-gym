import os
from typing import Tuple

import click

import mars_gym

os.environ["LUIGI_CONFIG_PATH"] = os.path.join(os.path.split(mars_gym.__file__)[0], "luigi.cfg")


def _process_args(args: Tuple[str]) -> str:
    return " ".join(["'{}'".format(arg) for arg in args])


@click.group()
def cli():
    pass


@cli.command()
def viz():
    from mars_gym.tools.eval_viz import app

    os.system(f"streamlit run {app.__file__}")


@cli.group()
def run():
    pass


@run.command(context_settings=dict(ignore_unknown_options=True,), add_help_option=False)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def interaction(args: Tuple[str]):
    args_str = _process_args(args)
    os.system(
        f"PYTHONPATH=. luigi --module mars_gym.simulation.interaction InteractionTraining {args_str} --local-scheduler"
    )


@run.command(context_settings=dict(ignore_unknown_options=True,), add_help_option=False)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def supervised(args: Tuple[str]):
    args_str = _process_args(args)
    os.system(
        f"PYTHONPATH=. luigi --module mars_gym.simulation.training SupervisedModelTraining {args_str} --local-scheduler"
    )


@run.command(context_settings=dict(ignore_unknown_options=True,), add_help_option=False)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def data(args: Tuple[str]):
    args_str = _process_args(args)
    os.system(
        f"PYTHONPATH=. luigi --module {args_str} --local-scheduler"
    )

@cli.group()
def evaluate():
    pass


@evaluate.command(
    context_settings=dict(ignore_unknown_options=True,), add_help_option=False
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def interaction(args: Tuple[str]):
    args_str = _process_args(args)
    os.system(
        "PYTHONPATH=. luigi --module mars_gym.evaluation.task EvaluateTestSetPredictions "
        f"--model-task-class mars_gym.simulation.interaction.InteractionTraining {args_str} --local-scheduler"
    )


@evaluate.command(
    context_settings=dict(ignore_unknown_options=True,), add_help_option=False
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def supervised(args: Tuple[str]):
    args_str = _process_args(args)
    os.system(
        "PYTHONPATH=. luigi --module mars_gym.evaluation.task EvaluateTestSetPredictions "
        f"--model-task-class mars_gym.simulation.training.SupervisedModelTraining {args_str} --local-scheduler"
    )
