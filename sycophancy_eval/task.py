from inspect_ai import Epochs, Task, task

from sycophancy_eval.dataset import load_questions
from sycophancy_eval.scorers import sycophancy_scorer
from sycophancy_eval.solvers import two_turn_pressure


def build_task(pressure_type: str) -> Task:
    return Task(
        dataset=load_questions(),
        solver=two_turn_pressure(pressure_type),
        scorer=sycophancy_scorer(),
        # The default mean reducer would turn the outcome letters into floats
        # (C=1, I=0, N=0, the rest unconvertible), so keep them as they are.
        epochs=Epochs(1, "mode"),
    )


@task
def sycophancy_simple() -> Task:
    return build_task("simple")


@task
def sycophancy_authority() -> Task:
    return build_task("authority")


@task
def sycophancy_assertive() -> Task:
    return build_task("assertive")
