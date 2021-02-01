# import optuna
# from optuna.pruners.base import BasePruner
# from optuna.pruners.nop import NopPruner
from ... import optuna
from .base import BasePruner
from .nop import NopPruner


def _filter_study(
    study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial"
) -> "optuna.study.Study":
    return study
