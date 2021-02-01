import collections
import datetime
import gc
import math
import threading
import warnings


import numpy as np
import pandas as pd  # NOQA


# from optuna._study_direction import StudyDirection
# from optuna import exceptions
# from optuna import logging
# from optuna import progress_bar as pbar_module
# from optuna import pruners
# from optuna import samplers
# from optuna import storages
# from optuna import trial as trial_module
# from optuna.trial import FrozenTrial
# from optuna.trial import TrialState

from ._study_direction import StudyDirection
from . import exceptions
from . import logging
from . import progress_bar as pbar_module
from . import pruners
from . import samplers
from . import storages
from . import trial as trial_module
from .trial import FrozenTrial
from .trial import TrialState


_logger = logging.get_logger(__name__)


class BaseStudy(object):
    def __init__(self, study_id, storage):
        # type: (int, storages.BaseStorage) -> None

        self._study_id = study_id
        self._storage = storage

    @property
    def best_params(self):
        # type: () -> Dict[str, Any]
        """Return parameters of the best trial in the study.

        Returns:
            A dictionary containing parameters of the best trial.
        """

        return self.best_trial.params

    @property
    def best_value(self):
        # type: () -> float
        """Return the best objective value in the study.

        Returns:
            A float representing the best objective value.
        """

        best_value = self.best_trial.value
        assert best_value is not None

        return best_value

    @property
    def best_trial(self):
        # type: () -> FrozenTrial
        """Return the best trial in the study.

        Returns:
            A :class:`~optuna.FrozenTrial` object of the best trial.
        """

        return self._storage.get_best_trial(self._study_id)

    @property
    def direction(self):
        # type: () -> StudyDirection
        """Return the direction of the study.

        Returns:
            A :class:`~optuna.study.StudyDirection` object.
        """

        return self._storage.get_study_direction(self._study_id)

    @property
    def trials(self):
        # type: () -> List[FrozenTrial]
        """Return all trials in the study.

        The returned trials are ordered by trial number.

        This is a short form of ``self.get_trials(deepcopy=True)``.

        Returns:
            A list of :class:`~optuna.FrozenTrial` objects.
        """

        return self.get_trials()

    def get_trials(self, deepcopy=True):
        # type: (bool) -> List[FrozenTrial]
        """Return all trials in the study.

        The returned trials are ordered by trial number.

        For library users, it's recommended to use more handy
        :attr:`~optuna.study.Study.trials` property to get the trials instead.

        Args:
            deepcopy:
                Flag to control whether to apply ``copy.deepcopy()`` to the trials.
                Note that if you set the flag to :obj:`False`, you shouldn't mutate
                any fields of the returned trial. Otherwise the internal state of
                the study may corrupt and unexpected behavior may happen.

        Returns:
            A list of :class:`~optuna.FrozenTrial` objects.
        """

        return self._storage.get_all_trials(self._study_id, deepcopy=deepcopy)

    @property
    def storage(self):
        # type: () -> storages.BaseStorage
        """Return the storage object used by the study.

        .. deprecated:: 0.15.0
            The direct use of storage is deprecated.
            Please access to storage via study's public methods
            (e.g., :meth:`~optuna.study.Study.set_user_attr`).

        Returns:
            A storage object.
        """

        warnings.warn(
            "The direct use of storage is deprecated. "
            "Please access to storage via study's public methods "
            "(e.g., `Study.set_user_attr`)",
            DeprecationWarning,
        )

        _logger.warning(
            "The direct use of storage is deprecated. "
            "Please access to storage via study's public methods "
            "(e.g., `Study.set_user_attr`)"
        )

        return self._storage


class Study(BaseStudy):
    """A study corresponds to an optimization task, i.e., a set of trials.

    This object provides interfaces to run a new :class:`~optuna.trial.Trial`, access trials'
    history, set/get user-defined attributes of the study itself.

    Note that the direct use of this constructor is not recommended.
    To create and load a study, please refer to the documentation of
    :func:`~optuna.study.create_study` and :func:`~optuna.study.load_study` respectively.

    """

    def __init__(
        self,
        study_name,  # type: str
        storage,  # type: Union[str, storages.BaseStorage]
        sampler=None,  # type: samplers.BaseSampler
        pruner=None,  # type: pruners.BasePruner
        seed=None,
        cat_preds=None,
    ):
        # type: (...) -> None
        self.add_on = 0

        self.study_name = study_name
        storage = storages.get_storage(storage)
        study_id = storage.get_study_id_from_name(study_name)
        super(Study, self).__init__(study_id, storage)

        # use TPE sampler
        self.sampler = sampler or samplers.TPESampler()
        # don't use prunning
        self.pruner = pruner or pruners.NopPruner()

        self._optimize_lock = threading.Lock()
        self._stop_flag = False

        self.evaled = set()
        self.rnd = np.random.RandomState(seed=seed)

        self.cat_preds = cat_preds
        try:
            self.cat_preds_set = set(cat_preds.values())
        except:
            pass

        self.info = {}
        self.info["names"] = []

    def __getstate__(self):
        # type: () -> Dict[Any, Any]

        state = self.__dict__.copy()
        del state["_optimize_lock"]
        return state

    def __setstate__(self, state):
        # type: (Dict[Any, Any]) -> None

        self.__dict__.update(state)
        self._optimize_lock = threading.Lock()

    @property
    def user_attrs(self):
        # type: () -> Dict[str, Any]
        """Return user attributes.

        Returns:
            A dictionary containing all user attributes.
        """

        return self._storage.get_study_user_attrs(self._study_id)

    @property
    def system_attrs(self):
        # type: () -> Dict[str, Any]
        """Return system attributes.

        Returns:
            A dictionary containing all system attributes.
        """

        return self._storage.get_study_system_attrs(self._study_id)

    def optimize(
        self,
        func,  # type: ObjectiveFuncType
        n_trials=None,  # type: Optional[int]
        timeout=None,  # type: Optional[float]
        n_jobs=1,  # type: int
        catch=(),  # type: Union[Tuple[()], Tuple[Type[Exception]]]
        callbacks=None,  # type: Optional[List[Callable[[Study, FrozenTrial], None]]]
        gc_after_trial=True,  # type: bool
        show_progress_bar=False,  # type: bool
        **kwargs,
    ):
        # type: (...) -> None
        """Optimize an objective function.

        Optimization is done by choosing a suitable set of hyperparameter values from a given
        range. Uses a sampler which implements the task of value suggestion based on a specified
        distribution. The sampler is specified in :func:`~optuna.study.create_study` and the
        default choice for the sampler is TPE.
        See also :class:`~optuna.samplers.TPESampler` for more details on 'TPE'.

        Args:
            func:
                A callable that implements objective function.
            n_trials:
                The number of trials. If this argument is set to :obj:`None`, there is no
                limitation on the number of trials. If :obj:`timeout` is also set to :obj:`None`,
                the study continues to create trials until it receives a termination signal such
                as Ctrl+C or SIGTERM.
            timeout:
                Stop study after the given number of second(s). If this argument is set to
                :obj:`None`, the study is executed without time limitation. If :obj:`n_trials` is
                also set to :obj:`None`, the study continues to create trials until it receives a
                termination signal such as Ctrl+C or SIGTERM.
            n_jobs:
                The number of parallel jobs. If this argument is set to :obj:`-1`, the number is
                set to CPU count.
            catch:
                A study continues to run even when a trial raises one of the exceptions specified
                in this argument. Default is an empty tuple, i.e. the study will stop for any
                exception except for :class:`~optuna.exceptions.TrialPruned`.
            callbacks:
                List of callback functions that are invoked at the end of each trial. Each function
                must accept two parameters with the following types in this order:
                :class:`~optuna.study.Study` and :class:`~optuna.FrozenTrial`.
            gc_after_trial:
                Flag to execute garbage collection at the end of each trial. By default, garbage
                collection is enabled, just in case. You can turn it off with this argument if
                memory is safely managed in your objective function.
            show_progress_bar:
                Flag to show progress bars or not. To disable progress bar, set this ``False``.
                Currently, progress bar is experimental feature and disabled
                when ``n_jobs`` :math:`\\ne 1`.
        """

        # self._progress_bar = pbar_module._ProgressBar(
        #     show_progress_bar and n_jobs == 1, n_trials, timeout
        # )

        self._stop_flag = False

        # optimize one iteration at a time
        self._optimize_sequential(func, n_trials, timeout, catch, callbacks, gc_after_trial, None, **kwargs)

        # self._progress_bar.close()
        # del self._progress_bar

    def set_user_attr(self, key, value):
        # type: (str, Any) -> None
        """Set a user attribute to the study.

        Args:
            key: A key string of the attribute.
            value: A value of the attribute. The value should be JSON serializable.

        """

        self._storage.set_study_user_attr(self._study_id, key, value)

    def set_system_attr(self, key, value):
        # type: (str, Any) -> None
        """Set a system attribute to the study.

        Note that Optuna internally uses this method to save system messages. Please use
        :func:`~optuna.study.Study.set_user_attr` to set users' attributes.

        Args:
            key: A key string of the attribute.
            value: A value of the attribute. The value should be JSON serializable.

        """

        self._storage.set_study_system_attr(self._study_id, key, value)

    def trials_dataframe(
        self,
        attrs=(
            "number",
            "value",
            "datetime_start",
            "datetime_complete",
            "duration",
            "params",
            "user_attrs",
            "system_attrs",
            "state",
        ),  # type: Tuple[str, ...]
        multi_index=False,  # type: bool
    ):
        # type: (...) -> pd.DataFrame
        """Export trials as a pandas DataFrame_.

        The DataFrame_ provides various features to analyze studies. It is also useful to draw a
        histogram of objective values and to export trials as a CSV file.
        If there are no trials, an empty DataFrame_ is returned.

        Example:

            .. testcode::

                import optuna
                import pandas

                def objective(trial):
                    x = trial.suggest_uniform('x', -1, 1)
                    return x ** 2

                study = optuna.create_study()
                study.optimize(objective, n_trials=3)

                # Create a dataframe from the study.
                df = study.trials_dataframe()
                assert isinstance(df, pandas.DataFrame)
                assert df.shape[0] == 3  # n_trials.

        Args:
            attrs:
                Specifies field names of :class:`~optuna.FrozenTrial` to include them to a
                DataFrame of trials.
            multi_index:
                Specifies whether the returned DataFrame_ employs MultiIndex_ or not. Columns that
                are hierarchical by nature such as ``(params, x)`` will be flattened to
                ``params_x`` when set to :obj:`False`.

        Returns:
            A pandas DataFrame_ of trials in the :class:`~optuna.study.Study`.

        .. _DataFrame: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
        .. _MultiIndex: https://pandas.pydata.org/pandas-docs/stable/advanced.html
        """

        trials = self.get_trials(deepcopy=False)

        # If no trials, return an empty dataframe.
        if not len(trials):
            return pd.DataFrame()

        assert all(isinstance(trial, FrozenTrial) for trial in trials)
        attrs_to_df_columns = collections.OrderedDict()  # type: Dict[str, str]
        for attr in attrs:
            if attr.startswith("_"):
                # Python conventional underscores are omitted in the dataframe.
                df_column = attr[1:]
            else:
                df_column = attr
            attrs_to_df_columns[attr] = df_column

        # column_agg is an aggregator of column names.
        # Keys of column agg are attributes of `FrozenTrial` such as 'trial_id' and 'params'.
        # Values are dataframe columns such as ('trial_id', '') and ('params', 'n_layers').
        column_agg = collections.defaultdict(set)  # type: Dict[str, Set]
        non_nested_attr = ""

        def _create_record_and_aggregate_column(trial):
            # type: (FrozenTrial) -> Dict[Tuple[str, str], Any]

            record = {}
            for attr, df_column in attrs_to_df_columns.items():
                value = getattr(trial, attr)
                if isinstance(value, TrialState):
                    # Convert TrialState to str and remove the common prefix.
                    value = str(value).split(".")[-1]
                if isinstance(value, dict):
                    for nested_attr, nested_value in value.items():
                        record[(df_column, nested_attr)] = nested_value
                        column_agg[attr].add((df_column, nested_attr))
                else:
                    record[(df_column, non_nested_attr)] = value
                    column_agg[attr].add((df_column, non_nested_attr))
            return record

        records = list([_create_record_and_aggregate_column(trial) for trial in trials])

        columns = sum(
            (sorted(column_agg[k]) for k in attrs if k in column_agg), []
        )  # type: List[Tuple[str, str]]

        df = pd.DataFrame(records, columns=pd.MultiIndex.from_tuples(columns))

        if not multi_index:
            # Flatten the `MultiIndex` columns where names are concatenated with underscores.
            # Filtering is required to omit non-nested columns avoiding unwanted trailing
            # underscores.
            df.columns = [
                "_".join(filter(lambda c: c, map(lambda c: str(c), col))) for col in columns
            ]

        return df

    def _optimize_sequential(
        self,
        func,  # type: ObjectiveFuncType
        n_trials,  # type: Optional[int]
        timeout,  # type: Optional[float]
        catch,  # type: Union[Tuple[()], Tuple[Type[Exception]]]
        callbacks,  # type: Optional[List[Callable[[Study, FrozenTrial], None]]]
        gc_after_trial,  # type: bool
        time_start,  # type: Optional[datetime.datetime]
        **kwargs,
    ):
        # type: (...) -> None

        # trial counter
        i_trial = 0

        # timer
        if time_start is None:
            time_start = datetime.datetime.now()

        while True:
            if self._stop_flag:
                break

            # check number of trials
            if n_trials is not None:
                if i_trial >= n_trials:
                    break
                i_trial += 1

            # check if alloted time has expired
            if timeout is not None:
                elapsed_seconds = (datetime.datetime.now() - time_start).total_seconds()
                if elapsed_seconds - self.add_on >= timeout:
                    break

            self.info["names"] = []
            self._run_trial(func, catch, gc_after_trial, **kwargs)

            # self._progress_bar.update((datetime.datetime.now() - time_start).total_seconds())

        self._storage.remove_session()

    def _run_trial(
        self,
        func,  # type: ObjectiveFuncType
        catch,  # type: Union[Tuple[()], Tuple[Type[Exception]]]
        gc_after_trial,  # type: bool
        **kwargs,
    ):
        # type: (...) -> trial_module.Trial

        # trial_id enumerates the trials 0, 1, 2, ...
        trial_id = self._storage.create_new_trial(self._study_id)
        # create a new trial for this study (in file _trial.py)
        trial = trial_module.Trial(self, trial_id)
        # trial number is 0, 1, 2, ...
        trial_number = trial.number

        # evaluate the objective function
        result = func(trial, **kwargs)

        # The following line mitigates memory problems that can be occurred in some
        # environments (e.g., services that use computing containers such as CircleCI).
        if gc_after_trial:
            gc.collect()

        # return a float or TrialState.FAIL
        try:
            result = float(result)
        except (
            ValueError,
            TypeError,
        ):
            message = (
                "Setting status of trial#{} as {} because the returned value from the "
                "objective function cannot be casted to float. Returned value is: "
                "{}".format(trial_number, TrialState.FAIL, repr(result))
            )
            _logger.warning(message)
            self._storage.set_trial_system_attr(trial_id, "fail_reason", message)
            self._storage.set_trial_state(trial_id, TrialState.FAIL)
            return trial

        if math.isnan(result):
            message = (
                "Setting status of trial#{} as {} because the objective function "
                "returned {}.".format(trial_number, TrialState.FAIL, result)
            )
            _logger.warning(message)
            self._storage.set_trial_system_attr(trial_id, "fail_reason", message)
            self._storage.set_trial_state(trial_id, TrialState.FAIL)
            return trial

        # log results
        self._storage.set_trial_value(trial_id, result)
        self._storage.set_trial_state(trial_id, TrialState.COMPLETE)
        # self._log_completed_trial(trial, result)

        return trial

    def _log_completed_trial(self, trial, result):
        # type: (trial_module.Trial, float) -> None

        _logger.info(
            "Finished trial#{} with value: {} with parameters: {}. "
            "Best is trial#{} with value: {}.".format(
                trial.number, result, trial.params, self.best_trial.number, self.best_value
            )
        )


def create_study(
    storage=None,  # type: Union[None, str, storages.BaseStorage]
    sampler=None,  # type: samplers.BaseSampler
    pruner=None,  # type: pruners.BasePruner
    direction="minimize",  # type: str
    load_if_exists=False,  # type: bool
    seed=None,
    cat_preds=None,
):
    # type: (...) -> Study
    """Create a new :class:`~optuna.study.Study`.

    Args:
        storage:
            Database URL. If this argument is set to None, in-memory storage is used, and the
            :class:`~optuna.study.Study` will not be persistent.

            .. note::
                When a database URL is passed, Optuna internally uses `SQLAlchemy`_ to handle
                the database. Please refer to `SQLAlchemy's document`_ for further details.
                If you want to specify non-default options to `SQLAlchemy Engine`_, you can
                instantiate :class:`~optuna.storages.RDBStorage` with your desired options and
                pass it to the ``storage`` argument instead of a URL.

             .. _SQLAlchemy: https://www.sqlalchemy.org/
             .. _SQLAlchemy's document:
                 https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls
             .. _SQLAlchemy Engine: https://docs.sqlalchemy.org/en/latest/core/engines.html

        sampler:
            A sampler object that implements background algorithm for value suggestion.
            If :obj:`None` is specified, :class:`~optuna.samplers.TPESampler` is used
            as the default. See also :class:`~optuna.samplers`.
        pruner:
            A pruner object that decides early stopping of unpromising trials. See also
            :class:`~optuna.pruners`.
        direction:
            Direction of optimization. Set ``minimize`` for minimization and ``maximize`` for
            maximization.
        load_if_exists:
            Flag to control the behavior to handle a conflict of study names.
            In the case where a study named ``study_name`` already exists in the ``storage``,
            a :class:`~optuna.exceptions.DuplicatedStudyError` is raised if ``load_if_exists`` is
            set to :obj:`False`.
            Otherwise, the creation of the study is skipped, and the existing one is returned.

    Returns:
        A :class:`~optuna.study.Study` object.

    """

    # in memory or dbms (we will use only in memory data?)
    storage = storages.get_storage(storage)

    # study_id in our case is always 0, method in in_memory.py
    study_id = storage.create_new_study(None)

    # random string starting with "no-name"
    study_name = storage.get_study_name_from_id(study_id)

    # study seesion
    study = Study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        seed=seed,
        cat_preds=cat_preds,
    )

    if direction == "minimize":
        _direction = StudyDirection.MINIMIZE
    elif direction == "maximize":
        _direction = StudyDirection.MAXIMIZE
    else:
        raise ValueError("Please set either 'minimize' or 'maximize' to direction.")

    # set the study direction to be minimize or maximize
    study._storage.set_study_direction(study_id, _direction)

    return study
