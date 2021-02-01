import datetime
import warnings

import numpy as np
from operator import itemgetter
from itertools import chain, combinations, product

# from optuna import distributions
# from optuna.distributions import CategoricalDistribution
# from optuna.distributions import DiscreteUniformDistribution
# from optuna.distributions import IntLogUniformDistribution
# from optuna.distributions import IntUniformDistribution
# from optuna.distributions import LogUniformDistribution
# from optuna.distributions import UniformDistribution
# from optuna import logging
# from optuna import pruners
# from optuna.trial._base import BaseTrial
# from optuna.trial._util import _adjust_discrete_uniform_high
from .. import distributions
from ..distributions import CategoricalDistribution
from ..distributions import DiscreteUniformDistribution
from ..distributions import IntLogUniformDistribution
from ..distributions import IntUniformDistribution
from ..distributions import LogUniformDistribution
from ..distributions import UniformDistribution
from .. import logging
from .. import pruners
from ._base import BaseTrial
from ._util import _adjust_discrete_uniform_high


class Trial(BaseTrial):
    """A trial is a process of evaluating an objective function.

    This object is passed to an objective function and provides interfaces to get parameter
    suggestion, manage the trial's state, and set/get user-defined attributes of the trial.

    Note that the direct use of this constructor is not recommended.
    This object is seamlessly instantiated and passed to the objective function behind
    the :func:`optuna.study.Study.optimize()` method; hence library users do not care about
    instantiation of this object.

    Args:
        study:
            A :class:`~optuna.study.Study` object.
        trial_id:
            A trial ID that is automatically generated.

    """

    def __init__(
        self,
        study,  # type: Study
        trial_id,  # type: int
    ):
        # type: (...) -> None

        self.study = study
        self._trial_id = trial_id

        # TODO(Yanase): Remove _study_id attribute, and use study._study_id instead.
        self._study_id = self.study._study_id
        self.storage = self.study._storage
        self.logger = logging.get_logger(__name__)

        self._init_relative_params()

    def _init_relative_params(self):
        # type: () -> None

        # get the current trial object
        trial = self.storage.get_trial(self._trial_id)

        # does something if hyperband pruner, else returns study unchanged
        study = pruners._filter_study(self.study, trial)

        # returns {} for TPE
        self.relative_search_space = self.study.sampler.infer_relative_search_space(study, trial)
        # returns {}
        self.relative_params = self.study.sampler.sample_relative(
            study, trial, self.relative_search_space
        )

    def suggest_float(self, name, low, high, *, log=False, step=None):
        # type: (str, float, float, bool, Optional[float]) -> float
        """Suggest a value for the floating point parameter.

        Note that this is a wrapper method for :func:`~optuna.trial.Trial.suggest_uniform`,
        :func:`~optuna.trial.Trial.suggest_loguniform` and
        :func:`~optuna.trial.Trial.suggest_discrete_uniform`.

        .. versionadded:: 1.3.0

        .. seealso::
            Please see also :func:`~optuna.trial.Trial.suggest_uniform`,
            :func:`~optuna.trial.Trial.suggest_loguniform` and
            :func:`~optuna.trial.Trial.suggest_discrete_uniform`.

        Example:

            Suggest a momentum, learning rate and scaling factor of learning rate
            for neural network training.

            .. testsetup::

                import numpy as np
                import optuna
                from sklearn.model_selection import train_test_split
                from sklearn.neural_network import MLPClassifier

                np.random.seed(seed=0)
                X = np.random.randn(200).reshape(-1, 1)
                y = np.random.randint(0, 2, 200)
                X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)


            .. testcode::

                def objective(trial):
                    momentum = trial.suggest_float('momentum', 0.0, 1.0)
                    learning_rate_init = trial.suggest_float('learning_rate_init',
                                                             1e-5, 1e-3, log=True)
                    power_t = trial.suggest_float('power_t', 0.2, 0.8, step=0.1)
                    clf = MLPClassifier(hidden_layer_sizes=(100, 50), momentum=momentum,
                                        learning_rate_init=learning_rate_init,
                                        solver='sgd', random_state=0, power_t=power_t)
                    clf.fit(X_train, y_train)

                    return clf.score(X_valid, y_valid)

                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=3)

        Args:
            name:
                A parameter name.
            low:
                Lower endpoint of the range of suggested values. ``low`` is included in the range.
            high:
                Upper endpoint of the range of suggested values. ``high`` is excluded from the
                range.
            log:
                A flag to sample the value from the log domain or not.
                If ``log`` is true, the value is sampled from the range in the log domain.
                Otherwise, the value is sampled from the range in the linear domain.
                See also :func:`suggest_uniform` and :func:`suggest_loguniform`.
            step:
                A step of discretization.

        Returns:
            A suggested float value.
        """

        if step is not None:
            if log:
                raise NotImplementedError(
                    "The parameter `step` is not supported when `log` is True."
                )
            else:
                return self.suggest_discrete_uniform(name, low, high, step)
        else:
            if log:
                return self.suggest_loguniform(name, low, high)
            else:
                return self.suggest_uniform(name, low, high)

    def suggest_uniform(self, name, low, high):
        # type: (str, float, float) -> float
        """Suggest a value for the continuous parameter.

        The value is sampled from the range :math:`[\\mathsf{low}, \\mathsf{high})`
        in the linear domain. When :math:`\\mathsf{low} = \\mathsf{high}`, the value of
        :math:`\\mathsf{low}` will be returned.

        Example:

            Suggest a momentum for neural network training.

            .. testsetup::

                import numpy as np
                from sklearn.model_selection import train_test_split

                np.random.seed(seed=0)
                X = np.random.randn(200).reshape(-1, 1)
                y = np.random.randint(0, 2, 200)
                X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)

            .. testcode::

                import optuna
                from sklearn.neural_network import MLPClassifier

                def objective(trial):
                    momentum = trial.suggest_uniform('momentum', 0.0, 1.0)
                    clf = MLPClassifier(hidden_layer_sizes=(100, 50), momentum=momentum,
                                        solver='sgd', random_state=0)
                    clf.fit(X_train, y_train)

                    return clf.score(X_valid, y_valid)

                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=3)

        Args:
            name:
                A parameter name.
            low:
                Lower endpoint of the range of suggested values. ``low`` is included in the range.
            high:
                Upper endpoint of the range of suggested values. ``high`` is excluded from the
                range.

        Returns:
            A suggested float value.
        """

        distribution = UniformDistribution(low=low, high=high)

        self._check_distribution(name, distribution)

        if low == high:
            return self._set_new_param_or_get_existing(name, low, distribution)

        return self._suggest(name, distribution)

    def suggest_loguniform(self, name, low, high):
        # type: (str, float, float) -> float
        """Suggest a value for the continuous parameter.

        The value is sampled from the range :math:`[\\mathsf{low}, \\mathsf{high})`
        in the log domain. When :math:`\\mathsf{low} = \\mathsf{high}`, the value of
        :math:`\\mathsf{low}` will be returned.

        Example:

            Suggest penalty parameter ``C`` of `SVC <https://scikit-learn.org/stable/modules/
            generated/sklearn.svm.SVC.html>`_.

            .. testsetup::

                import numpy as np
                from sklearn.model_selection import train_test_split

                np.random.seed(seed=0)
                X = np.random.randn(50).reshape(-1, 1)
                y = np.random.randint(0, 2, 50)
                X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)

            .. testcode::

                import optuna
                from sklearn.svm import SVC

                def objective(trial):
                    c = trial.suggest_loguniform('c', 1e-5, 1e2)
                    clf = SVC(C=c, gamma='scale', random_state=0)
                    clf.fit(X_train, y_train)
                    return clf.score(X_valid, y_valid)

                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=3)

        Args:
            name:
                A parameter name.
            low:
                Lower endpoint of the range of suggested values. ``low`` is included in the range.
            high:
                Upper endpoint of the range of suggested values. ``high`` is excluded from the
                range.

        Returns:
            A suggested float value.
        """

        distribution = LogUniformDistribution(low=low, high=high)

        self._check_distribution(name, distribution)

        if low == high:
            return self._set_new_param_or_get_existing(name, low, distribution)

        return self._suggest(name, distribution)

    def suggest_discrete_uniform(self, name, low, high, q):
        # type: (str, float, float, float) -> float
        """Suggest a value for the discrete parameter.

        The value is sampled from the range :math:`[\\mathsf{low}, \\mathsf{high}]`,
        and the step of discretization is :math:`q`. More specifically,
        this method returns one of the values in the sequence
        :math:`\\mathsf{low}, \\mathsf{low} + q, \\mathsf{low} + 2 q, \\dots,
        \\mathsf{low} + k q \\le \\mathsf{high}`,
        where :math:`k` denotes an integer. Note that :math:`high` may be changed due to round-off
        errors if :math:`q` is not an integer. Please check warning messages to find the changed
        values.

        Example:

            Suggest a fraction of samples used for fitting the individual learners of
            `GradientBoostingClassifier <https://scikit-learn.org/stable/modules/generated/
            sklearn.ensemble.GradientBoostingClassifier.html>`_.

            .. testsetup::

                import numpy as np
                from sklearn.model_selection import train_test_split

                np.random.seed(seed=0)
                X = np.random.randn(50).reshape(-1, 1)
                y = np.random.randint(0, 2, 50)
                X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)

            .. testcode::

                import optuna
                from sklearn.ensemble import GradientBoostingClassifier

                def objective(trial):
                    subsample = trial.suggest_discrete_uniform('subsample', 0.1, 1.0, 0.1)
                    clf = GradientBoostingClassifier(subsample=subsample, random_state=0)
                    clf.fit(X_train, y_train)
                    return clf.score(X_valid, y_valid)

                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=3)

        Args:
            name:
                A parameter name.
            low:
                Lower endpoint of the range of suggested values. ``low`` is included in the range.
            high:
                Upper endpoint of the range of suggested values. ``high`` is included in the range.
            q:
                A step of discretization.

        Returns:
            A suggested float value.
        """

        high = _adjust_discrete_uniform_high(name, low, high, q)
        distribution = DiscreteUniformDistribution(low=low, high=high, q=q)

        self._check_distribution(name, distribution)

        if low == high:
            return self._set_new_param_or_get_existing(name, low, distribution)

        return self._suggest(name, distribution)

    def suggest_int(self, name, low, high, step=1, log=False):
        # type: (str, int, int, int, bool) -> int
        """Suggest a value for the integer parameter.

        The value is sampled from the integers in :math:`[\\mathsf{low}, \\mathsf{high}]`.

        Example:

            Suggest the number of trees in `RandomForestClassifier <https://scikit-learn.org/
            stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_.

            .. testsetup::

                import numpy as np
                from sklearn.model_selection import train_test_split

                np.random.seed(seed=0)
                X = np.random.randn(50).reshape(-1, 1)
                y = np.random.randint(0, 2, 50)
                X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)

            .. testcode::

                import optuna
                from sklearn.ensemble import RandomForestClassifier

                def objective(trial):
                    n_estimators = trial.suggest_int('n_estimators', 50, 400)
                    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
                    clf.fit(X_train, y_train)
                    return clf.score(X_valid, y_valid)

                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=3)


        Args:
            name:
                A parameter name.
            low:
                Lower endpoint of the range of suggested values. ``low`` is included in the range.
            high:
                Upper endpoint of the range of suggested values. ``high`` is included in the range.
            log:
                A flag to sample the value from the log domain or not.
                If ``log`` is true, at first, the range of suggested values is divided into grid
                points of width ``step``. The range of suggested values is then converted to a log
                domain, from which a value is uniformly sampled. The uniformly sampled value is
                re-converted to the original domain and rounded to the nearest grid point that we
                just split, and the suggested value is determined.
                For example,
                if `low = 2`, `high = 8` and `step = 2`,
                then the range of suggested values is divided by ``step`` as `[2, 4, 6, 8]`
                and lower values tend to be more sampled than higher values.
        """

        # create the IntUniformDistribution
        distribution = IntUniformDistribution(
            low=low, high=high, step=step
        )  # type: Union[IntUniformDistribution, IntLogUniformDistribution]

        if log:
            high = (
                distribution.high - distribution.low
            ) // distribution.step * distribution.step + distribution.low
            distribution = IntLogUniformDistribution(low=low, high=high, step=step)

        self._check_distribution(name, distribution)

        if low == high:
            return self._set_new_param_or_get_existing(name, low, distribution)

        param_value, name, samples, scores, distribution = self._suggest(name, distribution)
        return int(param_value), name, samples, scores, distribution

    def suggest_categorical(self, name, choices):
        # type: (str, Sequence[CategoricalChoiceType]) -> CategoricalChoiceType
        """Suggest a value for the categorical parameter.

        The value is sampled from ``choices``.

        Example:

            Suggest a kernel function of `SVC <https://scikit-learn.org/stable/modules/generated/
            sklearn.svm.SVC.html>`_.

            .. testsetup::

                import numpy as np
                from sklearn.model_selection import train_test_split

                np.random.seed(seed=0)
                X = np.random.randn(50).reshape(-1, 1)
                y = np.random.randint(0, 2, 50)
                X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)

            .. testcode::

                import optuna
                from sklearn.svm import SVC

                def objective(trial):
                    kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf'])
                    clf = SVC(kernel=kernel, gamma='scale', random_state=0)
                    clf.fit(X_train, y_train)
                    return clf.score(X_valid, y_valid)

                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=3)


        Args:
            name:
                A parameter name.
            choices:
                Parameter value candidates.

        .. seealso::
            :class:`~optuna.distributions.CategoricalDistribution`.

        Returns:
            A suggested value.
        """

        # categorical values
        choices = tuple(choices)

        # There is no need to call self._check_distribution because
        # CategoricalDistribution does not support dynamic value space.

        return self._suggest(name, CategoricalDistribution(choices=choices))

    def report(self, value, step):
        # type: (float, int) -> None
        """Report an objective function value for a given step.

        The reported values are used by the pruners to determine whether this trial should be
        pruned.

        .. seealso::
            Please refer to :class:`~optuna.pruners.BasePruner`.

        .. note::
            The reported value is converted to ``float`` type by applying ``float()``
            function internally. Thus, it accepts all float-like types (e.g., ``numpy.float32``).
            If the conversion fails, a ``TypeError`` is raised.

        Example:

            Report intermediate scores of `SGDClassifier <https://scikit-learn.org/stable/modules/
            generated/sklearn.linear_model.SGDClassifier.html>`_ training.

            .. testsetup::

                import numpy as np
                from sklearn.model_selection import train_test_split

                np.random.seed(seed=0)
                X = np.random.randn(50).reshape(-1, 1)
                y = np.random.randint(0, 2, 50)
                X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)

            .. testcode::

                import optuna
                from sklearn.linear_model import SGDClassifier

                def objective(trial):
                    clf = SGDClassifier(random_state=0)
                    for step in range(100):
                        clf.partial_fit(X_train, y_train, np.unique(y))
                        intermediate_value = clf.score(X_valid, y_valid)
                        trial.report(intermediate_value, step=step)
                        if trial.should_prune():
                            raise TrialPruned()

                    return clf.score(X_valid, y_valid)

                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=3)


        Args:
            value:
                A value returned from the objective function.
            step:
                Step of the trial (e.g., Epoch of neural network training).
        """

        try:
            # For convenience, we allow users to report a value that can be cast to `float`.
            value = float(value)
        except (TypeError, ValueError):
            message = "The `value` argument is of type '{}' but supposed to be a float.".format(
                type(value).__name__
            )
            raise TypeError(message)

        if step < 0:
            raise ValueError("The `step` argument is {} but cannot be negative.".format(step))

        self.storage.set_trial_intermediate_value(self._trial_id, step, value)

    def should_prune(self, step=None):
        # type: (Optional[int]) -> bool
        """Suggest whether the trial should be pruned or not.

        The suggestion is made by a pruning algorithm associated with the trial and is based on
        previously reported values. The algorithm can be specified when constructing a
        :class:`~optuna.study.Study`.

        .. note::
            If no values have been reported, the algorithm cannot make meaningful suggestions.
            Similarly, if this method is called multiple times with the exact same set of reported
            values, the suggestions will be the same.

        .. seealso::
            Please refer to the example code in :func:`optuna.trial.Trial.report`.

        Args:
            step:
                Deprecated since 0.12.0: Step of the trial (e.g., epoch of neural network
                training). Deprecated in favor of always considering the most recent step.

        Returns:
            A boolean value. If :obj:`True`, the trial should be pruned according to the
            configured pruning algorithm. Otherwise, the trial should continue.
        """
        if step is not None:
            warnings.warn(
                "The use of `step` argument is deprecated. "
                "The last reported step is used instead of "
                "the step given by the argument.",
                DeprecationWarning,
            )

        trial = self.study._storage.get_trial(self._trial_id)
        return self.study.pruner.prune(self.study, trial)

    def set_user_attr(self, key, value):
        # type: (str, Any) -> None
        """Set user attributes to the trial.

        The user attributes in the trial can be access via :func:`optuna.trial.Trial.user_attrs`.

        Example:

            Save fixed hyperparameters of neural network training.

            .. testsetup::

                import numpy as np
                from sklearn.model_selection import train_test_split

                np.random.seed(seed=0)
                X = np.random.randn(50).reshape(-1, 1)
                y = np.random.randint(0, 2, 50)
                X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)

            .. testcode::

                import optuna
                from sklearn.neural_network import MLPClassifier

                def objective(trial):
                    trial.set_user_attr('BATCHSIZE', 128)
                    momentum = trial.suggest_uniform('momentum', 0, 1.0)
                    clf = MLPClassifier(hidden_layer_sizes=(100, 50),
                                        batch_size=trial.user_attrs['BATCHSIZE'],
                                        momentum=momentum, solver='sgd', random_state=0)
                    clf.fit(X_train, y_train)

                    return clf.score(X_valid, y_valid)

                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=3)
                assert 'BATCHSIZE' in study.best_trial.user_attrs.keys()
                assert study.best_trial.user_attrs['BATCHSIZE'] == 128


        Args:
            key:
                A key string of the attribute.
            value:
                A value of the attribute. The value should be JSON serializable.
        """

        self.storage.set_trial_user_attr(self._trial_id, key, value)

    def set_system_attr(self, key, value):
        # type: (str, Any) -> None
        """Set system attributes to the trial.

        Note that Optuna internally uses this method to save system messages such as failure
        reason of trials. Please use :func:`~optuna.trial.Trial.set_user_attr` to set users'
        attributes.

        Args:
            key:
                A key string of the attribute.
            value:
                A value of the attribute. The value should be JSON serializable.
        """

        self.storage.set_trial_system_attr(self._trial_id, key, value)

    def _suggest(self, name, distribution):
        # type: (str, BaseDistribution) -> Any

        # the first two statements don't execute?
        if self._is_fixed_param(name, distribution):
            param_value = self.system_attrs["fixed_params"][name]
        elif self._is_relative_param(name, distribution):
            param_value = self.relative_params[name]
        else:
            # get the trial
            trial = self.storage.get_trial(self._trial_id)

            # this will just return the study (not hyperband)
            study = pruners._filter_study(self.study, trial)

            # get parameter value
            param_value, samples, scores = self.study.sampler.sample_independent(
                study, trial, name, distribution
            )
            self.study.info["names"].append(name)
            self.study.info[f"{name}_smpls"] = samples
            self.study.info[f"{name}_scrs"] = scores
            self.study.info[f"{name}_dist"] = distribution

        # should be able to change parameters later
        param_value = self._set_new_param_or_get_existing(name, param_value, distribution)
        return param_value, name, samples, scores, distribution

    def _set_new_param_or_get_existing(self, name, param_value, distribution):
        # type: (str, Any, BaseDistribution) -> Any

        param_value_in_internal_repr = distribution.to_internal_repr(param_value)
        # can change trial_id
        set_success = self.storage.set_trial_param(
            self._trial_id, name, param_value_in_internal_repr, distribution
        )
        if not set_success:
            param_value_in_internal_repr = self.storage.get_trial_param(self._trial_id, name)
            param_value = distribution.to_external_repr(param_value_in_internal_repr)

        return param_value

    def _is_fixed_param(self, name, distribution):
        # type: (str, BaseDistribution) -> bool

        if "fixed_params" not in self.system_attrs:
            return False

        if name not in self.system_attrs["fixed_params"]:
            return False

        param_value = self.system_attrs["fixed_params"][name]
        param_value_in_internal_repr = distribution.to_internal_repr(param_value)

        contained = distribution._contains(param_value_in_internal_repr)
        if not contained:
            warnings.warn(
                "Fixed parameter '{}' with value {} is out of range "
                "for distribution {}.".format(name, param_value, distribution)
            )
        return contained

    def _is_relative_param(self, name, distribution):
        # type: (str, BaseDistribution) -> bool

        if name not in self.relative_params:
            return False

        if name not in self.relative_search_space:
            raise ValueError(
                "The parameter '{}' was sampled by `sample_relative` method "
                "but it is not contained in the relative search space.".format(name)
            )

        relative_distribution = self.relative_search_space[name]
        distributions.check_distribution_compatibility(relative_distribution, distribution)

        param_value = self.relative_params[name]
        param_value_in_internal_repr = distribution.to_internal_repr(param_value)
        return distribution._contains(param_value_in_internal_repr)

    def _check_distribution(self, name, distribution):
        # type: (str, BaseDistribution) -> None

        old_distribution = self.distributions.get(name, distribution)
        if old_distribution != distribution:
            warnings.warn(
                'Inconsistent parameter values for distribution with name "{}"! '
                "This might be a configuration mistake. "
                "Optuna allows to call the same distribution with the same "
                "name more then once in a trial. "
                "When the parameter values are inconsistent optuna only "
                "uses the values of the first call and ignores all following. "
                "Using these values: {}".format(name, old_distribution._asdict()),
                RuntimeWarning,
            )

    def check(self, cstrs, act_cat_cols, nc):
        if len(self.params) == nc:
            return True
        # keys, vals = list(self.params.keys()), list(self.params.values())
        # prms = tuple(zip(keys, vals))
        prms = tuple([(f"c{i}", cstrs[f"c{i}"]) for i in act_cat_cols])
        if prms in self.study.evaled:
            return True
        self.study.evaled.add(prms)
        return False

    def setter(self):

        prms = tuple([(name, self.params[name]) for name in self.study.info["names"]])

        if self.number < 10:

            # if prms in self.study.evaled or prms not in self.study.cat_preds:
            while prms in self.study.evaled or prms not in self.study.cat_preds_set:
                prms = self.study.cat_preds[self.study.rnd.choice(len(self.study.cat_preds))]

            cstrs = {}
            self.storage.clear_params_and_dists(self.number)
            for nam, val in prms:
                self.storage.set_trial_param(
                    self.number,
                    nam,
                    self.study.info[f"{nam}_dist"].to_internal_repr(val),
                    self.study.info[f"{nam}_dist"],
                )
                cstrs[nam] = val
            act_cat_cols = [int(prms[i][0][1]) for i in range(len(prms))]
            self.study.evaled.add(prms)
            return cstrs, act_cat_cols

        # cur_inds = {}
        # for name in self.study.info["names"]:
        #     cur_inds[f"{name}"] = 0

        self.storage.clear_params_and_dists(self.number)

        names = self.study.info["names"]
        lst = []
        if len(names) == 3:
            for i, val1 in enumerate(self.study.info[f"{names[0]}_smpls"]):
                mult = self.study.info[f"{names[0]}_scrs"][i]
                lst.append(((val1, None, None), mult))
                for j, val2 in enumerate(self.study.info[f"{names[1]}_smpls"]):
                    if i == 0:
                        mult = self.study.info[f"{names[1]}_scrs"][j]
                        lst.append(((None, val2, None), mult))
                    pred = ((names[0], val1), (names[1], val2))
                    if pred not in self.study.evaled and pred in self.study.cat_preds_set:
                        mult = (
                            self.study.info[f"{names[0]}_scrs"][i]
                            * self.study.info[f"{names[1]}_scrs"][j]
                        ) / 2
                        lst.append(((val1, val2, None), mult))
                    for k, val3 in enumerate(self.study.info[f"{names[2]}_smpls"]):
                        if i == 0 and j == 0:
                            mult = self.study.info[f"{names[2]}_scrs"][k]
                            lst.append(((None, None, val3), mult))
                        elif i == 0:
                            pred = ((names[1], val2), (names[2], val3))
                            if pred not in self.study.evaled and pred in self.study.cat_preds_set:
                                mult = (
                                    self.study.info[f"{names[1]}_scrs"][j]
                                    * self.study.info[f"{names[2]}_scrs"][k]
                                ) / 2
                                lst.append(((None, val2, val3), mult))  

                        elif j == 0:
                            pred = ((names[0], val1), (names[2], val3))
                            if pred not in self.study.evaled and pred in self.study.cat_preds_set:
                                mult = (
                                    self.study.info[f"{names[0]}_scrs"][i]
                                    * self.study.info[f"{names[2]}_scrs"][k]
                                ) / 2
                                lst.append(((val1, None, val3), mult)) 
                        pred = ((names[0], val1), (names[1], val2), (names[2], val3))
                        if pred not in self.study.evaled and pred in self.study.cat_preds_set:                       
                            mult = (
                                self.study.info[f"{names[0]}_scrs"][i]
                                * self.study.info[f"{names[1]}_scrs"][j]
                                * self.study.info[f"{names[2]}_scrs"][k]
                            ) / 3
                            lst.append(((val1, val2, val3), mult))
        elif len(names) == 4:
            for i, val1 in enumerate(self.study.info[f"{names[0]}_smpls"]):
                for j, val2 in enumerate(self.study.info[f"{names[1]}_smpls"]):
                    for k, val3 in enumerate(self.study.info[f"{names[2]}_smpls"]):
                        for l, val4 in enumerate(self.study.info[f"{names[3]}_smpls"]):
                            mult = (
                                self.study.info[f"{names[0]}_scrs"][i]
                                * self.study.info[f"{names[1]}_scrs"][j]
                                * self.study.info[f"{names[2]}_scrs"][k]
                                * self.study.info[f"{names[3]}_scrs"][l]
                            )
                            lst.append(((val1, val2, val3, val4), mult))
        # print(self.number, len(lst))
        # lst = sorted(lst, key=lambda x: x[1])
        # lst = lst[::-1]
        # print(lst[:30])

        while True:

            if len(lst) == 0:
                print(self.number)

            vals = max(lst, key=itemgetter(1))[0]

            cstrs = {}
            for i, name in enumerate(names):
                cstrs[name] = vals[i]

            prms = tuple([(name, cstrs[name]) for name in names])
            prms_check = tuple([(name, cstrs[name]) for name in names if cstrs[name] is not None])
            act_cat_cols = []
            for i, (nam, val) in enumerate(prms):
                if val is not None: 
                    self.storage.set_trial_param(
                        self.number,
                        nam,
                        self.study.info[f"{nam}_dist"].to_internal_repr(val),
                        self.study.info[f"{nam}_dist"],
                    )
                    act_cat_cols.append(i)
                cstrs[nam] = val
            self.study.evaled.add(prms_check)
            return cstrs, act_cat_cols

            # max_val, max_name = 0, self.study.info["names"][0]

            # for nam in self.study.info["names"]:
            #     if cur_inds[nam] == len(self.study.info[f"{nam}_smpls"]) - 1:
            #         continue
            #     cur_inds[nam] += 1
            #     new_high_score = 1
            #     for name in self.study.info["names"]:
            #         new_high_score *= self.study.info[f"{name}_scrs"][cur_inds[name]]

            #     if new_high_score > max_val:
            #         max_val = new_high_score
            #         max_name = nam

            #     cur_inds[nam] -= 1
            # print(cur_inds)
            # cur_inds[max_name] += 1


    def setter2(self):

        prms = tuple([(name, self.params[name]) for name in self.study.info["names"]])

        if self.number < 10:

            # if prms in self.study.evaled or prms not in self.study.cat_preds:
            while prms in self.study.evaled or prms not in self.study.cat_preds:
                prms = self.study.cat_preds[self.study.rnd.choice(len(self.study.cat_preds))]

            cstrs = {}
            self.storage.clear_params_and_dists(self.number)
            for nam, val in prms:
                self.storage.set_trial_param(
                    self.number,
                    nam,
                    self.study.info[f"{nam}_dist"].to_internal_repr(val),
                    self.study.info[f"{nam}_dist"],
                )
                cstrs[nam] = val
            act_cat_cols = [int(prms[i][0][1]) for i in range(len(prms))]
            self.study.evaled.add(prms)
            return cstrs, act_cat_cols

        # cur_inds = {}
        # for name in self.study.info["names"]:
        #     cur_inds[f"{name}"] = 0

        self.storage.clear_params_and_dists(self.number)

        names = self.study.info["names"]
        lst = []
        if len(names) == 3:
            for i, val1 in enumerate(self.study.info[f"{names[0]}_smpls"]):
                mult = self.study.info[f"{names[0]}_scrs"][i]
                lst.append(((val1, None, None), mult))
                for j, val2 in enumerate(self.study.info[f"{names[1]}_smpls"]):
                    if i == 0:
                        mult = self.study.info[f"{names[1]}_scrs"][j]
                        lst.append(((None, val2, None), mult))
                    mult = (
                        self.study.info[f"{names[0]}_scrs"][i]
                        * self.study.info[f"{names[1]}_scrs"][j]
                    ) / 2
                    lst.append(((val1, val2, None), mult))
                    for k, val3 in enumerate(self.study.info[f"{names[2]}_smpls"]):
                        if i == 0 and j == 0:
                            mult = self.study.info[f"{names[2]}_scrs"][k]
                            lst.append(((None, None, val3), mult))
                        elif i == 0:
                            mult = (
                                self.study.info[f"{names[1]}_scrs"][j]
                                * self.study.info[f"{names[2]}_scrs"][k]
                            ) / 2
                            lst.append(((None, val2, val3), mult))  

                        elif j == 0:
                            mult = (
                                self.study.info[f"{names[0]}_scrs"][i]
                                * self.study.info[f"{names[2]}_scrs"][k]
                            ) / 2
                            lst.append(((val1, None, val3), mult))                        
                        mult = (
                            self.study.info[f"{names[0]}_scrs"][i]
                            * self.study.info[f"{names[1]}_scrs"][j]
                            * self.study.info[f"{names[2]}_scrs"][k]
                        ) / 3
                        lst.append(((val1, val2, val3), mult))
        elif len(names) == 4:
            for i, val1 in enumerate(self.study.info[f"{names[0]}_smpls"]):
                for j, val2 in enumerate(self.study.info[f"{names[1]}_smpls"]):
                    for k, val3 in enumerate(self.study.info[f"{names[2]}_smpls"]):
                        for l, val4 in enumerate(self.study.info[f"{names[3]}_smpls"]):
                            mult = (
                                self.study.info[f"{names[0]}_scrs"][i]
                                * self.study.info[f"{names[1]}_scrs"][j]
                                * self.study.info[f"{names[2]}_scrs"][k]
                                * self.study.info[f"{names[3]}_scrs"][l]
                            )
                            lst.append(((val1, val2, val3, val4), mult))
        # print(self.number, len(lst))
        lst = sorted(lst, key=lambda x: x[1])
        lst = lst[::-1]
        # print(lst[:30])

        while True:

            if len(lst) == 0:
                print(self.number)

            vals = lst.pop(0)[0]

            cstrs = {}
            for i, name in enumerate(names):
                cstrs[name] = vals[i]

            prms = tuple([(name, cstrs[name]) for name in names])
            prms_check = tuple([(name, cstrs[name]) for name in names if cstrs[name] is not None])
            act_cat_cols = []
            if prms_check not in self.study.evaled and prms_check in self.study.cat_preds:
                for i, (nam, val) in enumerate(prms):
                    if val is not None: 
                        self.storage.set_trial_param(
                            self.number,
                            nam,
                            self.study.info[f"{nam}_dist"].to_internal_repr(val),
                            self.study.info[f"{nam}_dist"],
                        )
                        act_cat_cols.append(i)
                    cstrs[nam] = val
                self.study.evaled.add(prms_check)
                return cstrs, act_cat_cols

            # max_val, max_name = 0, self.study.info["names"][0]

            # for nam in self.study.info["names"]:
            #     if cur_inds[nam] == len(self.study.info[f"{nam}_smpls"]) - 1:
            #         continue
            #     cur_inds[nam] += 1
            #     new_high_score = 1
            #     for name in self.study.info["names"]:
            #         new_high_score *= self.study.info[f"{name}_scrs"][cur_inds[name]]

            #     if new_high_score > max_val:
            #         max_val = new_high_score
            #         max_name = nam

            #     cur_inds[nam] -= 1
            # print(cur_inds)
            # cur_inds[max_name] += 1


    def is_valid(self, df, pred, num_cols, cat_cols):
        # numerical constraints
        num_constrs = [f"{col} >= {pred[f'{col}_min']} & {col} <= {pred[f'{col}_min'] + pred[f'{col}_len']}" for col in num_cols]
        # categorical constraints
        cat_constrs = [f"{col} == \"{pred[col]}\"" for col in cat_cols]
        # return True if tuples satisfy the predicate, else False
        return df.query(" & ".join(num_constrs + cat_constrs)).shape[0] > 0

        # cstr_dfs = []
        # if num_cols:
        #     lt = [df[col] >= pred[f"{col}_min"] for col in num_cols]
        #     gt = [df[col] <= pred[f"{col}_min"] + pred[f"{col}_len"] for col in num_cols]
        #     cstr_dfs += lt + gt
        # if cat_cols:
        #     ne = [df[col] == pred[col] for col in cat_cols]
        #     cstr_dfs += ne
        # # this dataframe has removed all tuples that satisfy the predicate
        # return df[np.logical_and.reduce(cstr_dfs)].shape[0] > 0


    def update_params(self, prms):

        self.storage.clear_params_and_dists(self.number)
        # update trial params
        for name, val in prms.items():
            self.storage.set_trial_param(
                self.number,
                name,
                self.study.info[f"{name}_dist"].to_internal_repr(val),
                self.study.info[f"{name}_dist"],
            )

        # add parameters to the evaluated set
        # self.study.evaled.add(prms)

    def fixer_combined(self, df, k, cat_preds, probs, cat_cols, num_cols):

        if self.number < 10:
            while True:
                ind = self.study.rnd.choice(range(len(cat_preds)), p=probs)
                prms = dict(zip(cat_cols, cat_preds[ind]))
                for col in num_cols:
                    if np.issubdtype(df[col].dtype, np.signedinteger) or np.issubdtype(df[col].dtype, np.unsignedinteger):
                        prms[f"{col}_min"] = self.study.rnd.randint(self.study.info[f"{col}_min_dist"].low, self.study.info[f"{col}_min_dist"].high)
                        prms[f"{col}_len"] = self.study.rnd.randint(0, self.study.info[f"{col}_len_dist"].high - self.study.info[f"{col}_len_dist"].low)
                    elif np.issubdtype(self.df[col].dtype, np.floating):
                        prms[f"{col}_min"] = self.study.rnd.uniform(self.study.info[f"{col}_min_dist"].low, self.study.info[f"{col}_min_dist"].high)
                        prms[f"{col}_len"] = self.study.rnd.uniform(0, self.study.info[f"{col}_len_dist"].high - self.study.info[f"{col}_len_dist"].low)                        
                if self.is_valid(df, prms, num_cols, cat_cols):
                    self.update_params({k: v for k, v in prms.items() if k in self.study.info["names"]})
                    return prms
        
        # SRSWR or SRSWOR
        indicies = self.study.rnd.choice(range(len(cat_preds)), size=1, p=probs)
        self.study.info["cat_smpls"] = [cat_preds[i] for i in indicies]
        self.study.info["cat_scrs"] = [probs[i] for i in indicies]

        all_param_names = ["cat"] + self.study.info["names"]
        # all combinations of predicates
        prds = list(product(*[self.study.info[f"{name}_smpls"] for name in all_param_names]))
        # all multiplied combinations of scores
        scrs = [np.prod(scrs) for scrs in product(*[self.study.info[f"{name}_scrs"] for name in all_param_names])]
        prms_lst = list(zip(prds, scrs))

        prms_lst = sorted(prms_lst, key=lambda x: x[1], reverse=True)

        for prms, _ in prms_lst:
            prms = [y for x in prms for y in (x if isinstance(x, tuple) else (x,))]
            prms = dict(zip(cat_cols + self.study.info["names"], prms))
            if self.is_valid(df, prms, num_cols, cat_cols):
                self.update_params({k: v for k, v in prms.items() if k in self.study.info["names"]})
                return dict(prms)
        exit()
        


    def fixer(self, df, k, algo, cat_cols, num_cols, cat_vals, probs, code_to_value):

        prms = {}
        if self.number < 10:
            while True:
                if algo == "TPE_categorical" or algo == "TPE_individual_contribution":
                    prms = dict(zip(cat_cols, self.study.cat_preds[self.study.rnd.choice(len(self.study.cat_preds))]))
                elif algo == "weighted_sample_positive" or algo == "weighted_sample_shift" or algo == "weighted_sample_halving":
                    for col in cat_cols:
                        prms[col] = self.study.rnd.choice(cat_vals[col], replace=False, p=probs[col])
                    if tuple(prms.values())[:len(cat_vals)] not in self.study.cat_preds_set:
                        continue
                for col in num_cols:
                    prms[f"{col}_min"] = self.study.rnd.randint(self.study.info[f"{col}_min_dist"].low, self.study.info[f"{col}_min_dist"].high)
                    prms[f"{col}_len"] = self.study.rnd.randint(0, self.study.info[f"{col}_len_dist"].high - self.study.info[f"{col}_len_dist"].low)
                if algo == "TPE_individual_contribution":
                    prms_final = {k: (code_to_value[k][v] if k in cat_cols else v) for k, v in prms.items()}
                else:
                    prms_final = prms
                if self.is_valid(df, prms_final, num_cols, cat_cols):
                    break
            self.update_params({k: v for k, v in prms.items() if k in self.study.info["names"]}, )
            return prms

        if algo == "weighted_sample_positive" or algo == "weighted_sample_shift" or algo == "weighted_sample_halving":
            for col in cat_cols:
                indicies = self.study.rnd.choice(len(cat_vals[col]), size=min(k, len(cat_vals[col])), replace=False, p=probs[col])
                self.study.info[f"{col}_smpls"] = [cat_vals[col][i] for i in indicies]
                self.study.info[f"{col}_scrs"] = [probs[col][i] for i in indicies]
            all_param_names = list(cat_vals.keys()) + self.study.info["names"]
        elif algo == "TPE_categorical" or algo == "TPE_individual_contribution":
            all_param_names = self.study.info["names"]

        # all combinations of predicates
        prds = list(product(*[self.study.info[f"{name}_smpls"] for name in all_param_names]))
        # all multiplied combinations of scores
        scrs = [np.prod(scrs) for scrs in product(*[self.study.info[f"{name}_scrs"] for name in all_param_names])]
        prms_lst = list(zip(prds, scrs))

        prms_lst = sorted(prms_lst, key=lambda x: x[1], reverse=True)
        # print(self.study.cat_preds_set)
        for prms, _ in prms_lst:
            if algo == "weighted_sample_positive" or algo == "weighted_sample_shift" or algo == "weighted_sample_halving":
                if prms[:len(cat_vals.keys())] not in self.study.cat_preds_set:
                    continue
            elif algo == "TPE_categorical" or algo == "TPE_individual_contribution":
                names = prms[:len(cat_cols)]
                if names not in self.study.cat_preds_set:
                    continue
            prms = dict(zip(all_param_names, prms))
            if algo == "TPE_individual_contribution":
                prms_final = {k: (code_to_value[k][v] if k in cat_cols else v) for k, v in prms.items()}
            else:
                prms_final = prms
            if self.is_valid(df, prms_final, num_cols, cat_cols):
                self.update_params({k: v for k, v in prms.items() if k in self.study.info["names"]})
                return dict(prms)
        exit()
        # initial parameters for this trial
        # prms = tuple(self.params.items())
        prms = tuple({k: v for k, v in self.params.items() if "_min" not in k and "_len" not in k})

        if self.number < 10:
            # get a valid predicate that hasn't been evaluated
            while prms in self.study.evaled or prms not in self.study.cat_preds_set:
                prms = self.study.cat_preds[self.study.rnd.choice(len(self.study.cat_preds))]
            
            self.update_params(prms)
            return dict(prms)

        names = [name for name in self.study.info["names"] if "_min" not in name and "_len" not in name]
        prms_lst = []
        name_combos = list(
            chain.from_iterable(
                combinations(names, r) for r in range(len(names), len(names) + 1)
            )
        )
        for combo in name_combos:
            for name in combo:
                self.study.info[f"{name}_smpls"] = list(zip([name] * len(self.study.info[f"{name}_smpls"]), self.study.info[f"{name}_smpls"]))
            # all combinations of predicates
            prds = list(product(*[self.study.info[f"{name}_smpls"] for name in combo]))
            # all multiplied combinations of scores
            scrs = [np.prod(scrs) for scrs in product(*[self.study.info[f"{name}_scrs"] for name in combo])]
            prms_lst += list(zip(prds, scrs))

        prms_lst = sorted(prms_lst, key=lambda x: x[1])
        
        for prms in prms_lst:
            if prms not in self.study.evaled and prms in self.study.cat_preds_set:
                self.update_params(prms)
                return dict(prms)

        print("Random Predicate")
        # executes if the list of EI candidates is empty
        while prms in self.study.evaled or prms not in self.study.cat_preds_set:
            prms = self.study.cat_preds[self.study.rnd.choice(len(self.study.cat_preds))]
        
        self.update_params(prms)
        return dict(prms)


    def setter_fixed(self):
        
        # initial parameters for this trial
        # prms = tuple(self.params.items())
        prms = tuple({k: v for k, v in self.params.items() if "_min" not in k and "_len" not in k})

        if self.number < 10:
            # get a valid predicate that hasn't been evaluated
            while prms in self.study.evaled or prms not in self.study.cat_preds_set:
                prms = self.study.cat_preds[self.study.rnd.choice(len(self.study.cat_preds))]
            
            self.update_params(prms)
            return dict(prms)

        names = [name for name in self.study.info["names"] if "_min" not in name and "_len" not in name]
        prms_lst = []
        name_combos = list(
            chain.from_iterable(
                combinations(names, r) for r in range(len(names), len(names) + 1)
            )
        )
        for combo in name_combos:
            for name in combo:
                self.study.info[f"{name}_smpls"] = list(zip([name] * len(self.study.info[f"{name}_smpls"]), self.study.info[f"{name}_smpls"]))
            # all combinations of predicates
            prds = list(product(*[self.study.info[f"{name}_smpls"] for name in combo]))
            # all multiplied combinations of scores
            scrs = [np.prod(scrs) for scrs in product(*[self.study.info[f"{name}_scrs"] for name in combo])]
            prms_lst += list(zip(prds, scrs))

        prms_lst = sorted(prms_lst, key=lambda x: x[1])
        
        for prms, _ in prms_lst:
            if prms not in self.study.evaled and prms in self.study.cat_preds_set:
                self.update_params(prms)
                return dict(prms)

        print("Random Predicate")
        # executes if the list of EI candidates is empty
        while prms in self.study.evaled or prms not in self.study.cat_preds_set:
            prms = self.study.cat_preds[self.study.rnd.choice(len(self.study.cat_preds))]
        
        self.update_params(prms)
        return dict(prms)


    def setter_old(self):
        
        prms = tuple([(name, self.params[name]) for name in self.study.info["names"]])

        if self.number < 10:
            # if prms in self.study.evaled or prms not in self.study.cat_preds:
            while prms in self.study.evaled or prms not in self.study.cat_preds_set:
                prms = self.study.cat_preds[self.study.rnd.choice(len(self.study.cat_preds))]

            cstrs = {}
            self.storage.clear_params_and_dists(self.number)
            for nam, val in prms:
                self.storage.set_trial_param(
                    self.number,
                    nam,
                    self.study.info[f"{nam}_dist"].to_internal_repr(val),
                    self.study.info[f"{nam}_dist"],
                )
                cstrs[nam] = val
            act_cat_cols = [prms[i][0] for i in range(len(prms))]
            self.study.evaled.add(prms)
            return cstrs, act_cat_cols

        # cur_inds = {}
        # for name in self.study.info["names"]:
        #     cur_inds[f"{name}"] = 0

        self.storage.clear_params_and_dists(self.number)

        names = self.study.info["names"]
        lst = []
        if len(names) == 3:
            for i, val1 in enumerate(self.study.info[f"{names[0]}_smpls"]):
                for j, val2 in enumerate(self.study.info[f"{names[1]}_smpls"]):
                    for k, val3 in enumerate(self.study.info[f"{names[2]}_smpls"]):
                        pred = ((names[0], val1), (names[1], val2), (names[2], val3))
                        if pred not in self.study.evaled and pred in self.study.cat_preds_set:
                            mult = (
                                self.study.info[f"{names[0]}_scrs"][i]
                                * self.study.info[f"{names[1]}_scrs"][j]
                                * self.study.info[f"{names[2]}_scrs"][k]
                            )
                            lst.append(((val1, val2, val3), mult))
        elif len(names) == 4:
            for i, val1 in enumerate(self.study.info[f"{names[0]}_smpls"]):
                for j, val2 in enumerate(self.study.info[f"{names[1]}_smpls"]):
                    for k, val3 in enumerate(self.study.info[f"{names[2]}_smpls"]):
                        for l, val4 in enumerate(self.study.info[f"{names[3]}_smpls"]):
                            mult = (
                                self.study.info[f"{names[0]}_scrs"][i]
                                * self.study.info[f"{names[1]}_scrs"][j]
                                * self.study.info[f"{names[2]}_scrs"][k]
                                * self.study.info[f"{names[3]}_scrs"][l]
                            )
                            lst.append(((val1, val2, val3, val4), mult))
        # print(self.number, len(lst))
        # lst = sorted(lst, key=lambda x: x[1])
        # lst = lst[::-1]
        # print(len(lst))
        # print(lst[:10])
        # if self.number == 20:
        #     exit(0)
        
        while True:

            if len(lst) == 0:
                print(self.number)
                break

            vals = max(lst, key=itemgetter(1))[0]

            cstrs = {}
            for i, name in enumerate(names):
                cstrs[name] = vals[i]

            prms = tuple([(name, cstrs[name]) for name in names])
            act_cat_cols = []
            for i, (nam, val) in enumerate(prms):
                self.storage.set_trial_param(
                    self.number,
                    nam,
                    self.study.info[f"{nam}_dist"].to_internal_repr(val),
                    self.study.info[f"{nam}_dist"],
                )
                act_cat_cols.append(nam)
                cstrs[nam] = val
            self.study.evaled.add(prms)
            return cstrs, act_cat_cols

        # if prms in self.study.evaled or prms not in self.study.cat_preds:
        while prms in self.study.evaled or prms not in self.study.cat_preds_set:
            prms = self.study.cat_preds[self.study.rnd.choice(len(self.study.cat_preds))]

        cstrs = {}
        self.storage.clear_params_and_dists(self.number)
        for nam, val in prms:
            self.storage.set_trial_param(
                self.number,
                nam,
                self.study.info[f"{nam}_dist"].to_internal_repr(val),
                self.study.info[f"{nam}_dist"],
            )
            cstrs[nam] = val
        act_cat_cols = [prms[i][0] for i in range(len(prms))]
        self.study.evaled.add(prms)
        return cstrs, act_cat_cols

    def setter_old2(self):
        
        prms = tuple([(name, self.params[name]) for name in self.study.info["names"]])

        if self.number < 10:

            # if prms in self.study.evaled or prms not in self.study.cat_preds:
            while prms in self.study.evaled or prms not in self.study.cat_preds_set:
                prms = self.study.cat_preds[self.study.rnd.choice(len(self.study.cat_preds))]

            cstrs = {}
            self.storage.clear_params_and_dists(self.number)
            for nam, val in prms:
                self.storage.set_trial_param(
                    self.number,
                    nam,
                    self.study.info[f"{nam}_dist"].to_internal_repr(val),
                    self.study.info[f"{nam}_dist"],
                )
                cstrs[nam] = val
            act_cat_cols = [int(prms[i][0][1]) for i in range(len(prms))]
            self.study.evaled.add(prms)
            return cstrs, act_cat_cols

        # cur_inds = {}
        # for name in self.study.info["names"]:
        #     cur_inds[f"{name}"] = 0

        self.storage.clear_params_and_dists(self.number)

        names = self.study.info["names"]
        lst = []
        if len(names) == 3:
            for i, val1 in enumerate(self.study.info[f"{names[0]}_smpls"]):
                for j, val2 in enumerate(self.study.info[f"{names[1]}_smpls"]):
                    for k, val3 in enumerate(self.study.info[f"{names[2]}_smpls"]):
                        mult = (
                            self.study.info[f"{names[0]}_scrs"][i]
                            * self.study.info[f"{names[1]}_scrs"][j]
                            * self.study.info[f"{names[2]}_scrs"][k]
                        )
                        lst.append(((val1, val2, val3), mult))
        elif len(names) == 4:
            for i, val1 in enumerate(self.study.info[f"{names[0]}_smpls"]):
                for j, val2 in enumerate(self.study.info[f"{names[1]}_smpls"]):
                    for k, val3 in enumerate(self.study.info[f"{names[2]}_smpls"]):
                        for l, val4 in enumerate(self.study.info[f"{names[3]}_smpls"]):
                            mult = (
                                self.study.info[f"{names[0]}_scrs"][i]
                                * self.study.info[f"{names[1]}_scrs"][j]
                                * self.study.info[f"{names[2]}_scrs"][k]
                                * self.study.info[f"{names[3]}_scrs"][l]
                            )
                            lst.append(((val1, val2, val3, val4), mult))
        # print(self.number, len(lst))
        lst = sorted(lst, key=lambda x: x[1])
        lst = lst[::-1]
        # print(len(lst))
        # print(lst[:10])
        # if self.number == 20:
        #     exit(0)
        
        while True:

            if len(lst) == 0:
                print(self.number)
                break

            vals = lst.pop(0)

            cstrs = {}
            for i, name in enumerate(names):
                cstrs[name] = vals[i]

            prms = tuple([(name, cstrs[name]) for name in names])
            act_cat_cols = []
            if prms not in self.study.evaled and prms in self.study.cat_preds_set:
                for i, (nam, val) in enumerate(prms):
                    self.storage.set_trial_param(
                        self.number,
                        nam,
                        self.study.info[f"{nam}_dist"].to_internal_repr(val),
                        self.study.info[f"{nam}_dist"],
                    )
                    act_cat_cols.append(i)
                    cstrs[nam] = val
                self.study.evaled.add(prms)
                return cstrs, act_cat_cols

        # if prms in self.study.evaled or prms not in self.study.cat_preds:
        while prms in self.study.evaled or prms not in self.study.cat_preds_set:
            prms = self.study.cat_preds[self.study.rnd.choice(len(self.study.cat_preds))]

        cstrs = {}
        self.storage.clear_params_and_dists(self.number)
        for nam, val in prms:
            self.storage.set_trial_param(
                self.number,
                nam,
                self.study.info[f"{nam}_dist"].to_internal_repr(val),
                self.study.info[f"{nam}_dist"],
            )
            cstrs[nam] = val
        act_cat_cols = [int(prms[i][0][1]) for i in range(len(prms))]
        self.study.evaled.add(prms)
        return cstrs, act_cat_cols


    def get_new_random_params(self, param_list, distr):

        while True:
            params = {}
            cstrs = {}
            act_cat_cols = []
            checker = tuple()
            for i in range(len(param_list)):
                # choose whether or not to select a value from this column
                v = self.study.rnd.choice(2)
                if v == 1:
                    # choose a value randomly from this column
                    params[f"c{i}"] = "Yes"
                    act_cat_cols.append(i)
                    if distr == "int":
                        params[f"c{i}_val"] = self.study.rnd.choice(list(range(param_list[i] + 1)))
                        params[f"c{i}_distr"] = IntUniformDistribution(low=0, high=param_list[i])
                    elif distr == "cat":
                        params[f"c{i}_val"] = self.study.rnd.choice(param_list[i])
                        params[f"c{i}_distr"] = CategoricalDistribution(choices=param_list[i])
                    cstrs[f"c{i}"] = params[f"c{i}_val"]
                else:
                    params[f"c{i}"] = "No"
            if len(cstrs) == 0:
                continue

            checker = tuple([(f"c{i}", cstrs[f"c{i}"]) for i in act_cat_cols])

            if checker not in self.study.evaled:
                self.study.evaled.add(checker)
                break

        self.storage.clear_params_and_dists(self.number)

        col_distr = CategoricalDistribution(choices=("Yes", "No"))
        for i in range(len(param_list)):
            if params[f"c{i}"] == "Yes":
                self.storage.set_trial_param(self.number, f"c{i}", 0, col_distr)
                self.storage.set_trial_param(
                    self.number,
                    f"c{i}_val",
                    params[f"c{i}_distr"].to_internal_repr(params[f"c{i}_val"]),
                    params[f"c{i}_distr"],
                )
            elif params[f"c{i}"] == "No":
                self.storage.set_trial_param(self.number, f"c{i}", 1, col_distr)

        return cstrs, act_cat_cols

    @property
    def number(self):
        # type: () -> int
        """Return trial's number which is consecutive and unique in a study.

        Returns:
            A trial number.
        """

        return self.storage.get_trial_number_from_id(self._trial_id)

    @property
    def trial_id(self):
        # type: () -> int
        """Return trial ID.

        Note that the use of this is deprecated.
        Please use :attr:`~optuna.trial.Trial.number` instead.

        Returns:
            A trial ID.
        """

        warnings.warn(
            "The use of `Trial.trial_id` is deprecated. Please use `Trial.number` instead.",
            DeprecationWarning,
        )

        self.logger.warning(
            "The use of `Trial.trial_id` is deprecated. Please use `Trial.number` instead."
        )

        return self._trial_id

    @property
    def params(self):
        # type: () -> Dict[str, Any]
        """Return parameters to be optimized.

        Returns:
            A dictionary containing all parameters.
        """

        return self.storage.get_trial_params(self._trial_id)

    @property
    def distributions(self):
        # type: () -> Dict[str, BaseDistribution]
        """Return distributions of parameters to be optimized.

        Returns:
            A dictionary containing all distributions.
        """

        return self.storage.get_trial(self._trial_id).distributions

    @property
    def user_attrs(self):
        # type: () -> Dict[str, Any]
        """Return user attributes.

        Returns:
            A dictionary containing all user attributes.
        """

        return self.storage.get_trial_user_attrs(self._trial_id)

    @property
    def system_attrs(self):
        # type: () -> Dict[str, Any]
        """Return system attributes.

        Returns:
            A dictionary containing all system attributes.
        """

        return self.storage.get_trial_system_attrs(self._trial_id)

    @property
    def datetime_start(self):
        # type: () -> Optional[datetime.datetime]
        """Return start datetime.

        Returns:
            Datetime where the :class:`~optuna.trial.Trial` started.
        """
        return self.storage.get_trial(self._trial_id).datetime_start

    @property
    def study_id(self):
        # type: () -> int
        """Return the study ID.

        .. deprecated:: 0.20.0
            The direct use of this attribute is deprecated and it is recommended that you use
            :attr:`~optuna.trial.Trial.study` instead.

        Returns:
            The study ID.
        """

        message = "The use of `Trial.study_id` is deprecated. Please use `Trial.study` instead."
        warnings.warn(message, DeprecationWarning)
        self.logger.warning(message)

        return self.study._study_id
