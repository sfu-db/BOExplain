import time

from pandas.api.types import is_numeric_dtype

from .cat_xform import individual_contribution
from .tpe_wrapper import TpeBo
from .stats import Experiment, Stats

CAT_ALG_MAP = {
    "individual_contribution": "individual_contribution_warm_start_top1",
    "categorical": "categorical",
    "categorical_warm_start": "categorical_warm_start",
}


def fmin(
    data,
    f,
    num_cols=[],
    cat_cols=[],
    columns=[],
    cat_alg=["individual_contribution"],
    n_trials=2000,
    runtime=10000,
    runs=1,
    k=5,
    random=False,
    correct_pred=None,
    increment=5,
    name="experiment_name",
    file=None,
    return_viz=False,
    use_seeds_from_paper=False,
    **kwargs,
):
    """
    Use BOExplain to minimize the objective function.

    Parameters
    ----------

    data
        pandas DataFrame of source, training, or inference data
        from which to derive an explanation.
    f
        Objective function to be minimized.
    num_cols
        Numerical columns over which to derive an explanation.
    cat_cols
        Categorical columns over which to derive an explanation.
    columns
        Columns over which to derive an explanation.
    cat_alg
        Algorithms to handle categorical parameters. Can be
            * 'individual_contribution'
            * 'categorical'
            * 'categorical_warm_start'
        See the paper for details.
    n_trials
        Maximum number of trials to perform during a run.
    runtime
        Maximum allowed time for a run in seconds.
    runs
        Number of runs to perform.
    k
        Number of TPE candidates to consider. (deprecated)
    random
        If True, perform a run using random search to
        find the constraint parameters.
    correct_pred
        If provided, will compute f-score, precision, recall,
        and jaccard similarity of the found predicates and
        the correct predicate
    increment
        How frequently (in seconds) to log results when finding the best
        result in each increment.
    name
        The name of an experiment.
    file
        File name to output statistics from the run.
    return_viz
        If True, return an Altair visualization of the objective function
        with iteration on the x-axis.
    use_seeds_from_paper
        If True, use the seeds that were used in the paper. For reproducibility.

    Returns
    -------

    The input DataFrame filtered to contain all tuples that do not
    satisfy the explanation
    """

    return _drop_tuples_satisfying_optimal_predicate(
        data,
        f,
        num_cols,
        cat_cols,
        columns,
        cat_alg,
        n_trials,
        runtime,
        runs,
        k,
        random,
        correct_pred,
        increment,
        name,
        file,
        return_viz,
        use_seeds_from_paper,
        direction="minimize",
        **kwargs,
    )


def fmax(
    data,
    f,
    num_cols=[],
    cat_cols=[],
    columns=[],
    cat_alg=["individual_contribution"],
    n_trials=2000,
    runtime=10000,
    runs=1,
    k=5,
    random=False,
    correct_pred=None,
    increment=5,
    name="experiment_name",
    file=None,
    return_viz=False,
    use_seeds_from_paper=False,
    **kwargs,
):
    """
    Use BOExplain to maximize the objective function.

    Parameters
    ----------

    data
        pandas DataFrame of source, training, or inference data
        from which to derive an explanation.
    f
        Objective function to be minimized.
    num_cols
        Numerical columns over which to derive an explanation.
    cat_cols
        Categorical columns over which to derive an explanation.
    columns
        Columns over which to derive an explanation.
    cat_alg
        Algorithms to handle categorical parameters. Can be
            * 'individual_contribution'
            * 'categorical'
            * 'categorical_warm_start'
        See the paper for details.
    n_trials
        Maximum number of trials to perform during a run.
    runtime
        Maximum allowed time for a run in seconds.
    runs
        Number of runs to perform.
    k
        Number of TPE candidates to consider. (deprecated)
    random
        If True, perform a run using random search to
        find the constraint parameters.
    correct_pred
        If provided, will compute f-score, precision, recall,
        and jaccard similarity of the found predicates and
        the correct predicate
    increment
        How frequently (in seconds) to log results when finding the best
        result in each increment.
    name
        The name of an experiment.
    file
        File name to output statistics from the run.
    return_viz
        If True, return an Altair visualization of the objective function
        with iteration on the x-axis.
    use_seeds_from_paper
        If True, use the seeds that were used in the paper. For reproducibility.

    Returns
    -------

    The input DataFrame filtered to contain all tuples that do not
    satisfy the explanation
    """
    return _drop_tuples_satisfying_optimal_predicate(
        data,
        f,
        num_cols,
        cat_cols,
        columns,
        cat_alg,
        n_trials,
        runtime,
        runs,
        k,
        random,
        correct_pred,
        increment,
        name,
        file,
        return_viz,
        use_seeds_from_paper,
        direction="maximize",
        **kwargs,
    )


def _drop_tuples_satisfying_optimal_predicate(
    data,
    f,
    num_cols=[],
    cat_cols=[],
    columns=[],
    cat_alg=["individual_contribution"],
    n_trials=2000,
    runtime=10000,
    runs=1,
    k=5,
    random=False,
    correct_pred=None,
    increment=5,
    name="experiment_name",
    file=None,
    return_viz=False,
    use_seeds_from_paper=False,
    direction="minimize",
    **kwargs,
):
    assert direction == "minimize" or direction == "maximize"

    for col in columns:
        if is_numeric_dtype(data[col]):
            num_cols.append(col)
        else:
            cat_cols.append(col)

    # cast categorical columns as string type
    if cat_cols:
        data[cat_cols] = data[cat_cols].astype(str)

    # get the nuber of unique values in each column
    num_cols_range = [(data[col].min(), data[col].max()) for col in num_cols]
    cat_cols_n_uniq = [data[col].nunique() for col in cat_cols]

    # dataset length
    dataset_length = len(data)

    experiment = Experiment(
        num_cols,
        cat_cols,
        direction,
        n_trials,
        runs,
        correct_pred,
        name,
        file,
        num_cols_range,
        cat_cols_n_uniq,
        dataset_length,
        runtime,
        increment,
        use_seeds_from_paper,
    )

    cat_alg = [CAT_ALG_MAP[alg] for alg in cat_alg]

    for alg in cat_alg:
        stats = Stats(experiment, alg)
        cat_val_to_indiv_cont = {}
        if cat_cols and alg in {
            "individual_contribution_warm_start_topk",
            "categorical_warm_start",
            "individual_contribution_warm_start_top1",
        }:
            start = time.time()
            # encode categorical columns as numerical and record their encoding maps
            cat_val_to_indiv_cont = individual_contribution(
                data,
                objective=f,
                cat_cols=cat_cols,
                **kwargs,
            )
            run_encoding_time = time.time() - start
            # print(alg, run_encoding_time)
            stats.set_run_encoding_time(run_encoding_time)

        # initialize a TpeBo object
        tpebo = TpeBo(
            df=data,
            objective=f,
            num_cols=num_cols,
            cat_cols=cat_cols,
            direction=direction,
            k=k,
            cat_alg=alg,
            cat_val_to_indiv_cont=cat_val_to_indiv_cont,
            correct_pred=correct_pred,
        )
        # run the bayesian optimization
        df_rem = tpebo.run(stats, **kwargs)
        experiment.set_experiment(stats)

    if random:
        stats = Stats(experiment, None)
        tpebo = TpeBo(
            df=data,
            objective=f,
            num_cols=num_cols,
            cat_cols=cat_cols,
            direction=direction,
            k=k,
            cat_alg="random",
            cat_val_to_indiv_cont={},
            correct_pred=correct_pred,
        )
        df_rem = tpebo.random(stats, **kwargs)
        experiment.set_experiment(stats)

    viz = experiment.visualize_results()

    if file is not None:
        experiment.output_file()

    if return_viz:
        viz = experiment.visualize_results()
        return df_rem, viz

    return df_rem
