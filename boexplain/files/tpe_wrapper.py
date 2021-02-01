import time
from itertools import product

import numpy as np

from ..optuna.optuna.samplers import tpe
from ..optuna.optuna.study import create_study


class TpeBo(object):
    def __init__(
        self,
        df,
        objective,
        num_cols,
        cat_cols,
        direction,
        k,
        cat_alg,
        cat_val_to_indiv_cont,
        correct_pred,
    ):
        self.df = df
        self.objective = objective
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.dir_enc = 1 if direction == "minimize" else -1
        self.direction = direction
        self.k = k
        self.cat_alg = cat_alg
        self.correct_pred = correct_pred
        self.cat_val_to_indiv_cont = cat_val_to_indiv_cont
        self.n_warm_up = 10

        # domains or ranges of the categorical and numerical columns
        self.dom = {f"{c}_{str(f)[19:22]}": f(df[c]) for c in num_cols for f in {min, max}}
        if "individual_contribution_warm_start" in cat_alg and cat_cols:
            self.dom.update({f"{c}_min": 0 for c in cat_cols})
            self.dom.update({f"{c}_max": len(cat_val_to_indiv_cont[c]) - 1 for c in cat_cols})
        else:
            self.dom.update({c: list(df[c].dropna().unique()) for c in cat_cols})

        if "individual_contribution_warm_start" in cat_alg and cat_cols:
            self.code_to_cat_val, self.cat_val_to_code = {}, {}
            for col in cat_cols:
                ordered = dict(sorted(cat_val_to_indiv_cont[col].items(), key=lambda x: x[1]))
                self.code_to_cat_val[col] = dict(zip(range(len(ordered)), list(ordered.keys())))
                self.cat_val_to_code[col] = dict(zip(list(ordered.keys()), range(len(ordered))))

        if cat_cols:
            preds = self.df.groupby(list(cat_cols)).size().index
            if len(self.cat_cols) == 1:
                preds = [(pred,) for pred in preds]
            self.cat_preds = dict(zip(range(len(preds)), preds))
            self.cat_preds_set = set(preds)

            # for random
            self.uniq_cat_vals = {
                c: dict(zip(range(df[c].nunique()), df[c].unique())) for c in cat_cols
            }

            if (
                cat_alg
                in {
                    "individual_contribution_warm_start_topk",
                    "categorical_warm_start",
                    "individual_contribution_warm_start_top1",
                }
                and cat_cols
            ):
                grp_cont = [
                    (
                        grp,
                        sum([cat_val_to_indiv_cont[col][val] for col, val in zip(cat_cols, grp)]),
                    )
                    for grp in self.cat_preds_set
                ]
                grp_cont = sorted(grp_cont, key=lambda x: x[1], reverse=(direction == "maximize"))
                self.good_preds = [grp for (grp, _) in grp_cont[: self.n_warm_up]]

        if self.correct_pred:
            self.df_true = self.tuples_satisfying_pred(correct_pred)

    def candidates_to_pred(self, cnds):

        while self.run_stats["iter_cnt"] < self.n_warm_up:
            pred = {}

            if self.cat_cols:
                if self.cat_alg in {
                    "individual_contribution_warm_start_topk",
                    "categorical_warm_start",
                    "individual_contribution_warm_start_top1",
                }:
                    pred.update(zip(self.cat_cols, self.good_preds[self.run_stats["iter_cnt"]]))
                else:
                    pred.update(
                        zip(self.cat_cols, self.cat_preds[np.random.randint(len(self.cat_preds))])
                    )
                if not self.num_cols:  # don't do a data scan with only categorical columns
                    return pred

            for col in self.num_cols:
                if np.issubdtype(self.df[col].dtype, np.signedinteger) or np.issubdtype(
                    self.df[col].dtype, np.unsignedinteger
                ):
                    pred[f"{col}_min"] = np.random.randint(
                        cnds[f"{col}_min_dist"].low, cnds[f"{col}_min_dist"].high
                    )
                    pred[f"{col}_len"] = np.random.randint(
                        0, cnds[f"{col}_len_dist"].high - cnds[f"{col}_len_dist"].low
                    )
                elif np.issubdtype(self.df[col].dtype, np.floating):
                    pred[f"{col}_min"] = np.random.uniform(
                        cnds[f"{col}_min_dist"].low, cnds[f"{col}_min_dist"].high
                    )
                    pred[f"{col}_len"] = np.random.uniform(
                        0, cnds[f"{col}_len_dist"].high - cnds[f"{col}_len_dist"].low
                    )

            if self.is_valid(pred):
                return pred

        # all combinations of predicates
        preds = list(product(*[cnds[f"{name}_smpls"] for name in cnds["names"]]))
        # normalize
        for n in cnds["names"]:
            cnds[f"{n}_scrs"] = [scr / sum(cnds[f"{n}_scrs"]) for scr in cnds[f"{n}_scrs"]]
        # all multiplied combinations of scores
        scrs = [np.prod(scrs) for scrs in product(*[cnds[f"{n}_scrs"] for n in cnds["names"]])]
        # sorted list of (pred, score)
        pred_lst = sorted(list(zip(preds, scrs)), key=lambda x: x[1], reverse=True)

        for pred, _ in pred_lst:
            if self.cat_cols:
                if self.cat_alg == "individual_contribution_warm_start_topk":
                    pred = (
                        tuple(
                            [
                                self.code_to_cat_val[col][val]
                                for col, val in zip(self.cat_cols, pred[: len(self.cat_cols)])
                            ]
                        )
                        + pred[len(self.cat_cols) :]
                    )
                if pred[: len(self.cat_cols)] not in self.cat_preds_set:
                    continue
            pred = dict(zip(cnds["names"], pred))
            if self.is_valid(pred):
                return pred

        print("Random predicate")
        while True:
            pred = {}

            if self.cat_cols:
                pred.update(
                    zip(self.cat_cols, self.cat_preds[np.random.randint(len(self.cat_cols))])
                )
                if not self.num_cols:
                    return pred

            for col in self.num_cols:
                if np.issubdtype(self.df[col].dtype, np.signedinteger) or np.issubdtype(
                    self.df[col].dtype, np.unsignedinteger
                ):
                    pred[f"{col}_min"] = np.random.randint(
                        cnds[f"{col}_min_dist"].low, cnds[f"{col}_min_dist"].high
                    )
                    pred[f"{col}_len"] = np.random.randint(
                        0, cnds[f"{col}_len_dist"].high - cnds[f"{col}_len_dist"].low
                    )
                elif np.issubdtype(self.df[col].dtype, np.floating):
                    pred[f"{col}_min"] = np.random.uniform(
                        cnds[f"{col}_min_dist"].low, cnds[f"{col}_min_dist"].high
                    )
                    pred[f"{col}_len"] = np.random.uniform(
                        0, cnds[f"{col}_len_dist"].high - cnds[f"{col}_len_dist"].low
                    )

            if self.is_valid(pred):
                return pred

    def objective_wrapper(self, trial, **kwargs):

        if time.time() - self.add_on - self.start >= (self.step + 1) * self.incr:
            self.time_array[self.step] = self.opt_res[self.run_stats["iter_cnt"] - 1]
            if self.correct_pred:
                self.precision_time_array[self.step] = self.run_stats["precision"][
                    self.run_stats["iter_cnt"] - 1
                ]
                self.recall_time_array[self.step] = self.run_stats["recall"][
                    self.run_stats["iter_cnt"] - 1
                ]
                self.f_score_time_array[self.step] = self.run_stats["f_score"][
                    self.run_stats["iter_cnt"] - 1
                ]
                self.jaccard_time_array[self.step] = self.run_stats["jaccard"][
                    self.run_stats["iter_cnt"] - 1
                ]
            self.step += 1

        cnds = {"names": []}  # candidate predicate constraint values dict
        dom = self.dom  # parameter domains
        pred = {}

        for c in self.cat_cols:
            if self.cat_alg in {"categorical", "categorical_warm_start", "categorical_topk"}:
                param_cnds = trial.suggest_categorical(c, dom[c])
            elif "individual_contribution_warm_start" in self.cat_alg:
                param_cnds = trial.suggest_int(c, dom[f"{c}_min"], dom[f"{c}_max"])
            cnds.update(zip([f"{c}_{x}" for x in ("smpls", "scrs", "dist")], param_cnds[2:5]))
            cnds["names"].append(param_cnds[1])
            pred[c] = param_cnds[0]

        for c in self.num_cols:
            if np.issubdtype(self.df[c].dtype, np.signedinteger) or np.issubdtype(
                self.df[c].dtype, np.unsignedinteger
            ):
                min_cnds = trial.suggest_int(f"{c}_min", dom[f"{c}_min"], dom[f"{c}_max"])
                len_cnds = trial.suggest_int(f"{c}_len", 0, dom[f"{c}_max"] - dom[f"{c}_min"])
            elif np.issubdtype(self.df[c].dtype, np.floating):
                min_cnds = trial.suggest_uniform(f"{c}_min", dom[f"{c}_min"], dom[f"{c}_max"])
                len_cnds = trial.suggest_uniform(f"{c}_len", 0, dom[f"{c}_max"] - dom[f"{c}_min"])
            cnds.update(zip([f"{c}_min_{x}" for x in ("smpls", "scrs", "dist")], min_cnds[2:5]))
            cnds.update(zip([f"{c}_len_{x}" for x in ("smpls", "scrs", "dist")], len_cnds[2:5]))
            cnds["names"].extend([min_cnds[1], len_cnds[1]])
            pred.update({f"{c}_min": min_cnds[0], f"{c}_len": len_cnds[0]})

        if self.cat_cols and (
            self.cat_alg == "categorical_topk"
            or self.cat_alg == "individual_contribution_warm_start_topk"
            or self.cat_alg == "individual_contribution_warm_start_top1"
            and self.run_stats["iter_cnt"] < self.n_warm_up
            or self.cat_alg == "categorical_warm_start"
            and self.run_stats["iter_cnt"] < self.n_warm_up
        ):
            # begin_metric_comps = time.time()
            pred = self.candidates_to_pred(cnds)
            # self.add_on += time.time() - begin_metric_comps

        if self.cat_cols and (
            self.cat_alg == "individual_contribution_warm_start_topk"
            or self.cat_alg == "individual_contribution_warm_start_top1"
            and self.run_stats["iter_cnt"] < self.n_warm_up
        ):
            trial.update_params(
                {
                    k: (self.cat_val_to_code[k][v] if k in self.cat_cols else v)
                    for k, v in pred.items()
                }
            )
        else:
            trial.update_params(pred)

        if (
            self.cat_cols
            and self.cat_alg == "individual_contribution_warm_start_top1"
            and self.run_stats["iter_cnt"] >= self.n_warm_up
        ):
            pred = {
                k: (self.code_to_cat_val[k][v] if k in self.cat_cols else v)
                for k, v in pred.items()
            }

        # keep track of repeated values
        if (dpl_check := tuple(pred.items())) not in self.run_stats["dpls"]:
            self.run_stats["dpls"].add(dpl_check)
        else:
            self.run_stats["dpl_cnt"] += 1
        # print(pred)
        df_rem_pred = self.drop_tuples_satisfying_pred(pred)

        # if no tuples satisfy the predicate, return. This should never execute
        if len(df_rem_pred) == len(self.df):
            self.run_stats["zero_tup_cnt"] += 1
            self.opt_res[self.run_stats["iter_cnt"]] = self.opt_res[self.run_stats["iter_cnt"] - 1]
            if self.correct_pred:
                self.run_stats["precision"][self.run_stats["iter_cnt"]] = self.run_stats[
                    "precision"
                ][self.run_stats["iter_cnt"] - 1]
                self.run_stats["recall"][self.run_stats["iter_cnt"]] = self.run_stats["recall"][
                    self.run_stats["iter_cnt"] - 1
                ]
                self.run_stats["f_score"][self.run_stats["iter_cnt"]] = self.run_stats["f_score"][
                    self.run_stats["iter_cnt"] - 1
                ]
                self.run_stats["jaccard"][self.run_stats["iter_cnt"]] = self.run_stats["jaccard"][
                    self.run_stats["iter_cnt"] - 1
                ]
            self.run_stats["iter_cnt"] += 1
            return self.dir_enc * 1e9

        result = self.objective(df_rem_pred, **kwargs)

        # record results
        if (
            self.dir_enc == 1
            and result < self.opt_res.min()
            or self.dir_enc == -1
            and result > self.opt_res.max()
        ):
            self.opt_res[self.run_stats["iter_cnt"]] = result
            self.run_stats["best_pred"] = pred
        else:
            self.opt_res[self.run_stats["iter_cnt"]] = self.opt_res[self.run_stats["iter_cnt"] - 1]
        if self.correct_pred:
            begin_metric_comps = time.time()
            df_only_pred = self.tuples_satisfying_pred(pred)

            if (prec := self.precision(df_only_pred)) >= self.run_stats["precision"].max():
                self.run_stats["precision"][self.run_stats["iter_cnt"]] = prec
            else:
                self.run_stats["precision"][self.run_stats["iter_cnt"]] = self.run_stats[
                    "precision"
                ][self.run_stats["iter_cnt"] - 1]
            if (recall := self.recall(df_only_pred)) >= self.run_stats["recall"].max():
                self.run_stats["recall"][self.run_stats["iter_cnt"]] = recall
            else:
                self.run_stats["recall"][self.run_stats["iter_cnt"]] = self.run_stats["recall"][
                    self.run_stats["iter_cnt"] - 1
                ]
            if (f_score := self.f_score(df_only_pred)) >= self.run_stats["f_score"].max():
                self.run_stats["f_score"][self.run_stats["iter_cnt"]] = f_score
            else:
                self.run_stats["f_score"][self.run_stats["iter_cnt"]] = self.run_stats["f_score"][
                    self.run_stats["iter_cnt"] - 1
                ]
            if (jaccard := self.jaccard(df_only_pred)) >= self.run_stats["jaccard"].max():
                self.run_stats["jaccard"][self.run_stats["iter_cnt"]] = jaccard
            else:
                self.run_stats["jaccard"][self.run_stats["iter_cnt"]] = self.run_stats["jaccard"][
                    self.run_stats["iter_cnt"] - 1
                ]
            self.add_on += time.time() - begin_metric_comps
            self.study.add_on = self.add_on

        self.run_stats["iter_cnt"] += 1

        return result

    def run(self, stats, **kwargs):

        for i in range(stats.runs):

            self.new_run_stats(stats.n_trials)
            self.opt_res = stats.get_run_opt_res_array()
            np.random.seed(stats.seeds[i])

            self.incr = stats.increment
            self.time_array = stats.get_run_time_array()
            self.precision_time_array = stats.get_run_time_array()
            self.recall_time_array = stats.get_run_time_array()
            self.f_score_time_array = stats.get_run_time_array()
            self.jaccard_time_array = stats.get_run_time_array()
            self.step = int(stats.encoding_time // self.incr)
            self.time_array[: self.step] = np.nan
            self.precision_time_array[: self.step] = np.nan
            self.recall_time_array[: self.step] = np.nan
            self.f_score_time_array[: self.step] = np.nan
            self.jaccard_time_array[: self.step] = np.nan
            self.add_on = 0

            # minimize the objective over the space
            self.study = create_study(
                sampler=tpe.TPESampler(seed=stats.seeds[i], k=self.k),
                direction=self.direction,
                seed=stats.seeds[i],
            )
            self.start = time.time() - stats.encoding_time
            self.study.optimize(
                self.objective_wrapper,
                n_trials=stats.n_trials,
                timeout=stats.runtime - stats.encoding_time,
                **kwargs,
            )
            stats.set_run_time(time.time() - self.start, i)

            self.time_array[len(self.time_array) - 1] = self.opt_res[
                self.run_stats["iter_cnt"] - 1
            ]
            self.precision_time_array[len(self.precision_time_array) - 1] = self.run_stats[
                "precision"
            ][self.run_stats["iter_cnt"] - 1]
            self.recall_time_array[len(self.recall_time_array) - 1] = self.run_stats["recall"][
                self.run_stats["iter_cnt"] - 1
            ]
            self.f_score_time_array[len(self.f_score_time_array) - 1] = self.run_stats["f_score"][
                self.run_stats["iter_cnt"] - 1
            ]
            self.jaccard_time_array[len(self.jaccard_time_array) - 1] = self.run_stats["jaccard"][
                self.run_stats["iter_cnt"] - 1
            ]

            output = []
            for col in self.cat_cols:
                output.append(f"{col} = \"{self.run_stats['best_pred'][col]}\"")
            for col in self.num_cols:
                minv, leng = (
                    self.run_stats["best_pred"][f"{col}_min"],
                    self.run_stats["best_pred"][f"{col}_len"],
                )
                maxv = min(minv + leng, self.df[col].max())
                output.append(f"{minv} <= {col} <= {maxv}")
            print("Predicate:", " and ".join(output))

            best_params = self.run_stats["best_pred"]

            stats.set_run_n_duplicates(self.run_stats["dpl_cnt"], i)
            stats.set_run_n_zero_tup_preds(self.run_stats["zero_tup_cnt"], i)
            stats.set_run_opt_res(self.opt_res, i)
            stats.set_run_preds(best_params, i)
            stats.set_run_iter_completed(self.run_stats["iter_cnt"], i)
            stats.set_min_iter_completed(self.run_stats["iter_cnt"])

            stats.set_run_best_objective_value(self.study.best_value, i)
            stats.set_example_best_predicate(best_params, i)

            df_rem_pred = self.drop_tuples_satisfying_pred(best_params)
            stats.set_run_n_tuples_removed_from_data(len(self.df) - len(df_rem_pred), i)

            if self.correct_pred:
                df_only_pred = self.tuples_satisfying_pred(best_params)

                stats.set_precision(self.run_stats["precision"], i)
                stats.set_recall(self.run_stats["recall"], i)
                stats.set_f_score(self.run_stats["f_score"], i)
                stats.set_jaccard(self.run_stats["jaccard"], i)

                stats.set_final_precision(self.precision(df_only_pred), i)
                stats.set_final_recall(self.recall(df_only_pred), i)
                stats.set_final_f_score(self.f_score(df_only_pred), i)
                stats.set_final_jaccard(self.jaccard(df_only_pred), i)

                stats.set_precision_time_array(self.precision_time_array, i)
                stats.set_recall_time_array(self.recall_time_array, i)
                stats.set_f_score_time_array(self.f_score_time_array, i)
                stats.set_jaccard_time_array(self.jaccard_time_array, i)

            stats.set_run_time_array(self.time_array, i)
            stats.set_add_on(self.add_on, i)

            # stats.output_temp_file()

        # stats.standard_output()

        return self.drop_tuples_satisfying_pred(stats.example_best_predicate)

    def random(self, stats, **kwargs):

        for i in range(stats.runs):

            self.new_run_stats(stats.n_trials)
            opt_res = stats.get_run_opt_res_array()
            np.random.seed(stats.seeds[i])
            best_pred, best_value = None, None

            # time for averaging
            start = time.time()
            add_on = 0  # time for computing the metrics

            incr, step = stats.increment, 0
            time_array = stats.get_run_time_array()
            precision_time_array = stats.get_run_time_array()
            recall_time_array = stats.get_run_time_array()
            f_score_time_array = stats.get_run_time_array()
            jaccard_time_array = stats.get_run_time_array()

            for _ in range(stats.n_trials):

                if time.time() - add_on - start >= (step + 1) * incr:
                    time_array[step] = opt_res[self.run_stats["iter_cnt"] - 1]
                    if self.correct_pred:
                        precision_time_array[step] = self.run_stats["precision"][
                            self.run_stats["iter_cnt"] - 1
                        ]
                        recall_time_array[step] = self.run_stats["recall"][
                            self.run_stats["iter_cnt"] - 1
                        ]
                        f_score_time_array[step] = self.run_stats["f_score"][
                            self.run_stats["iter_cnt"] - 1
                        ]
                        jaccard_time_array[step] = self.run_stats["jaccard"][
                            self.run_stats["iter_cnt"] - 1
                        ]

                    step += 1
                    if step >= len(time_array):
                        break

                pred = {}
                for col in self.cat_cols:
                    pred[col] = self.uniq_cat_vals[col][
                        np.random.randint(len(self.uniq_cat_vals[col]))
                    ]
                for col in self.num_cols:
                    if np.issubdtype(self.df[col].dtype, np.signedinteger) or np.issubdtype(
                        self.df[col].dtype, np.unsignedinteger
                    ):
                        pred[f"{col}_min"] = np.random.randint(
                            self.dom[f"{col}_min"], self.dom[f"{col}_max"]
                        )
                        pred[f"{col}_len"] = np.random.randint(
                            0, self.dom[f"{col}_max"] - self.dom[f"{col}_min"]
                        )
                    elif np.issubdtype(self.df[col].dtype, np.floating):
                        pred[f"{col}_min"] = np.random.uniform(
                            self.dom[f"{col}_min"], self.dom[f"{col}_max"]
                        )
                        pred[f"{col}_len"] = np.random.uniform(
                            0, self.dom[f"{col}_max"] - self.dom[f"{col}_min"]
                        )

                if (dpl_check := tuple(pred.items())) not in self.run_stats["dpls"]:
                    self.run_stats["dpls"].add(dpl_check)
                else:
                    self.run_stats["dpl_cnt"] += 1

                df_rem_pred = self.drop_tuples_satisfying_pred(pred)

                if len(df_rem_pred) == len(self.df):
                    self.run_stats["zero_tup_cnt"] += 1
                    opt_res[self.run_stats["iter_cnt"]] = opt_res[self.run_stats["iter_cnt"] - 1]
                    if self.correct_pred:
                        self.run_stats["precision"][self.run_stats["iter_cnt"]] = self.run_stats[
                            "precision"
                        ][self.run_stats["iter_cnt"] - 1]
                        self.run_stats["recall"][self.run_stats["iter_cnt"]] = self.run_stats[
                            "recall"
                        ][self.run_stats["iter_cnt"] - 1]
                        self.run_stats["f_score"][self.run_stats["iter_cnt"]] = self.run_stats[
                            "f_score"
                        ][self.run_stats["iter_cnt"] - 1]
                        self.run_stats["jaccard"][self.run_stats["iter_cnt"]] = self.run_stats[
                            "jaccard"
                        ][self.run_stats["iter_cnt"] - 1]
                    self.run_stats["iter_cnt"] += 1
                    continue

                result = self.objective(df_rem_pred, **kwargs)

                # record results
                if (
                    self.dir_enc == 1
                    and result < opt_res[self.run_stats["iter_cnt"] - 1]
                    or self.dir_enc == -1
                    and result > opt_res[self.run_stats["iter_cnt"] - 1]
                ):
                    opt_res[self.run_stats["iter_cnt"]] = result
                    best_pred, best_value = pred, result
                else:
                    opt_res[self.run_stats["iter_cnt"]] = opt_res[self.run_stats["iter_cnt"] - 1]
                if self.correct_pred:
                    begin_metrics_comps = time.time()
                    df_only_pred = self.tuples_satisfying_pred(pred)

                    if (prec := self.precision(df_only_pred)) >= self.run_stats["precision"].max():
                        self.run_stats["precision"][self.run_stats["iter_cnt"]] = prec
                    else:
                        self.run_stats["precision"][self.run_stats["iter_cnt"]] = self.run_stats[
                            "precision"
                        ][self.run_stats["iter_cnt"] - 1]

                    if (recall := self.recall(df_only_pred)) >= self.run_stats["recall"].max():
                        self.run_stats["recall"][self.run_stats["iter_cnt"]] = recall
                    else:
                        self.run_stats["recall"][self.run_stats["iter_cnt"]] = self.run_stats[
                            "recall"
                        ][self.run_stats["iter_cnt"] - 1]

                    if (f_score := self.f_score(df_only_pred)) >= self.run_stats["f_score"].max():
                        self.run_stats["f_score"][self.run_stats["iter_cnt"]] = f_score
                    else:
                        self.run_stats["f_score"][self.run_stats["iter_cnt"]] = self.run_stats[
                            "f_score"
                        ][self.run_stats["iter_cnt"] - 1]

                    if (jaccard := self.jaccard(df_only_pred)) >= self.run_stats["jaccard"].max():
                        self.run_stats["jaccard"][self.run_stats["iter_cnt"]] = jaccard
                    else:
                        self.run_stats["jaccard"][self.run_stats["iter_cnt"]] = self.run_stats[
                            "jaccard"
                        ][self.run_stats["iter_cnt"] - 1]
                    add_on += time.time() - begin_metrics_comps

                self.run_stats["iter_cnt"] += 1

            time_array[len(time_array) - 1] = opt_res[self.run_stats["iter_cnt"] - 1]
            precision_time_array[len(precision_time_array) - 1] = self.run_stats["precision"][
                self.run_stats["iter_cnt"] - 1
            ]
            recall_time_array[len(recall_time_array) - 1] = self.run_stats["recall"][
                self.run_stats["iter_cnt"] - 1
            ]
            f_score_time_array[len(f_score_time_array) - 1] = self.run_stats["f_score"][
                self.run_stats["iter_cnt"] - 1
            ]
            jaccard_time_array[len(jaccard_time_array) - 1] = self.run_stats["jaccard"][
                self.run_stats["iter_cnt"] - 1
            ]

            stats.set_run_time(time.time() - start, i)
            stats.set_run_n_duplicates(self.run_stats["dpl_cnt"], i)
            stats.set_run_n_zero_tup_preds(self.run_stats["zero_tup_cnt"], i)
            stats.set_run_opt_res(opt_res, i)
            stats.set_run_preds(best_pred, i)
            stats.set_run_iter_completed(self.run_stats["iter_cnt"], i)
            stats.set_min_iter_completed(self.run_stats["iter_cnt"])

            stats.set_run_best_objective_value(best_value, i)
            stats.set_example_best_predicate(best_pred, i)

            df_rem_pred = self.drop_tuples_satisfying_pred(best_pred)
            stats.set_run_n_tuples_removed_from_data(len(self.df) - len(df_rem_pred), i)

            if self.correct_pred:
                df_only_pred = self.tuples_satisfying_pred(best_pred)

                stats.set_precision(self.run_stats["precision"], i)
                stats.set_recall(self.run_stats["recall"], i)
                stats.set_f_score(self.run_stats["f_score"], i)
                stats.set_jaccard(self.run_stats["jaccard"], i)

                stats.set_final_precision(self.precision(df_only_pred), i)
                stats.set_final_recall(self.recall(df_only_pred), i)
                stats.set_final_f_score(self.f_score(df_only_pred), i)
                stats.set_final_jaccard(self.jaccard(df_only_pred), i)

                stats.set_precision_time_array(precision_time_array, i)
                stats.set_recall_time_array(recall_time_array, i)
                stats.set_f_score_time_array(f_score_time_array, i)
                stats.set_jaccard_time_array(jaccard_time_array, i)

            stats.set_run_time_array(time_array, i)
            stats.set_add_on(add_on, i)

            # stats.output_temp_file()

        # stats.standard_output()

        return self.drop_tuples_satisfying_pred(stats.example_best_predicate)

    def drop_tuples_satisfying_pred(self, pred):
        # list of constrained dataframes for doing boolean logic over
        cstrs = [self.df[col] < pred[f"{col}_min"] for col in self.num_cols]
        cstrs += [self.df[col] > pred[f"{col}_min"] + pred[f"{col}_len"] for col in self.num_cols]
        cstrs += [self.df[col] != pred[col] for col in self.cat_cols]
        # this dataframe has removed all tuples that satisfy the predicate
        return self.df[np.logical_or.reduce(cstrs)]

    def tuples_satisfying_pred(self, pred):
        # list of constrained dataframes for doing boolean logic over
        cstrs = [self.df[col] >= pred[f"{col}_min"] for col in self.num_cols]
        cstrs += [self.df[col] <= pred[f"{col}_min"] + pred[f"{col}_len"] for col in self.num_cols]
        cstrs += [self.df[col] == pred[col] for col in self.cat_cols]
        # this dataframe has only all tuples that satisfy the predicate
        return self.df[np.logical_and.reduce(cstrs)]

    def is_valid(self, pred):
        return 0 < len(self.tuples_satisfying_pred(pred)) < len(self.df)

    # def drop_tuples_satisfying_pred(self, pred):
    #     # numerical constraints
    #     num = [
    #         f"`{col}` < {pred[f'{col}_min']} | `{col}` > {pred[f'{col}_min'] + pred[f'{col}_len']}"
    #         for col in self.num_cols
    #     ]
    #     # categorical constraints
    #     cat = [f'`{col}` != "{pred[col]}"' for col in self.cat_cols]
    #     # return tuples that satisfy the predicate
    #     return self.df.query(" | ".join(num + cat))

    # def tuples_satisfying_pred(self, pred):
    #     # numerical constraints
    #     num = [
    #         f"`{col}` >= {pred[f'{col}_min']} & `{col}` <= {pred[f'{col}_min'] + pred[f'{col}_len']}"
    #         for col in self.num_cols
    #     ]
    #     # categorical constraints
    #     cat = [f'`{col}` == "{pred[col]}"' for col in self.cat_cols]
    #     # return tuples that satisfy the predicate
    #     return self.df.query(" & ".join(num + cat))

    def precision(self, df_only_pred):
        # precision
        return len(set(df_only_pred.index).intersection(set(self.df_true.index))) / len(
            df_only_pred.index
        )

    def recall(self, df_only_pred):
        # recall
        return len(set(df_only_pred.index).intersection(set(self.df_true.index))) / len(
            self.df_true.index
        )

    def f_score(self, df_only_pred):
        # precision
        prec = len(set(df_only_pred.index).intersection(set(self.df_true.index))) / len(
            df_only_pred.index
        )
        # recall
        rec = len(set(df_only_pred.index).intersection(set(self.df_true.index))) / len(
            self.df_true.index
        )
        return 2 * prec * rec / (prec + rec + 1e-9)

    def jaccard(self, df_only_pred):

        return len(set(df_only_pred.index).intersection(set(self.df_true.index))) / len(
            set(df_only_pred.index).union(set(self.df_true.index))
        )

    def new_run_stats(self, n_trials):

        self.run_stats = {
            "iter_cnt": 0,
            "dpls": set(),
            "dpl_cnt": 0,
            "zero_tup_cnt": 0,
            "precision": np.zeros(n_trials),
            "recall": np.zeros(n_trials),
            "f_score": np.zeros(n_trials),
            "jaccard": np.zeros(n_trials),
        }
