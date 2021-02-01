from typing import Any
import re
import random
import numpy as np
import pandas as pd
import altair as alt

alt.data_transformers.disable_max_rows()
from json import dumps
from numpyencoder import NumpyEncoder


class Experiment:

    experiments = dict()
    n_exp = 0

    def __init__(
        self,
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
    ):

        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.direction = direction
        self.dir_enc = 1 if direction == "minimize" else -1
        self.n_trials = n_trials
        self.runs = runs
        self.correct_pred = correct_pred
        self.name = name
        self.file = file
        self.num_cols_range = num_cols_range
        self.cat_cols_n_uniq = cat_cols_n_uniq
        self.dataset_length = dataset_length
        self.runtime = runtime
        self.increment = increment

        if use_seeds_from_paper:
            self.seeds = [
                529840,
                664234,
                978546,
                283991,
                819362,
                348229,
                536289,
                480291,
                500927,
                386602,
            ]
        else:
            self.seeds = random.sample(range(1000000), runs)

    def set_experiment(self, results) -> None:

        self.experiments[self.n_exp] = results.__dict__.copy()
        self.n_exp += 1

    def output_file(self):

        fo = open(self.file, "w")

        for v in self.experiments.values():
            fo.write(f"{dumps(v, cls=NumpyEncoder)}\n")

        fo.close()

    def visualize_results(self):

        df = pd.DataFrame({}, columns=["Algorithm", "Iteration", "Value"])
        for i in range(len(self.experiments)):
            df_new = pd.DataFrame.from_dict(
                {
                    "Algorithm": self.experiments[i]["cat_enc"],
                    "Iteration": list(range(self.experiments[i]["n_trials"])),
                    "Value": self.experiments[i]["opt_res"],
                },
                orient="index",
            ).T
            df = df.append(df_new)
        df = df.explode("Value")
        df = df.set_index(["Algorithm"]).apply(pd.Series.explode).reset_index()

        num_cols = f"{len(self.experiments[0]['num_cols'])} numerical columns: "
        for i, col in enumerate(self.experiments[0]["num_cols"]):
            num_cols += f"{col} (range {self.experiments[0]['num_cols_range'][i][0]} to {self.experiments[0]['num_cols_range'][i][1]}), "
        cat_cols = f"{len(self.experiments[0]['cat_cols'])} categorical columns: "
        for i, col in enumerate(self.experiments[0]["cat_cols"]):
            cat_cols += f"{col} ({self.experiments[0]['cat_cols_n_uniq'][i]} unique values), "

        out_str = f"Experiment: {self.experiments[0]['name']}. Completed {self.experiments[0]['n_trials']} iterations for {self.experiments[0]['runs']} runs. Search space includes "

        if len(self.experiments[0]["num_cols"]) > 0:
            out_str += num_cols
            if len(self.experiments[0]["cat_cols"]) > 0:
                out_str += "and "

        if len(self.experiments[0]["cat_cols"]) > 0:
            out_str += cat_cols

        out_str = f"{out_str[:-2]}."

        out_lst = [line.strip() for line in re.findall(r".{1,80}(?:\s+|$)", out_str)]

        line = (
            alt.Chart(df)
            .mark_line()
            .encode(
                x="Iteration",
                y=alt.Y("mean(Value)", scale=alt.Scale(zero=False)),
                color="Algorithm",
            )
            .properties(title=out_lst)  # {"text": out_lst, "subtitle": ""}
        )
        band = (
            alt.Chart(df)
            .mark_errorband(extent="stdev")
            .encode(
                x="Iteration",
                y=alt.Y("Value", title="Mean Objective Function Value"),
                color="Algorithm",
            )
        )
        chart = band + line
        chart = chart.configure_title(
            anchor="start",
        )
        return chart


class Stats(Experiment):
    def __init__(self, experiment, cat_enc) -> None:
        self.__dict__ = experiment.__dict__
        self.cat_enc = cat_enc

        self.run_times = np.zeros(self.runs)
        self.n_duplicates = np.zeros(self.runs)
        self.n_zero_tup_preds = np.zeros(self.runs)
        self.preds = dict()
        self.opt_res = np.full((self.runs, self.n_trials), self.dir_enc * 1e9)
        self.run_time_of_opt_res = np.zeros((self.runs, self.n_trials))
        self.iter_completed = np.zeros(self.runs)
        self.min_iter_completed = self.n_trials
        self.n_tuples_removed_from_data = np.zeros(self.runs)
        self.best_obj_values = np.full(self.runs, self.dir_enc * 1e9)
        self.add_on = np.zeros(self.runs)

        if self.correct_pred:
            self.precision = np.zeros((self.runs, self.n_trials))
            self.recall = np.zeros((self.runs, self.n_trials))
            self.f_score = np.zeros((self.runs, self.n_trials))
            self.jaccard = np.zeros((self.runs, self.n_trials))

            self.final_precision = np.zeros(self.runs)
            self.final_recall = np.zeros(self.runs)
            self.final_f_score = np.zeros(self.runs)
            self.final_jaccard = np.zeros(self.runs)

        self.encoding_time = 0
        self.example_best_predicate = None

        self.time_array = np.zeros((self.runs, self.runtime // self.increment))
        self.precision_time_array = np.zeros((self.runs, self.runtime // self.increment))
        self.recall_time_array = np.zeros((self.runs, self.runtime // self.increment))
        self.f_score_time_array = np.zeros((self.runs, self.runtime // self.increment))
        self.jaccard_time_array = np.zeros((self.runs, self.runtime // self.increment))

    def get_run_opt_res_array(self) -> np.ndarray:
        return np.full(self.n_trials, self.dir_enc * 1e9)

    def get_run_time_array(self) -> np.ndarray:
        return np.zeros(self.runtime // self.increment)

    def get_run_time_of_opt_res_array(self) -> np.ndarray:
        return np.zeros(self.n_trials)

    def set_run_encoding_time(self, run_encoding_time):

        self.encoding_time = run_encoding_time

    def set_run_opt_res(self, run_opt_res: np.ndarray, run: int) -> None:

        self.opt_res[run] = run_opt_res

    def set_run_time_array(self, run_time_array: np.ndarray, run: int) -> None:

        self.time_array[run] = run_time_array

    def set_precision_time_array(self, precision_time_array: np.ndarray, run: int) -> None:

        self.precision_time_array[run] = precision_time_array

    def set_recall_time_array(self, recall_time_array: np.ndarray, run: int) -> None:

        self.recall_time_array[run] = recall_time_array

    def set_f_score_time_array(self, f_score_time_array: np.ndarray, run: int) -> None:

        self.f_score_time_array[run] = f_score_time_array

    def set_jaccard_time_array(self, jaccard_time_array: np.ndarray, run: int) -> None:

        self.jaccard_time_array[run] = jaccard_time_array

    def set_run_time_of_opt_res(self, run_time_opt_res: np.ndarray, run: int) -> None:

        self.run_time_of_opt_res[run] = run_time_opt_res

    def set_run_time(self, run_time: float, run: int) -> None:

        self.run_times[run] = run_time

    def set_add_on(self, add_on: float, run: int) -> None:

        self.add_on[run] = add_on

    def set_run_n_duplicates(self, run_n_dups: float, run: int) -> None:

        self.n_duplicates[run] = run_n_dups

    def set_run_n_zero_tup_preds(self, run_n_zero_tup_preds: float, run: int) -> None:

        self.n_zero_tup_preds[run] = run_n_zero_tup_preds

    def set_run_preds(self, best_pred: dict[Any], run: int) -> None:

        self.preds[run] = best_pred

    def set_run_iter_completed(self, n_iter: int, run) -> None:

        self.iter_completed[run] = n_iter

    def set_run_best_objective_value(self, obj_value: int, run) -> None:

        self.best_obj_values[run] = obj_value

    def set_example_best_predicate(self, best_pred: dict[Any], run) -> None:

        if self.direction == "minimize":
            if self.best_obj_values[run] == self.best_obj_values.min():
                self.example_best_predicate = best_pred
        else:
            if self.best_obj_values[run] == self.best_obj_values.max():
                self.example_best_predicate = best_pred

    def set_min_iter_completed(self, n_iter: int) -> None:

        if n_iter < self.min_iter_completed:
            self.min_iter_completed = n_iter

    def set_run_n_tuples_removed_from_data(self, num_removed: int, run: int):

        self.n_tuples_removed_from_data[run] = num_removed

    def set_final_precision(self, precision: float, run: int) -> None:

        self.final_precision[run] = precision

    def set_final_recall(self, recall: float, run: int) -> None:

        self.final_recall[run] = recall

    def set_final_f_score(self, f_score: float, run: int) -> None:

        self.final_f_score[run] = f_score

    def set_final_jaccard(self, jaccard: float, run: int) -> None:

        self.final_jaccard[run] = jaccard

    def set_precision(self, precision: np.ndarray, run: int) -> None:

        self.precision[run] = precision

    def set_recall(self, recall: np.ndarray, run: int) -> None:

        self.recall[run] = recall

    def set_f_score(self, f_score: np.ndarray, run: int) -> None:

        self.f_score[run] = f_score

    def set_jaccard(self, jaccard: np.ndarray, run: int) -> None:

        self.jaccard[run] = jaccard

    def output_temp_file(self) -> None:

        fo = open("temp.json", "w")

        fo.write(f"{dumps(self.__dict__, cls=NumpyEncoder)}\n")

        fo.close()

    def standard_output(self) -> None:

        print("BEST SCORE", self.best_obj_values)
        print("AVERAGE NUMBER OF TUPLES REMOVED", self.n_tuples_removed_from_data.mean())
        print("AVERAGE TIME", self.run_times.mean())
        print("AVERAGE DUPLICATE COUNT", self.n_duplicates.mean())
        print("AVERAGE ZERO TUPLE", self.n_zero_tup_preds.mean(), "\n")
