import pandas as pd
import numpy as np


def individual_contribution(df, objective, cat_cols, **kwargs):
    # dictionary of dictionaries, one dictionary for each column
    # dictinary keys are the categorical values and the values are the individual contribution
    # for each value in the column, compute the individual contribution of that column
    # ie, remove tuples satisfying the single-clause predicate 'col=val',
    # and compute the objective function with this data

    cat_val_to_indiv_cont = {
        col: {val: objective(df[df[col] != val], **kwargs) for val in df[col].unique()}
        for col in cat_cols
    }

    return cat_val_to_indiv_cont
