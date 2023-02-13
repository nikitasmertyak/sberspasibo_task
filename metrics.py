"""
Model evaluation functions.
"""

import pandas as pd


def calc_metrics(column: str, interactions: pd.DataFrame) -> float:
    """
    Calculating the quality of the model

    Args:
        column : str
                 Name of the using model 
        interactions: DataFrame with true_train and true_test information
    Returns
         Mean Average Precision score.
    """
    
    return (
        interactions
        .apply(
            lambda row:
            len(set(row['true_test']).intersection(
                set(row[column]))) /
            min(len(row['true_test']) + 0.001, 10.0),
            axis=1)).mean()
