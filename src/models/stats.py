import scipy

import pandas as pd


def ks_test(sample: pd.DataFrame, population: pd.DataFrame):
    """
    Kolmogorov-Smirnov test for null hypothesis that the two samples came from
    the same distribution.
    """
    p_values = {}
    for col in population.columns:
        _, pval = scipy.stats.ks_2samp(
            sample[col], population[col], alternative="two-sided", 
            method="auto")
        p_values[col] = pval
    return p_values
