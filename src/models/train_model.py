"""Here goes the training code."""
import itertools

import scipy
import numpy as np
import pandas as pd

from src.features.build_features import standardize


def fit_distribution(df, column, pct, pct_lower):
    """
    Fit distribution for Optimal Stopping Point Prediction.

    This function helps to list out the chi-square statistics for each
    distribution and further sorts them to find the best distribution.

    Returns a table that contains sorted chi-square values as well as
    the parameters such as mu (shape), loc (location) and scale for each
    distribution.
    """
    # Set up list of candidate distributions to use
    y_std, size, y_org = standardize(df, column, pct, pct_lower)
    dist_names = [
        "weibull_min",
        "norm",
        "weibull_max",
        "beta",
        "invgauss",
        "uniform",
        "gamma",
        "expon",
        "lognorm",
        "pearson3",
        "triang",
    ]

    chi_square_statistics = []

    # 50 bins
    percentile_bins = np.linspace(0, 100, 50)
    percentile_cutoffs = np.percentile(y_std, percentile_bins)
    observed_frequency, bins = np.histogram(y_std, bins=percentile_cutoffs)
    cum_observed_frequency = np.cumsum(observed_frequency)
    # Data frame to store results
    dist_param = pd.DataFrame()
    dist_param["Distribution Names"] = dist_names
    param_list = []

    # Loop through candidate distributions
    for distribution in dist_names:
        # Set up distribution and get fitted distribution parameters
        dist = getattr(scipy.stats, distribution)
        param = dist.fit(y_std)
        param_list.append(param)

        # Get expected counts in percentile bins
        # cdf of fitted distribution across bins
        cdf_fitted = dist.cdf(percentile_cutoffs, *param)
        expected_frequency = []
        for bin in range(len(percentile_bins) - 1):
            expected_cdf_area = cdf_fitted[bin + 1] - cdf_fitted[bin]
            expected_frequency.append(expected_cdf_area)

        # Chi-square Statistics
        expected_frequency = np.array(expected_frequency) * size
        cum_expected_frequency = np.cumsum(expected_frequency)
        ss = scipy.stats.chisquare(
            f_obs=cum_observed_frequency, f_exp=cum_expected_frequency
        )
        chi_square_statistics.append(ss)

    # Append results to data frame
    dist_param["Parameters"] = param_list
    dist_param.set_index("Distribution Names")
    # Sort by minimum ch-square statistics
    results = pd.DataFrame()
    results["Distribution"] = dist_names
    results["chi_square and p-value"] = chi_square_statistics
    results.sort_values(["chi_square and p-value"], inplace=True)

    return dist_param, results


def best_distribution(dist_param, results):
    """
    Find best distribution for Optimal Stopping Point Prediction.

    This function takes the distribution paramaters and results
    from fit_distribution function and finds the best distribution
    and returns disribution name and parameters.
    """
    best_dist = results["Distribution"][0]
    params = dist_param[dist_param["Distribution Names"] == best_dist][
        "Parameters"
    ].values
    params = list(itertools.chain(*params))
    return best_dist, params
