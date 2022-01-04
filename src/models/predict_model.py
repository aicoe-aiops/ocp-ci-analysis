"""Here goes the prediction code."""
import scipy
from intersect import intersection


def optimal_stopping_point(
    best_dist,
    y_std_failing,
    y_failing,
    parameters_failing,
    y_std_passing,
    y_passing,
    parameters_passing,
):
    """
    Predict Optimal Stopping Point.

    This function takes the best_distribution,
    failing and passing distributions and parameters
    and returns an optimal stopping point for the test.
    """
    dist = getattr(scipy.stats, best_dist)

    # Obtain the intersection points between the distribution curves
    x, y = intersection(
        y_failing,
        dist.pdf(
            y_std_failing,
            parameters_failing[0],
            parameters_failing[1],
            parameters_failing[2],
        ),
        y_passing,
        dist.pdf(
            y_std_passing,
            parameters_passing[0],
            parameters_passing[1],
            parameters_passing[2],
        ),
    )
    osp = max(x)
    return osp
