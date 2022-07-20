#!/usr/bin/env python
# coding: utf-8

# # Optimal Stopping Point - Helper Functions
# 
# This notebook contains some helper functions that are needed for the calculation of Optimal Stopping Point.

# In[1]:


## Import libraries
from enum import Enum
import pandas as pd
import numpy as np
import datetime
from io import BytesIO
from pathlib import Path
import warnings
import itertools


from intersect import intersection
from sklearn.preprocessing import StandardScaler
import scipy.stats

from dotenv import load_dotenv, find_dotenv
import boto3

load_dotenv(find_dotenv())
warnings.filterwarnings("ignore")


# ## Metric Template Functions
# 
# These are functions copied from `../data_sources/TestGrid/metrics_template.ipynb` as a workaround to [this issue](https://github.com/elyra-ai/elyra/issues/1734) where functions imported from notebooks must be in the same directory.

# In[2]:


class CephCommunication:
    """
    Class to establish communication with a ceph s3 bucket.
    It connects with the bucket and provides methods to read and write data in the parquet format.
    """

    def __init__(
        self, s3_endpoint_url, aws_access_key_id, aws_secret_access_key, s3_bucket
    ):
        self.s3_endpoint_url = s3_endpoint_url
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.s3_resource = boto3.resource(
            "s3",
            endpoint_url=self.s3_endpoint_url,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
        )
        self.bucket = s3_bucket
        ## Todo: Add try catch

    def upload_to_ceph(self, dataframe, s3_path, filename):
        """
        This helper function takes as input the data frame to be uploaded, and the output filename.
        It then saves the data frame in the defined ceph bucket.
        """
        parquet_buffer = BytesIO()
        dataframe.to_parquet(parquet_buffer)
        s3_obj = self.s3_resource.Object(self.bucket, f"{s3_path}/{filename}")
        status = s3_obj.put(Body=parquet_buffer.getvalue())
        return status

    def read_from_ceph(self, s3_path, filename):
        """
        Helper function to read from ceph and see if the saved data is correct.
        """
        buffer = BytesIO()
        s3_object = self.s3_resource.Object(self.bucket, f"{s3_path}/{filename}")
        s3_object.download_fileobj(buffer)
        df_temp = pd.read_parquet(buffer)
        return df_temp


# In[3]:


def save_to_disk(dataframe, path, filename):
    """
    Helper function to save the dataframe
    as a parquet file to disk.
    """
    dataset_base_path = Path(path)
    dataset_base_path.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(f"{path}/{filename}")
    return True


def read_from_disk(path, filename):
    """
    Helper function to read from disk and see if the saved data is the same.
    """
    return pd.read_parquet(f"{path}/{filename}")


# In[4]:


## Helper classes and functions
## In your metric notebook, you could just
## load the template notebook functions
## If your metric requires specific helper
## functions, write them here, otherwise if these
## functions are general, update the template.
## Author: AIOps


class TestStatus(Enum):
    """
    Enum to encode what test status each value in testgrid corresponds to

    Basically python equivalent of the enum here:
    https://github.com/GoogleCloudPlatform/testgrid/blob/a18fe953cf98174c215c43e0258b0515e37c283b/pb/test_status/test_status.proto#L3
    """

    NO_RESULT = 0
    PASS = 1
    PASS_WITH_ERRORS = 2
    PASS_WITH_SKIPS = 3
    RUNNING = 4
    CATEGORIZED_ABORT = 5
    UNKNOWN = 6
    CANCEL = 7
    BLOCKED = 8
    TIMED_OUT = 9
    CATEGORIZED_FAIL = 10
    BUILD_FAIL = 11
    FAIL = 12
    FLAKY = 13
    TOOL_FAIL = 14
    BUILD_PASSED = 15


def decode_run_length(x):
    """
    Decodes the run length encoded data into an unrolled form.
    Returns a list of values.

    E.g. takes in [{"value":12, "count":3}, {"value":1, "count":2}]
    and gives [12, 12, 12, 1, 1]
    """
    lst = []
    for run_length in x:
        extension = [run_length["value"]] * run_length["count"]
        lst.extend(extension)
    return lst


def testgrid_labelwise_encoding(data, label, overall_only=True):

    """
    Run length encode the dataset and unroll the dataset into a list.
    Return flattened list after encoding specified value as
    True and rest as False
    """

    percent_label_by_grid_csv = []

    for tab in data.keys():

        for grid in data[tab].keys():
            current_grid = data[tab][grid]

            if len(current_grid["grid"]) == 0:
                pass
            else:
                # get all timestamps for this grid (x-axis of grid)
                timestamps = [
                    datetime.datetime.fromtimestamp(x // 1000)
                    for x in current_grid["timestamps"]
                ]
                # get all test names for this grid (y-axis of grid)
                tests = [
                    current_grid["grid"][i]["name"]
                    for i in range(len(current_grid["grid"]))
                ]

                graphs = [
                    current_grid["grid"][i]["graphs"]
                    for i in range(len(current_grid["grid"]))
                ]

                # unroll the run-length encoding and set bool for flake or not (x==13)
                decoded = [
                    (
                        np.array(decode_run_length(current_grid["grid"][i]["statuses"]))
                        == label
                    ).tolist()
                    for i in range(len(current_grid["grid"]))
                ]

                # add the timestamp to bool value
                decoded = [list(zip(timestamps, g)) for g in decoded]
                # add the test, tab and grid name to each entry
                # TODO: any ideas for avoiding this quad-loop
                # if the label is passed as an arg, add the timestamp, tab,
                # grid, tests, graphs metric and the bool values
                if label:
                    for i, d in enumerate(decoded):
                        for j, k in enumerate(d):
                            # here we are fetching the test duration values for the tests
                            # however,since not all tests contain time duration values,
                            # we are only considering the 'Overall' test and fetching the
                            # time duration values for this test and setting it to 'None'
                            # for all the other tests in each grid
                            if overall_only:
                                if "Overall" in tests[i]:
                                    try:
                                        test_duration = graphs[i][0]["values"][0][j]
                                    except IndexError:
                                        test_duration = None
                                else:
                                    test_duration = None
                            else:
                                try:
                                    graphs[i][0].keys()
                                    try:
                                        graphs[i][0]["values"][0][j]
                                        test_duration = graphs[i][0]["values"][0][j]
                                    except IndexError:
                                        test_duration = None
                                except TypeError:
                                    test_duration = None

                            decoded[i][j] = (
                                k[0],
                                tab,
                                grid,
                                tests[i],
                                test_duration,
                                k[1],
                            )
                    # accumulate the results
                    percent_label_by_grid_csv.append(decoded)

                # if label is 'None', add only the timestamp, tab, grid, tests and test
                # duration values
                else:
                    for i, d in enumerate(decoded):
                        for j, k in enumerate(d):
                            # here we are fetching the time duration values for the tests
                            # however,since not all tests contain time duration values,
                            # we are only considering the 'Overall' test and fetching the time duration
                            # values for this test in each grid
                            if overall_only:
                                if "Overall" in tests[i]:
                                    try:
                                        test_duration = graphs[i][0]["values"][0][j]
                                    except IndexError:
                                        print(
                                            "Test duration value does not exist for all \
                                            timestamps for test Overall in grid ",
                                            grid,
                                            "in tab ",
                                            tab,
                                        )
                                        test_duration = None
                                else:
                                    test_duration = None
                            else:
                                try:
                                    graphs[i][0].keys()
                                    try:
                                        graphs[i][0]["values"][0][j]
                                        test_duration = graphs[i][0]["values"][0][j]
                                    except IndexError:
                                        test_duration = None
                                except TypeError:
                                    test_duration = None

                            decoded[i][j] = (k[0], tab, grid, tests[i], test_duration)
                    percent_label_by_grid_csv.append(decoded)

    # output above leaves us with a doubly nested list. Flatten
    flat_list = [item for sublist in percent_label_by_grid_csv for item in sublist]
    flatter_list = [item for sublist in flat_list for item in sublist]

    return flatter_list


# ## Optimal Stopping point functions

# In[5]:


def fetch_all_tests(raw_data, status_code):
    """
    This function takes raw data and status code of the test
    and returns a dataframe with all the tests with that status code.
    """
    # Fetch the list of all tests
    tests_list = testgrid_labelwise_encoding(raw_data, status_code, overall_only=False)
    # Convert to dataframe
    tests_df = pd.DataFrame(
        tests_list,
        columns=[
            "timestamp",
            "tab",
            "grid",
            "test",
            "test_duration",
            "failure/passing",
        ],
    )
    tests_df.head()
    # We will drop all the rows having NaN values
    tests_df = tests_df.dropna()
    tests_df = tests_df[tests_df["failure/passing"]]
    return tests_df


# In[6]:


def filter_test_type(df, test):
    """
    This function takes the dataframe with all tests and
    filters the dataframe for a specified test name.
    """
    list_test = df[df["test"] == test]
    list_test = list_test.reset_index(drop=True)
    return list_test


# In[7]:


def standardize(df, column, pct, pct_lower):
    """
    Function to standardize the features by removing the mean
    and scaling to unit variance using StandardScaler library.

    Returns standandardized feature, length of the feature
    and the original feature.
    """
    sc = StandardScaler()
    y = df[column][df[column].notnull()].to_list()
    y.sort()
    len_y = len(y)
    y = y[int(pct_lower * len_y) : int(len_y * pct)]
    len_y = len(y)
    yy = [[x] for x in y]
    sc.fit(yy)
    y_std = sc.transform(yy)
    y_std = y_std.flatten()
    return y_std, len_y, y


# In[8]:


def fit_distribution(df, column, pct, pct_lower):
    """
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

    print("\nDistributions listed by Betterment of fit:")
    print("............................................")
    print(results)
    return dist_param, results


# In[9]:


def best_distribution(dist_param, results):
    """
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


# In[10]:


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

