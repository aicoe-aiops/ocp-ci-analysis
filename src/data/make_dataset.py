"""Data collection code."""
import click
import logging
import datetime
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import numpy as np
import pandas as pd

_LOGGER = logging.getLogger(__name__)


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Run data processing scripts to turn raw data to cleaned data.

    Transforms raw data from (../raw) into cleaned data ready to be analyzed
    (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")


def fetch_all_tests(raw_data, status_code, test_name):
    """fetch_all_tests function takes raw data and status code of the test.

    Returns a dataframe with all the tests with that status code.
    """
    # Fetch the list of all tests
    tests_list = testgrid_labelwise_encoding(
        raw_data, status_code, test_name, overall_only=False
    )
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
    # We will drop all the rows having NaN values
    tests_df = tests_df.dropna()
    tests_df = tests_df[tests_df["failure/passing"]]
    return tests_df


def decode_run_length(x):
    """Decode the run length encoded data into an unrolled form.

    Returns a list of values.

    E.g. takes in [{"value":12, "count":3}, {"value":1, "count":2}]
    and gives [12, 12, 12, 1, 1]
    """
    lst = []
    for run_length in x:
        extension = [run_length["value"]] * run_length["count"]
        lst.extend(extension)
    return lst


def testgrid_labelwise_encoding(data, label, test_name, overall_only=True):
    """Run length encode the dataset and unroll the dataset into a list.

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
                # get all test names for this grid (y-axis of grid)
                tests = [
                    current_grid["grid"][i]["name"]
                    for i in range(len(current_grid["grid"]))
                ]
                if test_name not in tests:
                    continue

                # get all timestamps for this grid (x-axis of grid)
                timestamps = [
                    datetime.datetime.fromtimestamp(x // 1000)
                    for x in current_grid["timestamps"]
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
                                        _LOGGER.error(
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


def filter_test_type(df, test):
    """Filter the dataframe for a specified test name."""
    list_test = df[df["test"] == test]
    list_test = list_test.reset_index(drop=True)
    return list_test


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
