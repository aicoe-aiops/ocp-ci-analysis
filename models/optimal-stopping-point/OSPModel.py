"""Here goes the prediction code."""
import os
import json
import boto3
import logging
import datetime
import itertools
from typing import Iterable, Dict, List, Union

import numpy as np

from src.features.build_features import standardize
from src.models.predict_model import optimal_stopping_point
from src.data.make_dataset import fetch_all_tests, filter_test_type
from src.models.train_model import fit_distribution, best_distribution


_LOGGER = logging.getLogger(__name__)


class OSPModel(object):
    """
    Model template.

    You can load your model parameters in __init__ from a location accessible at runtime.
    """

    def __init__(self):
        """
        Add any initialization parameters.

        These will be passed at runtime from the graph definition parameters
        defined in your seldondeployment kubernetes resource manifest.
        """
        _LOGGER.info("Initializing Optimal Stopping Point model.")

        self.s3_bucket = os.getenv("S3_BUCKET")
        self.s3_resource = self.init_s3_resource()

    def init_s3_resource(self):
        """Create s3 resource to read testgrid data from ceph s3 bucket."""
        _LOGGER.info("Creating s3 resource.")

        # init s3 client
        s3_res = boto3.resource(
            "s3",
            endpoint_url=os.getenv("S3_ENDPOINT"),
            aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
        )
        return s3_res

    def class_names(self) -> Iterable[str]:
        """Return names of output classes."""
        return ["Optimal Stopping Point (in minutes)"]

    def transform_input(
        self, X: np.ndarray, names: Iterable[str], meta: Dict = None  # noqa: N803
    ) -> Union[np.ndarray, List, str, bytes]:
        """Preprocess input data."""
        return X

    def predict_raw(
        self,
        request: Dict,
    ) -> float:
        """
        Determine optimal stopping point for a given test in a given grid.

        request is a dictionary that must have a key called `jsonData`, which in turn has keys
        `test_name` and `timestamp` that define the test to predict osp for
        """
        test_name = request["jsonData"]["test_name"]
        timestamp = request["jsonData"]["timestamp"]

        eval_datetime = datetime.datetime.fromtimestamp(int(timestamp))
        # file where raw testgrid data is stored
        filename = f"testgrid_{eval_datetime.day}{eval_datetime.month}.json"
        s3_object = self.s3_resource.Object(self.s3_bucket, f"raw_data/{filename}")

        # load raw data
        file_content = s3_object.get()["Body"].read().decode("utf-8")
        testgrid_data = json.loads(file_content)

        # failing and passing test data
        # NOTE: requires >16GB memory otherwise OOMkills
        # As a workaround, we'll modify this func to include only selected test
        failures_df = fetch_all_tests(testgrid_data, 12, test_name=test_name)
        passing_df = fetch_all_tests(testgrid_data, 1, test_name=test_name)

        # keep test data relevant to given test name only
        failures_test = filter_test_type(
            failures_df,
            test_name,
        )
        passing_test = filter_test_type(
            passing_df,
            test_name,
        )

        # try on many distributions for failures
        failure_dist, failures_r = fit_distribution(
            failures_test, "test_duration", 0.99, 0.01
        )

        best_dist, parameters_failing = best_distribution(failure_dist, failures_r)

        # try on many distributions for passing
        passing_dist, passing_r = fit_distribution(
            passing_test, "test_duration", 0.99, 0.01
        )

        parameters_passing = passing_dist[
            passing_dist["Distribution Names"] == best_dist
        ]["Parameters"].values
        parameters_passing = list(itertools.chain(*parameters_passing))

        # standardize both
        y_std_failing, len_y_failing, y_failing = standardize(
            failures_test, "test_duration", 0.99, 0.01
        )
        y_std_passing, len_y_passing, y_passing = standardize(
            passing_test, "test_duration", 0.99, 0.01
        )

        # get stopping point
        osp = optimal_stopping_point(
            best_dist,
            y_std_failing,
            y_failing,
            parameters_failing,
            y_std_passing,
            y_passing,
            parameters_passing,
        )

        return osp
