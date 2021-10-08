"""Here goes the prediction code."""
import os
import joblib
import boto3
from io import BytesIO
from typing import Iterable, Dict, List, Union

import numpy as np
import pandas as pd


class GitHubTTMModel(object):
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
        print("Initializing")
        self.model = self.load_model_from_s3()

    def load_model_from_s3(self):
        """Load pretrained model from ceph s3 bucket."""
        # init s3 client
        s3_resource = boto3.resource(
            "s3",
            endpoint_url=os.getenv("S3_ENDPOINT"),
            aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
        )

        # download into a buffer
        buffer = BytesIO()
        s3_object = s3_resource.Object(
            os.getenv("S3_BUCKET"),
            f"{os.getenv('S3_MODEL_KEY')}/model.joblib",
        )
        s3_object.download_fileobj(buffer)

        # load from buffer
        model = joblib.load(buffer)
        return model

    def class_names(self) -> Iterable[str]:
        """Return names of output classes."""
        return [f"Class_{i}" for i in range(10)]

    def transform_input(
        self, X: np.ndarray, names: Iterable[str], meta: Dict = None  # noqa: N803
    ) -> Union[np.ndarray, List, str, bytes]:
        """Preprocess input data."""
        return X

    def predict(
        self,
        X,  # noqa: N803
        features_names=[
            "title",
            "body",
            "size",
            "created_by",
            "created_at",
            "closed_at",
            "closed_by",
            "merged_at",
            "commits_number",
            "changed_files_number",
            "interactions",
            "reviews",
            "labels",
            "commits",
            "changed_files",
        ],
    ):  # noqa: N803
        """
        Return a prediction.

        Parameters
        ----------
        X : array-like
        feature_names : array of feature names (optional)
        """
        # must convert to df first
        # TODO: move this to transform_input
        x_df = pd.DataFrame(data=X, columns=features_names)
        return self.model.predict_proba(x_df)
