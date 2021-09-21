"""Feature extraction code."""
import os
import yaml
import pathlib
import requests
from datetime import datetime
from collections import Counter

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


# determine if PR made changes in some specific directories in repo
DIRS_TO_CHECK = [
    ".github",
    "docs",
    "pkg",
    "test",
    "vendor",
    "root",
]

# how many files of the given extension were changed in PR
FILE_EXTENSIONS_TO_COUNT = [
    ".1",
    ".adoc",
    ".bash",
    ".bats",
    ".c",
    ".centos7",
    ".cert",
    ".conf",
    ".crt",
    ".empty",
    ".feature",
    ".files_generated_oc",
    ".files_generated_openshift",
    ".gitattributes",
    ".gitignore",
    ".go",
    ".gz",
    ".html",
    ".ini",
    ".json",
    ".key",
    ".mailmap",
    ".markdown",
    ".md",
    ".mk",
    ".mod",
    ".pl",
    ".proto",
    ".rhel",
    ".s",
    ".sec",
    ".service",
    ".sh",
    ".signature",
    ".spec",
    ".sum",
    ".sysconfig",
    ".template",
    ".txt",
    ".xml",
    ".yaml",
    ".yaml-merge-patch",
    ".yml",
    "AUTHORS",
    "BUILD",
    "CONTRIBUTORS",
    "Dockerfile",
    "LICENSE",
    "MAINTAINERS",
    "Makefile",
    "NOTICE",
    "PATENTS",
    "README",
    "Readme",
    "VERSION",
    "Vagrantfile",
    "cert",
    "key",
    "oadm",
    "oc",
    "openshift",
    "result",
    "run",
    "test",
]

# how many times these words appeared in PR title
WORDS_TO_COUNT = [
    "add",
    "bug",
    "bump",
    "diagnostics",
    "disable",
    "fix",
    "haproxy",
    "oc",
    "publishing",
    "revert",
    "router",
    "sh",
    "staging",
    "support",
    "travis",
]


class IsUserPrivTransformer(BaseEstimator, TransformerMixin):
    """Check if pr is created by approvers / reviewers / other privileged users."""

    def __init__(
        self, priv="approvers", org="openshift", repo="origin", branch="master"
    ):
        """Initialize."""
        self.priv, self.org, self.repo, self.branch = priv, org, repo, branch

        # read and parse owners file from repo
        ret = requests.get(
            f"https://raw.githubusercontent.com/{self.org}/{self.repo}/{self.branch}/OWNERS"
        )
        if ret.status_code != requests.codes.ok:
            raise FileNotFoundError(
                f"Failed to get OWNERS file from repo https://github.com/{self.org}/{self.repo}/tree/{self.branch}.\
                    Please ensure the repo contains this file, as it is required by the current preprocessor/model"
            )
        content = ret.content.decode("utf-8")
        self.owners_file = yaml.load(content, Loader=yaml.SafeLoader)

        # find the 'approvers'/'reviewers' key in this yaml and load its content
        self.usernames = self._finditem(self.owners_file, self.priv)

    def _finditem(self, obj, key):
        if key in obj:
            return obj[key]
        for k, v in obj.items():
            if isinstance(v, dict):
                return self._finditem(v, key)

    def fit(self, X, y=None):  # noqa: N803
        """Fit."""
        return self

    def transform(self, X, y=None):  # noqa: N803
        """Transform."""
        return X.isin(self.usernames)


class DateTimeDetailsTransformer(BaseEstimator, TransformerMixin):
    """Get day, month, date, etc. from a unix timestamp."""

    def __init__(self, colname="created_at"):
        """Initialize."""
        self.colname = colname

    def fit(self, X, y=None):  # noqa: N803
        """Fit."""
        return self

    def transform(self, X, y=None):  # noqa: N803
        """Transform."""
        # convert to datetime object
        X[f"{self.colname}_datetime"] = (
            X[self.colname].fillna("").apply(lambda x: datetime.fromtimestamp(x))
        )
        # get day as categorical variable
        X[f"{self.colname}_day"] = X[f"{self.colname}_datetime"].apply(lambda x: x.day)

        # get month as categorical variable
        X[f"{self.colname}_month"] = X[f"{self.colname}_datetime"].apply(
            lambda x: x.month
        )

        # get weekday as categorical variable
        X[f"{self.colname}_weekday"] = X[f"{self.colname}_datetime"].apply(
            lambda x: x.weekday()
        )

        # get hour of day as categorical variable
        X[f"{self.colname}_hour"] = X[f"{self.colname}_datetime"].apply(
            lambda x: x.hour
        )

        return X.drop(columns=[self.colname, f"{self.colname}_datetime"])


class ChangeInDirTransformer(BaseEstimator, TransformerMixin):
    """Determine which directories have had their files changed."""

    def __init__(self, colname="changed_files", dirs_to_check=DIRS_TO_CHECK):
        """Initialize."""
        self.colname = colname
        self.dirs_to_check = dirs_to_check

    @staticmethod
    def _directories_from_filepaths(list_of_filepaths):
        directories = []
        for filepath in list_of_filepaths:
            if "/" in filepath:
                directories.append(filepath.split("/", 1)[0])
            else:
                directories.append("root")
        return directories

    def fit(self, X, y=None):  # noqa: N803
        """Fit."""
        return self

    def transform(self, X, y=None):  # noqa: N803
        """Transform."""
        change_in_dir_df = pd.DataFrame()
        for d in self.dirs_to_check:
            change_in_dir_df[f"change_in_{d}"] = X[self.colname].apply(
                lambda fpaths: 1 if d in self._directories_from_filepaths(fpaths) else 0
            )
        return change_in_dir_df


class NumChangedFilesTransformer(BaseEstimator, TransformerMixin):
    """Get number of files changed in PR."""

    def __init__(self, colname="changed_files_number"):
        """Initialize."""
        self.colname = colname

    def fit(self, X, y=None):  # noqa: N803
        """Fit."""
        return self

    def transform(self, X, y=None):  # noqa: N803
        """Transform."""
        X[self.colname] = X[self.colname].astype("int")
        return X


class StringLenTransformer(BaseEstimator, TransformerMixin):
    """Get number of words in PR description."""

    def __init__(self, colname="body"):
        """Initialize."""
        self.colname = colname

    def fit(self, X, y=None):  # noqa: N803
        """Fit."""
        return self

    def transform(self, X, y=None):  # noqa: N803
        """Transform."""
        X[self.colname] = X[self.colname].fillna("").apply(lambda x: len(x.split()))
        return X


class NumPrevPRsTransformer(BaseEstimator, TransformerMixin):
    """Get number of PRs that the creator of this PR has previously contributed."""

    def __init__(self, colname="created_by"):
        """Initialize."""
        self.colname = colname

    def fit(self, X, y=None):  # noqa: N803
        """Fit."""
        return self

    def transform(self, X, y=None):  # noqa: N803
        """Transform."""
        counts = X.groupby(self.colname).cumcount()
        return counts.to_frame(name="num_prev_prs")


class FileTypeCountTransformer(BaseEstimator, TransformerMixin):
    """Count how many files of each extension have been changed."""

    def __init__(
        self,
        colname="changed_files",
        file_extensions=FILE_EXTENSIONS_TO_COUNT,
    ):
        """Initialize."""
        self.file_extensions = file_extensions
        self.colname = colname

    def _get_filetype(self, filepath):
        # if standard file extension, return file extension
        ppath = pathlib.Path(filepath)
        if ppath.suffix:
            ftype = ppath.suffix
        # else return base filename e.g. Dockerfile, OWNERS, .gitignore
        else:
            ftype = os.path.basename(filepath)

        # if the filetype is not in the list of file types we care about
        # for counting, then return None, else return the filetype
        return ftype if ftype in self.file_extensions else None

    def fit(self, X, y=None):  # noqa: N803
        """Fit."""
        return self

    def transform(self, X, y=None):  # noqa: N803
        """Transform."""
        # get types/extensions of the files that were changed in each PR
        changed_file_types = X[self.colname].apply(
            lambda flist: [self._get_filetype(f) for f in flist]
        )

        # count the types/extensions of files changed in each PR
        filetype_count_df = changed_file_types.apply(Counter).apply(pd.Series)

        # drop any extra cols
        # filetype_count_df = filetype_count_df[self.file_extensions]
        filetype_count_df = filetype_count_df.reindex(
            self.file_extensions,
            axis=1,
            fill_value=0,
        )

        # fill nans with 0's
        filetype_count_df = filetype_count_df.fillna(0)
        return filetype_count_df


class TitleWordCountTransformer(BaseEstimator, TransformerMixin):
    """Count how many times some "key" words appear in title."""

    def __init__(
        self,
        colname="title",
        words=WORDS_TO_COUNT,
    ):
        """Initialize."""
        self.words = words
        self.colname = colname

    def fit(self, X, y=None):  # noqa: N803
        """Fit."""
        return self

    def transform(self, X, y=None):  # noqa: N803
        """Transform."""
        # convert to lowercase
        preprocessed = X[self.colname].str.lower()

        # remove punctuations and symbols like : ; , # ( ) [ ] etc
        preprocessed = preprocessed.str.replace(
            r'[`#-.?!,:;\/()\[\]"\']', " ", regex=True
        )

        # save wordcount of each word in each title
        wordcount_df = pd.DataFrame()
        for word in self.words:
            wordcount_df[f"title_wordcount_{word}"] = preprocessed.apply(
                lambda x: x.split().count(word)
            )
        return wordcount_df
