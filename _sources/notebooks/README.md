### Notebooks

This folder contains various reproducible and interactive Jupyter notebooks created for the project. These notebooks are also available for viewing on our public [JupyterHub instance](https://jupyterhub-opf-jupyterhub.apps.zero.massopen.cloud/hub/login).
The notebook organizational structure is described in detail below.

#### Data Sources

The notebooks defined in the [`data-sources`](data-sources) folder, explore and analyze the [Sippy](https://github.com/openshift/sippy) and [TestGrid](https://github.com/GoogleCloudPlatform/testgrid) data sources, as well as log output from OpenShift CI.

1. **Sippy** - A Continuous Integration Private Investigator tool to process the job results from https://testgrid.k8s.io/. It reports on which tests fail most frequently along different dimensions such as by job, by platform, by sig etc.
2. **TestGrid** -  A highly-configurable, interactive dashboard for viewing your test results in a grid.

This folder contains:

* a [`Sippy`](data-sources/Sippy) folder, which consists of:
  1. [`sippy_failure_correlation.ipynb`](data-sources/Sippy/sippy_failure_correlation.ipynb) - Notebook which analyzes the available Sippy CI data and determine which test failures appear to be correlated with each other
  2. a [`stage`](data-sources/Sippy/stage) folder which contains some new features and exploratory work we are trying out. It consists of:
      1. [`sippy_EDA.ipynb`](data-sources/Sippy/stage/sippy_EDA.ipynb) - Notebook which uncovers and understands the Sippy data set
      2. [`sippy-analysis.ipynb`](data-sources/Sippy/stage/sippy-analysis.ipynb) - Notebook analyzing the OpenShift CI test/job data from the [testgrid dashboards](https://testgrid.k8s.io/redhat-openshift-informing) that Sippy provides

* a [`TestGrid`](data-sources/TestGrid) folder which consists of:
  1. [`Metrics`](data-sources/TestGrid/metrics) - This folder contains notebooks that define, calculate, and save several KPIs that we believe are relevant to various personas (developer, manager, etc.) involved in the CI process.
  1. [`testgrid_EDA.ipynb`](data-sources/TestGrid/testgrid_EDA.ipynb) - Notebook which explores the existing TestGrid data at testgrid.k8s.io, giving specific attention to [Red Hat's OpenShift CI dashboards](https://testgrid.k8s.io/redhat-openshift-informing).
  2. [`testgrid_indepth_EDA.ipynb`](data-sources/TestGrid/testgrid_indepth_EDA.ipynb) - Notebook which follows up on the above notebook and provides additional insights to the testgrid data.
  3. [`testgrid_metadata_EDA.ipynb`](data-sources/TestGrid/testgrid_metadata_EDA.ipynb) - Notebook which explores metadata present at a Test level within the existing TestGrid data at testgrid.k8s.io.
  3. a [`background`](data-sources/TestGrid/background) folder which contains:
      1. [`testgrid_feature_confirmation.ipynb`](data-sources/TestGrid/background/testgrid_feature_confirmation.ipynb) - Notebook determining if the testgrid features analzed in the [testgrid_EDA.ipynb](https://github.com/aicoe-aiops/ocp-ci-analysis/blob/master/notebooks/data-sources/TestGrid/testgrid_EDA.ipynb) notebook are uniform across grids.

* a [`gcsweb-ci`](data-sources/gcsweb-ci) folder which consists of:
  1. a [`build-logs`](data-sources/gcsweb-ci/build-logs) folder to work with build logs from OpenShift CI jobs:
     1. [`build_log_EDA.ipynb`](data-sources/gcsweb-ci/build-logs/build_log_EDA.ipynb) - Notebook that can download build log data and provides an overview of it.
     2. [`build_log_term_freq.ipynb`](data-sources/gcsweb-ci/build-logs/build_log_term_freq.ipynb) - Notebook that applies term frequency analysis to build logs

#### Failure Type Classification

The notebooks defined in the [`failure-type-classification`](failure-type-classification) folder focuses on addressing the problem of automating the task of test failure classification with TestGrid data. Failures which occur in a test can be legitimate or due to some other issues like an infrastructure flake, install flake, flaky test, or some other type of failure. Unsupervised machine learning methods and heuristics are explored in these notebooks to classify the test failures. The notebooks are organized into:

* a [`background`](failure-type-classification/background) folder which consists of:
  1. [`testgrid_flakiness_detection.ipynb`](failure-type-classification/background/testgrid_flakiness_detection.ipynb) - Notebook which tries to detect one of the test failure types such as a Flaky test

* a [`stage`](failure-type-classification/stage) folder which consists of:
  1. [`failure_type_classifier.ipynb`](failure-type-classification/stage/failure_type_classifier.ipynb) - Notebook for analyzing testgrids and generating a report that will identify the tests and dates where 4 different types of failures may have occurred
