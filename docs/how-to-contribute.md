# Contribute to the AI for Continuous Integration Project

To get started with familiarizing yourself with the AI for Continuous Integration (AI4CI) project, check how to [Get Started](get-started.md).

## Contribute to a KPI Metric

In order to quantify and evaluate the current state of the CI workflow, we have started to establish and collect the relevant metrics and key performance indicators (KPIs) needed to measure them. We have a list of [metrics](../notebooks/data-sources/TestGrid/metrics/README.md) that we currently collect. We encourage contributions to this work of developing additional KPIs and metrics.

- In order to include an additional KPI, you can open a KPI Request [issue](https://github.com/aicoe-aiops/ocp-ci-analysis/issues?q=is%3Aissue+is%3Aopen+%22KPI+Request%22+). Specify the KPI you wish to collect with the prefix `KPI Request:`.
- In order to add a notebook to fulfill one of the existing open `KPI Request:` issues, you can use the [KPI template notebook](../notebooks/data-sources/TestGrid/metrics/metric_template.ipynb). The template notebook has helper functions and examples to make contributing new metrics as simple and as uniform as possible.
- When defining the file prefix for your metrics stored in the shared ceph instance, please be sure to use the following format: `s3://opf-datacatalog/ai4ci/<dat_source>/metrics/<metric_name>/<date_analysis_was_generated>.parquet`
- Submit a Pull Request to the [project repo](https://github.com/aicoe-aiops/ocp-ci-analysis) with your KPI analysis notebook.
- In order to add the notebook to the automated Kubeflow workflow, follow intructions in the [guide](automating-using-elyra.md).

## Contribute to an ML Analysis

With the necessary KPIs available to quantify and evaluate the CI workflow, we can start to apply some AI and machine learning techniques to help improve the CI workflow. We encourage you to contribute to this work developing additional machine learning analyses or adding features to the existing analyses.

- In order to include an additional ML Analysis, you can open an ML Request [issue](https://github.com/aicoe-aiops/ocp-ci-analysis/issues?q=is%3Aissue+is%3Aopen+%22ML+Request%22+). Specify the machine learning application you would like to have included with the prefix `ML Request:`.
- When uploading your model to the shared ceph storage instance please use the following prefix format to ensure no files get overwritten: `s3://opf-datacatalog/ai4ci/<ML_directory_name>/model/<model_file>`
- Submit a Pull Request to the [project repo](https://github.com/aicoe-aiops/ocp-ci-analysis) with your ML analysis notebook.
- In order to add the notebook to the automated Kubeflow workflow, follow instructions in the [guide](automating-using-elyra.md).
