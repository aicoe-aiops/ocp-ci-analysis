# Get Involved as a Developer

If you are a developer and would like to engage with the AI4CI project, here are a few ways to get started and we compile a list of ways you can engage with this project.

## CI/CD Data Sources

As part of the AI4CI project, we collect data from various CI/CD data sources and build tools to analyze them and derive meaningful insights from them.

The applications and CI/CD tools we currently collect data from are as follows:
* [TestGrid](content.md#testgrid)
* [Prow/GCS Artifacts](content.md#prowgcs-artifacts)
* [Github](content.md#github)
* [Bugzilla](content.md#bugzilla)
* [Telemetry](content.md#telemetry)

## Interact with Dashboards

As a part of AI4CI, we collect the relevant metrics and key performance indicators (KPIs) from various CI/CD data sources and visualize them using dashboards. You can view and interact with the publicly available dashboard [here](https://superset.apps.devconfus2021.aws.operate-first.cloud/superset/dashboard/ocp-ci-kpi-dashboard/).

## Interact with Model Endpoints

* **Github Time to Merge Model**: We have an interactive endpoint available for a model which can predict the time taken to merge a PR and classifies it into one of a few predefined time ranges. To interact with the model, check out this [Model Inference Notebook](../notebooks/time-to-merge-prediction/model_inference.ipynb)

    You can find more information about Github Time to Merge Model [here](../notebooks/time-to-merge-prediction/README.md).

* **Build Log Clustering Model** : We also have an interactive endpoint for a model that uses unsupervised machine learning techniques such as k-means and tf-idf for clustering build logs. To interact with the model, check out this [Seldon deployment](../notebooks/data-sources/gcsweb-ci/build-logs/model_seldon.ipynb).

## Video Playlist

[Here](https://www.youtube.com/playlist?list=PL8VBRDTElCWoGwMhCp04rQFMcIhshv33U) is a video playlist for the AI4CI project which goes over different analyses and walks through various notebooks within the project.
