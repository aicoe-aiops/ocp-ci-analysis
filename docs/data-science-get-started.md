# Get Involved as a Data Scientist

If you are a Data Scientist and would like to get involved with this project, here are a few ways to get started.

## Leverage Open Data Sources

One of the goals of the AI4CI project is to build an open AIOps community centered around open IT operations datasets. As a part of this project, we've curated various open source data sets from the Continuous Integration (CI) process of OpenShift (Other projects to be added in the future)  and made them available for others to explore and apply there own analytics and ML. A detailed overview of our current efforts with each of these data sources, including data collection scripts and exploratory analyses is available [here](content.md#data-engineering-metrics-and-kpis-for-ci) for you to learn more.

## Try out the Notebooks

There are interactive and reproducible notebooks for this entire project available for anyone to start using on our public [JupyterHub](https://jupyterhub-opf-jupyterhub.apps.smaug.na.operate-first.cloud) instance on the [Massachusetts Open Cloud](https://massopen.cloud/) (MOC) right now!

1. To get started, access [JupyterHub](https://jupyterhub-opf-jupyterhub.apps.smaug.na.operate-first.cloud), select log in with `moc-sso` and sign in using your Google Account.
2. After signing in, on the spawner page, please select the `ocp-ci-analysis:latest` image in the JupyterHub Notebook Image section from the list and select `Medium` from the container size drop down and hit `Start` to spawn your server.
3. Once your server has spawned, you should see a directory titled `ocp-ci-analysis-<current-timestamp>`. This directory contains the entire project repo, including notebooks that can be run directly in this Jupyter Hub environment.
4. To interact with the S3 bucket and access the stored datasets, make sure you have a `.env` file at the root of your repo. Check [.env-example](../.env-example) for an example `.env` file and open an [issue](https://github.com/aicoe-aiops/ocp-ci-analysis/issues) for access credentials.

You can find more information on the various notebooks and their purpose [here](content.md).

If you need more help navigating the Operate First environment, we have a few [short videos](https://www.youtube.com/playlist?list=PL8VBRDTElCWpneB4dBu4u1kHElZVWfAwW) to help you get started.

## Interact with Dashboards
As a part of AI4CI, we collect the relevant metrics and key performance indicators (KPIs) and visualize them using dashboards. You can view and interact with the publicly available dashboard [here](https://superset.operate-first.cloud/superset/dashboard/ai4ci/).

## Interact with Model Endpoints

* **Github Time to Merge Model**: We have an interactive endpoint available for a model which can predict the time taken to merge a PR and classifies it into one of a few predefined time ranges. To interact with the model, check out this [Model Inference Notebook](../notebooks/time-to-merge-prediction/model_inference.ipynb).

    You can find more information about Github Time to Merge Model [here](../notebooks/time-to-merge-prediction/README.md).

* **Build Log Clustering Model** : We also have an interactive endpoint for a model that uses unsupervised machine learning techniques such as k-means and tf-idf for clustering build logs. To interact with the model, check out this [Seldon deployment](../notebooks/data-sources/gcsweb-ci/build-logs/model_seldon.ipynb).

## Automate your AI/ML Workflows

We use Elyra and Kubeflow pipelines to automate the various steps in the project responsible for data collection, metric calculation and ML analysis. To automate your notebook workflows you can follow this [guide](automating-using-elyra.md) or [tutorial video](https://youtu.be/bh5WpKq3W7Y).

## Video Playlist

[Here](https://www.youtube.com/playlist?list=PL8VBRDTElCWoGwMhCp04rQFMcIhshv33U) is a video playlist for the AI4CI project which goes over different analyses and walks through various notebooks within the project.
