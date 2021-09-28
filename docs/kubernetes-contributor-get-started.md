# Get Involved as a Kubernetes Contributor

If you are a Kubernetes contributor and would like to engage with the AI4CI project, here are a few ways to get started.

## Kubernetes CI/CD Data Sources

### Analyzing TestGrid

In order to quantify the current state of the CI workflow and identify any gaps within the CI process, we calculate certain key performance indicators and metrics related to the test runs using the data obtained from [TestGrid](https://testgrid.k8s.io/).

Here is a short [demo](https://youtu.be/4Mlg2qijEgQ) on how we extract data from the testgrid UI, calculate metrics from it and display them on dashboards.

You can find links to various notebooks and videos describing analyses done on Testgrid data [here](content.md#testgrid).

### Analyzing Prow

[Prow](https://prow.k8s.io/) logs represent a rich source of information for automated triaging and root cause analysis. But unfortunately, these logs are noisy data types, i.e. two logs of the same kind but from two different sources may be different enough at a character level that traditional comparison methods are insufficient to capture this similarity. To overcome this issue, we perform clustering and term frequency analysis on build logs in order to determine whether the contents of build logs can provide additional insight into job execution.

Here is a short [demo](https://youtu.be/JjFWFaMfUJA) on how to discover and interact with the prow logs data available on GCS/origin-ci-test as well as provide some simple EDA to help folks get started with analyzing this data.

You can find links to various notebooks and videos describing analyses done on Prow Logs [here](content.md#prowgcs-artifacts).

## Interact with Dashboards

As a part of AI4CI, we collect and store data from TestGrid and calculate relevant metrics and key performance indicators (KPIs) and visualize them using dashboards. You can view and interact with the publicly available dashboard [here](https://superset.apps.devconfus2021.aws.operate-first.cloud/superset/dashboard/ocp-ci-kpi-dashboard/).
