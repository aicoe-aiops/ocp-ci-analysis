# AI Supported Continuous Integration Testing

_Developing AI tools for developers by leveraging the open data made available by OpenShift and Kubernetes CI
platforms._

AI Ops is a critical component of supporting any Open Hybrid Cloud infrastructure. As the systems we operate become
larger and more complex, intelligent monitoring tools and response agents will become a necessity. In an effort to
accelerate the development, access and reliability of these intelligent operations solutions, our aim here is to
provide access to an open community, open operations data and open infrastructure for data scientists and devops
engineers.

One major component of the operations workflow is Continuous Integration (CI), which involves running automated builds
and tests of software before it is merged into a production code base. If you are developing a container orchestration
platform like Kubernetes or OpenShift, these builds and tests will produce a lot of data that can be difficult to parse
when you are trying to figure out why a build is failing or why a certain set of tests aren’t passing.

OpenShift, Kubernetes and a few other platforms have made their CI data public. This is real world multimodal
production operations data, a rarity for public data sets today. This represents a great starting point and first initial
problem for the AIOps community to tackle. Our aim is to begin cultivating this open source AIOps project by
developing, integrating and operating AI tools for CI by leveraging the open data that has been made available by
OpenShift, Kubernetes and others.

The higher order goal to keep in mind for any tools developed here should be to assist developers in decreasing their
time to resolution for issues that are signaled by anything present in the CI data. There are a lot of ways that this
could be done, and instead of pre-defining a specific problem to solve, our immediate thinking is to make the initial
tools and relevant data as accessible as possible to foster collaboration and contributions between data scientists and
DevOps engineers.

## Current Work

* Interactive and reproducible notebooks for the [entire project](https://github.com/aicoe-aiops/ocp-ci-analysis) are available as an image on a public [jupyterhub](https://jupyterhub-opf-jupyterhub.apps.cnv.massopen.cloud/hub/login) instance on the MOC.
* Video explainer (forthcoming)


## Active Projects:

**1 ) Data Access and Management**

There is a lot of operations data available to us between gcsweb, prow and TestGrid, as data scientists, we need to develop methods for accessing and processing it in data science friendly formats and identify what kind of ML approaches are possible. The outcome of this work will be a set of reproducible notebooks that explain the data and make it easy to start performing analysis or training models.

* [TestGrid data access and preprocessing](notebooks/TestGrid_EDA.ipynb)
* [Indepth TestGrid exploratory data analysis for machine learing](notebooks/TestGrid_indepth_EDA.ipynb)
* [TestGrid sample dataset](data/raw/testgrid_810.json.gz)
* Prow data access and preprocessing (forthcoming)
*  gcsweb data access and preprocessing (forthcoming)

**2 ) TestGrid Failure Type Classification**

Automate an existing manual process of identifying different failure types for individual testgrids, identifying flakey tests, infra flakes, install flakes and new test failures. [Detailed project description](docs/failure-type-classification-with-the-testgrid-data-project-doc.md)

* [Flake detection in TestGrid](notebooks/Testgrid_flakiness_detection.ipynb)
* [Failure Type Classification](https://github.com/aicoe-aiops/ocp-ci-analysis/issues/41)

**3 ) Improved Reporting and Visualization**

Any analysis or predictive tool needs to be accessible and actionable for them to drive value to the end user. To that end, we are also looking at ways to improve upon the existing reporting and visualizations available to developers who rely on CI data.

* [Review existing CI data reporting tools](https://github.com/aicoe-aiops/ocp-ci-analysis/issues/32)

## Contribute:

Please review our [issues](https://github.com/aicoe-aiops/ocp-ci-analysis/issues) page, we welcome contributions there by creating new issues for additional analysis projects or taking on an existing analysis issue.
