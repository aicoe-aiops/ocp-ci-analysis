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

OpensShift, Kubernetes and a few other platforms have made their CI data public. This is real world multimodal
production operations data, a rarity for public data sets today. This represents a great starting point and first initial
problem for the AIOps community to tackle. Our aim is to begin cultivating this open source AIOps project by
developing, integrating and operating AI tools for CI by leveraging the open data that has been made available by
OpenShift, Kubernetes and others.

The higher order goal to keep in mind for any tools developed here should be to assist developers in decreasing their
time to resolution for issues that are signaled by anything present in the CI data. There are a lot of ways that this
could be done, and instead of pre-defining a specific problem to solve, our immediate thinking is to make the initial
tools and relevant data as accessible as possible to foster collaboration and contributions between data scientists and
OpenShift engineers.

## First Step/ Ways to get Involved:

**1 )** Exploratory data analysis of available data sources (gcsweb, TestGrid, Sippy) to understand the different data
types that need to be addressed and identify what kind of ML approaches are possible.

- Outcome will be notebooks that explain the data, how to access it and preprocess it in a data science friendly
  format (see this
  [notebook for TestGrid data](notebooks/TestGrid_EDA.ipynb)
  for an example).

**2 )** Failure type classification with the Testgrid data to address an existing manual process executed by subject
matter experts (SME).

- Automate the manual process outlined by an SME [here](https://github.com/aicoe-aiops/ocp-ci-analysis/issues/1).

**3 )** Tooling to generate analysis reports for platforms, grids and test within the TestGrid data.
