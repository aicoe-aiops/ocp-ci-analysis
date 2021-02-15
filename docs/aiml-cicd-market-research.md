## Companies Involved in AI/ML for CI/CD

A curated list of companies involved in AI/ML for CI/CD.

### AppSurify

[Appsurify’s TestBrain](https://appsurify.com/flaky-tests/) finds the right subset of your automated tests to match the specific code changes, making your testing run 10x faster. TestBrain also quarantines flaky failures so they don’t break your builds, and identifies the risks in recent commits to prioritize your manual testing so you can release faster with fewer bugs. Their solution is based mainly on GitHub related data such as commit history and focused on evaluating various commit metrics and metrics specific to developers.

Links:

*   [Website](https://appsurify.com/flaky-tests/)
*   [White Paper explaining their ML model for detecting flaking tests](https://appsurify.com/wp-content/uploads/2019/12/TestBrain-Risk-Model.pdf)

### BuildPulse.io

[BuildPulse](https://buildpulse.io/) is a GitHub application that automatically detects flaky tests in the configured GitHub repositories and highlights the most disruptive ones so you know exactly where to focus first for maximum impact. It assesses the impact of each flaky test and provides a one-click access to view the build logs and keep your test suite healthy. The percentage of commits being disrupted by flaky tests, the week-over-week change, and a summary of the biggest problem areas are some of the metrics evaluated. BuildPulse integrates with a variety of CI services such as CircleCI, Jenkins, Travis CI, Semaphore etc.

Links:

*   [Website](https://buildpulse.io/ )

### Cypress

[Cypress](https://www.cypress.io/) Dashboard enables the users to track and monitor flaky test runs in their CI. The first step in combating a “flake” is to triage and assess the severity of flaky tests so that you can appropriately prioritize the work needed to fix them. They have a new "Flaky Tests" Analytics page that provides the exact context to do just that. The flaky tests analytics pages provides a bird-eye-view on the state of flake for your project. Their code is open source and can be found [here](https://github.com/cypress-io/cypress).

Links:

*   [Website](https://www.cypress.io/)
*   [Blog for Flaky Test Detection and Alerts](https://www.cypress.io/blog/2020/10/20/introducing-flaky-test-detection-alerts/#flake-detection)
*   [Git Repo](https://github.com/cypress-io/cypress)

### Logz.io

_Log specific offering*_

[Logz.io](https://logz.io/) has a number of different offerings, but their primary product is a fully managed platform built with Grafana, Prometheus, Elasticsearch, Kibana and other open source tools for performing a number of AIOps/ DevOps monitoring tasks. Given their name, much of their analytics are focused around log data. They promote interesting AIOps capabilities such as grouping log patterns and filtering by pattern frequency, crowdsourcing application issues from online forums like StackOverflow to identify possible problematic logs and correlating logs with stack trace issues.

Links:

*   [Website](https://logz.io/)
*   [Application insights demo](https://logz.io/learn/application-insights-demo/)
*   [Introducing log patterns ](https://logz.io/learn/introducing-log-patterns/)
*   I[ntroducing distributed tracing](https://logz.io/learn/introducing-distributed-tracing/)

### Microfocus

[Microfocus](https://www.microfocus.com/en-us/home) has a variety of offerings and all are broadly related to analytics and automation for IT operations. One of their solutions, “[Accelerate Application Delivery](https://www.microfocus.com/solutions/accelerate-application-delivery)” focuses on reliably scaling Agile and DevOps operations across all environments, from from mainframe to cloud. [ALM QC](https://www.microfocus.com/en-us/products/alm-quality-center/overview), is one of the products offered that consolidates QC (Quality Center) data and provides a dashboard to view stats like test status, number of tests to run, number of tests failing, number of tests running, etc. [ALM Octane](https://www.microfocus.com/en-us/products/alm-octane/overview) is another product that provides insights into CI pipelines and failure analysis. It tries to cut down mean time to recovery, analyzes tests to find root causes and recommends potential failure owners. For test runs, they try to map commits to error messages and tests to specific modules and clusters related failures together.

Links:

*   [Website](https://www.microfocus.com/en-us/home)
*   [Accelerated Application Delivery](https://www.microfocus.com/solutions/accelerate-application-delivery)
*   [ALM QC](https://www.microfocus.com/en-us/products/alm-quality-center/overview)
*   [ALM Octane](https://www.microfocus.com/en-us/products/alm-octane/overview)

### Moogsoft

[Moogsoft](https://www.moogsoft.com/aiops-platform/) aims to deliver an enterprise cloud native platform that provides a self-servicing AI driven observability platform with capabilities such as [anomaly detection](https://www.moogsoft.com/features/anomaly-detection/), [correlation analysis](https://www.moogsoft.com/features/correlation/),[ noise reduction](https://www.moogsoft.com/features/noise-reduction/) etc. Looking into their [technical solution offered for DevOps](https://www.moogsoft.com/solutions/devops/), the platform’s main focus is to collect metrics and identify anomalies in real time across the entire application stack. It also defines significant alerts from metrics/logs/events and correlates incidents. They use algorithms such as Cookbook, Tempus, Vertex entropy which are deterministic clustering and time based algorithms for detecting correlation between alerts.

Links:

*   [Website](https://www.moogsoft.com/aiops-platform/)
*   [Observability for DevOps/SRE](https://www.moogsoft.com/wp-content/uploads/2020/10/Moogosft-Solution-Brief-100120.pdf)
*   [Using Observability for CI/CD pipelines](https://www.moogsoft.com/blog/using-observability-to-inspect-and-adapt-ci-cd-pipelines/)
*   [AI drive alert correlation](https://info.moogsoft.com/rs/092-EGH-780/images/moogsoft-at-a-glance-correlation-ns.pdf)
*   Other [whitepapers](https://www.moogsoft.com/content-library/) describing their AI/ML solutions

### SCALYR

_Log specific offering*_

[Scalyr](https://www.scalyr.com/) is a SaaS platform focused on ingesting massive amounts of machine data (high cardinality and high dimensionality) in real time with a strong focus on low-cost and high speed searching and storing. Its primary offering is a [log analytics](https://www.scalyr.com/product/) service to help engineers identify and fix problems in a large system easily. They are known to monitor and troubleshoot [Kubernetes environments](https://resources.scalyr.com/kubernetes-logging-solution-guide-eo) effectively.

Links:

*   [Website](https://www.scalyr.com/)
*   [Log analytics](https://www.scalyr.com/product/)
*   [Kubernetes environments](https://resources.scalyr.com/kubernetes-logging-solution-guide-eo)
*   [Blog](https://prod-design.eu.scalyr.com/blog/log-analyzer-what-it-is-and-how-it-can-help-you/) on their log analysis tool

### Sealights

[Sealights](https://www.sealights.io/) primarily focuses on “software quality governance”. They use AI/ML to provide recommendations on where the test gaps are, what you should fix first, how you can decide which tests are worth running and which tests aren’t worth running. Their main data sources include GitHub, Jenkins and other pipeline and log analysis tools. They apply AI/ML to CI/CD for prioritizing job queues using graph neural networks and regression methods. Classification and other statistical models are also used to identify anomalies in test time, network usage patterns and code paths.

Links:

*   [Website](https://www.sealights.io/)
*   Other [whitepapers](https://www.sealights.io/learn/) describing their AI/ML solutions
*   [Demo](https://www.youtube.com/watch?v=rbxl4nUdFOI)

### Splunk

[Splunk](https://www.splunk.com/) aims to provide a multipurpose, all-in-one business and systems monitoring tool with specific focus on security, IT and DevOps. The [DevOps CI/CD monitoring](https://www.splunk.com/en_us/devops/cicd-pipeline-monitoring.html) solution provides real-time insights across all stages of the application delivery lifecycle. Their software validates performance pre-production, post-production and in-flight, enabling DevOps teams to perform the frequent code pushes needed to stay agile with real-time monitoring of their CI/CD delivery pipeline. The Splunk test analysis feature shows all the failing tests with stack traces, flags regression failures, groups test failures by errors, captures Jenkins environment variables, and provides nifty filters to find tests with long run times, particular errors, testsuites, etc.

Links:

*   [Website](https://www.splunk.com/)
*   [DevOps CI/CD monitorin](https://www.splunk.com/en_us/devops/cicd-pipeline-monitoring.html)g
*   [Whitepaper](https://drive.google.com/file/d/1t9G3ITmjtH5Z9UtYrs-j4L03oJASV7kv/view?usp=sharing ) describing their AI/Ml implementations for DevOps

### SumoLogic

_Log specific offering*_

[SumoLogic](https://www.sumologic.com/) is a cloud-native, multi-tenant platform that helps you make data-driven decisions and reduces your time to investigate security and operational issues. It aims to help reduce downtime and move from reactive to proactive monitoring with cloud-based modern analytics powered by machine learning. Their machine-learning powered analytics aims to identify and predict anomalies in real time with outlier detection and uncover root causes using a patented [LogReduce and LogCompare](https://www.sumologic.com/solutions/machine-learning-powered-analytics/) pattern analysis. Sumo Logic’s[ Continuous Intelligence Solution for Kubernetes](https://www.sumologic.com/solutions/kubernetes/) provides the visibility teams need to confidently and securely implement Kubernetes anywhere - on premise, AWS, Azure, and GCP. It provides native integrations with best practice data sources for Kubernetes such as Prometheus, OpenTelemetry, FluentD, Fluentbit, and Falco. Overall, it seems to be an “all-in-one” AIOps and Security platform.

Links:

*   [Website](https://www.sumologic.com/)
*   [LogReduce and LogCompare](https://www.sumologic.com/solutions/machine-learning-powered-analytics/)
*   [Continuous Intelligence Solution for Kubernetes](https://www.sumologic.com/solutions/kubernetes/)

### Zebrium

[Zebrium](https://www.zebrium.com/) offers automated ML-driven root cause analysis for software incidents. It can work as a standalone application for detecting incidents and anomalies and can be well integrated with existing monitoring tools. It aims to find incidents with metric monitoring tools like [APM](https://en.wikipedia.org/wiki/Application_performance_management), identify root cause analysis from ELK and Kibana logs and integrate incident reports from third party help desks softwares such as Slack, PagerDuty, Opsgenie etc.

Links:

*   [Website](https://www.zebrium.com/)
*   [Example](https://youtu.be/JaGArw9sdhM) with cloud native application
*   [Technical talk](https://www.youtube.com/watch?v=5V1jB8crs1c) for log monitoring
