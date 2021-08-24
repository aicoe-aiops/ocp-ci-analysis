# KPI Metrics

In order to measure the effectiveness and quality of our CI testing process, we need to establish the relevant key performance indicators, or KPIs. These KPIs can not only help us evaluate any AI-based enhancements we make to the CI processes, but also pinpoint what specific areas need the most improvement and therefore should be devoted resources to. The notebooks contained in this directory define, calculate, and save several KPIs that we believe are relevant to various personas (developer, manager, etc.) involved in the CI process.

These notebooks are also available for viewing and running on the public [JupyterHub instance](https://jupyterhub-opf-jupyterhub.apps.zero.massopen.cloud/hub/login) provided via the Operate First initiative.

When addressing a KPI request, please make sure to follow the [Metric-template.ipynb](metric_template.ipynb) that defines a template for calculating metrics using [number-of-flakes.ipynb](number_of_flakes.ipynb) as an example.

The following section describes what each of these KPIs represent, and links to the notebooks that calculate these.

## List of Available Metrics

1. **Number of tests blocked** : What is the total number of blocked tests on testgrid i.e. where the value in the cell is "8”.

    [Blocked-timed-out.ipynb](blocked_timed_out.ipynb)

2. **Blocked tests Percentage :** What is the percentage of blocked tests on testgrid i.e. where the value in the cell is “8”.

    [Blocked-timed-out.ipynb](blocked_timed_out.ipynb)

3. **Number of tests timed out** : What is the total number of timed out tests on testgrid i.e. where the value in the cell is "9".

    [Blocked-timed-out.ipynb](blocked_timed_out.ipynb)

4. **Timed out tests Percentage** :  What is the percentage of timed out tests on testgrid i.e. where the value in the cell is “9”.

    [Blocked-timed-out.ipynb](blocked_timed_out.ipynb)

5. **Number of builds passed** : What is the total number of builds that were passing i.e. had the “Overall” cell labeled as passing on testgrid.

    [Build-pass-failure.ipynb](build_pass_failure.ipynb)

6. **Percentage of builds passed** :  What is the percentage of builds that were passing i.e. had the “Overall” cell labeled as passing on testgrid.

    [Build-pass-failure.ipynb](build_pass_failure.ipynb)

7. **Number of builds failed** :  What is the total number of builds that were failing i.e. had the “Overall” cell labeled as failing on testgrid.

    [Build-pass-failure.ipynb](build_pass_failure.ipynb)

8. **Percentage of builds failed** :  What is the percentage of builds that were failing i.e. had the “Overall” cell labeled as failing on testgrid.

    [Build-pass-failure.ipynb](build_pass_failure.ipynb)

9. **Change in success and failure** : What is the change in success and failure builds for all tests relative to the total number of builds.

    [Build-pass-failure.ipynb](build_pass_failure.ipynb)

10. **Correlated test failure sets per test** : What is the number of tests highly correlated with a given test i.e. with a correlation coefficient of 0.9 or above.

    [Correlated-failures.ipynb](correlated_failures.ipynb)

11. **Average size of correlation set** : What is the average size of correlated test sets as calculated in the above metric.

    [Correlated-failures.ipynb](correlated_failures.ipynb)

12. **Flaky tests** : What is the number of flaky tests i.e. where the value of the cell is "13", aggregated over each platform, grid, or tab for each day/week/month.

    [Number-of-flakes.ipynb](number_of_flakes.ipynb)

13. **Flake Severity** : What is the percentage of flakes by test overall. This can also be seen as a severity level or overall flake rate of test.

    [Number-of-flakes.ipynb](number_of_flakes.ipynb)

14. **Percent of Tests Fixed**: What is the percentage of tests that were failing in the previous run of the build, but are now passing. This metric can be aggregated over each platform, grid, or tab.

    [Pct-fixed-each-ts.ipynb](pct_fixed_each_ts.ipynb)

15. **Mean Length of Failures**: How many times was the build (test suite) run before a failing test started to pass.

    [Persistent-failures-analysis.ipynb](persistent_failures_analysis.ipynb)

16. **Mean Time to Fix**: How much time was taken before a failing test started to pass.

    [Persistent-failures_analysis.ipynb](persistent_failures_analysis.ipynb)

17. **Consecutive Failure Rate**: What percentage of failing tests stay failing for more than one build. That is, what percentage of failing tests are not “one-off” failures.

    [Persistent-failures-analysis.ipynb](persistent_failures_analysis.ipynb)

18. **Pass to Fail Rate**: What percentage of test runs resulted in a “pass” to “fail” transition.

    [Persistent-failures-analysis.ipynb](persistent_failures_analysis.ipynb)

19. **Fail to Pass Rate**: What percentage of test runs resulted in a “fail” to “pass” transition.

    [Persistent-failures-analysis.ipynb](persistent_failures_analysis.ipynb)

20. **Total number of test cases**: What is the total number of tests runs (cells) on testgrid.

    [Test-pass-failures.ipynb](test_pass_failures.ipynb)

21. **Number of Passing Tests**: What is the total number of test runs that passed i.e. total number of green cells on testgrid.

    [Test-pass-failures.ipynb](test_pass_failures.ipynb)

22. **Number of Failing Tests**: What is the total number of test runs that failed i.e. total number of red cells on testgrid.

    [Test-pass-failures.ipynb](test_pass_failures.ipynb)

23. **Percent of Passing Tests**: What is the percent of test runs that passed i.e. percent of green cells on testgrid.

    [Test-pass-failures.ipynb](test_pass_failures.ipynb)

24. **Percent of Failing Tests**: What is the percent of test runs that failed i.e. percent of red cells on testgrid.

    [Test-pass-failures.ipynb](test_pass_failures.ipynb)

25. **Average time to test**: What is the average time taken to run builds and how does it change over time.

    [Time-to-test.ipynb](time_to_test.ipynb)

26. **Probability to Fail**: What is the probability that a given test will fail for a given moment in time.

    [Probability-to-fail.ipynb](probability_to_fail.ipynb)
