# Interpreation of cell values and labels in TestGrid data

**Interpretation of cell values:**

In this [file](https://github.com/GoogleCloudPlatform/testgrid/blob/a18fe953cf98174c215c43e0258b0515e37c283b/pb/test_status/test_status.proto#L3), we can see the meaning of different cell values.

| Value | Interpretation |
| ----- | -------------- |
| 0     | `NO_RESULT`    |
| 1     | `PASS`         |
| 2     | `PASS_WITH_ERRORS`    |
| 3     | `PASS_WITH_SKIPS`         |
| 4     | `RUNNING`    |
| 5     | `CATEGORIZED_ABORT`         |
| 6     | `UNKNOWN`    |
| 7     | `CANCEL`         |
| 8     | `BLOCKED`    |
| 9     | `TIMED_OUT`         |
| 10     | `CATEGORIZED_FAIL`    |
| 11     | `BUILD_FAIL`         |
| 12     | `FAIL`    |
| 13     | `FLAKY`         |
| 14     | `TOOL_FAIL`         |
| 15     | `BUILD_PASSED`    |



**Interpretation of labels:**

| Label | Interpretation |
| ----- | -------------- |
| `F`   | Failed         |
| `R`   | Running        |
