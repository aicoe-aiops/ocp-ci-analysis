In this document, our objective is to document the Interpretation of the cell values and cell labels in [testgrid](https://testgrid.k8s.io/):

**Interpretation of cell values:**

In this [file](https://github.com/GoogleCloudPlatform/testgrid/blob/a18fe953cf98174c215c43e0258b0515e37c283b/pb/test_status/test_status.proto#L3), we can see the meaning of different cell values.

<table>
  <tr>
    <td>Value</td>
    <td>Interpretation</td>
  </tr>
  <tr>
    <td>0</td>
    <td>NO_RESULT</td>
  </tr>
  <tr>
    <td>1</td>
    <td>PASS</td>
  </tr>
  <tr>
    <td>2</td>
    <td>PASS_WITH_ERRORS</td>
  </tr>
  <tr>
    <td>3</td>
    <td>PASS_WITH_SKIPS</td>
  </tr>
  <tr>
    <td>4</td>
    <td>RUNNING</td>
  </tr>
  <tr>
    <td>5</td>
    <td>CATEGORIZED_ABORT</td>
  </tr>
  <tr>
    <td>6</td>
    <td>UNKNOWN</td>
  </tr>
  <tr>
    <td>7</td>
    <td>CANCEL</td>
  </tr>
  <tr>
    <td>8</td>
    <td>BLOCKED</td>
  </tr>
  <tr>
    <td>9</td>
    <td>TIMED_OUT</td>
  </tr>
  <tr>
    <td>10</td>
    <td>CATEGORIZED_FAIL</td>
  </tr>
  <tr>
    <td>11</td>
    <td>BUILD_FAIL</td>
  </tr>
  <tr>
    <td>12</td>
    <td>FAIL</td>
  </tr>
  <tr>
    <td>13</td>
    <td>FLAKY</td>
  </tr>
  <tr>
    <td>14</td>
    <td>TOOL_FAIL</td>
  </tr>
  <tr>
    <td>15</td>
    <td>BUILD_PASSED</td>
  </tr>
</table>


**Interpretation of labels:**

<table>
  <tr>
    <td>Label</td>
    <td>Interpretation</td>
  </tr>
  <tr>
    <td>F</td>
    <td>Failed</td>
  </tr>
  <tr>
    <td>R</td>
    <td>Running</td>
  </tr>
</table>
