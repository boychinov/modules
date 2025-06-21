# ABTester 

A Python class for conducting professional A/B testing with numerical and categorical data.\
Supports normality testing, variance checks, t-tests, non-parametric tests, and chi-square tests — all in one streamlined and modular class.


##  Features

- ✅ Load data from CSV, Excel, or directly from DataFrame
- ✅ Assign groups via label column, condition functions, or separate files
- ✅ Automatically test for:
  - Normality (Shapiro or D’Agostino depending on sample size)
  - Variance homogeneity (Levene’s test)
  - Parametric tests (Independent T-test, Welch’s test)
  - Non-parametric test (Mann-Whitney U)
  - Categorical test (Chi-Square)
- ✅ Smart auto-detection of test type and hypothesis direction
- ✅ Neatly formatted test results using `tabulate`


##  Installation

Install required dependencies:

```bash
pip install pandas scipy tabulate openpyxl
```


## Quick Start

```python
from ab_testing import ABTester

ab = ABTester()

# Load your dataset
ab.load_data(filepath='data.csv', col_name='conversion_rate', file_type='csv')

# Assign groups using a column
ab.assign_groups(group_col='variant', label_a='A', label_b='B')

# Run significance test
ab.is_ab_test_significant()
```

##  Available Methods

### `load_data(...)`

Load dataset from a DataFrame, CSV, or Excel file.

### `assign_groups(...)`

Assign A/B groups based on a column or using filter functions.

### `assign_groups_from_files(...)`

Load Group A and Group B from two separate files or Excel sheets.

### `is_normal(...)`

Check normality of both groups using Shapiro or D’Agostino test.

### `is_variance_homogeneous()`

Check variance equality with Levene’s test.

### `apply_ttest(...)`

Run standard independent t-test.

### `apply_welch_ttest(...)`

Run Welch’s t-test for unequal variances.

### `apply_mann_whitney_u(...)`

Run Mann-Whitney U test for non-normal distributions.

### `apply_chi_square(...)`

Run chi-square test between two categorical variables.

### `decide_alternative(...)`

Auto-determine hypothesis direction (`'greater'`, `'less'`, `'two-sided'`).

### `is_ab_test_significant(...)`

Run full pipeline (normality → variance → proper test) and print summary.

### `auto_test()`

Automatically selects numeric or categorical test based on data type.


## 📊 Example Output

```
Metric        Value
------------  ------------------
Test Used     Independent T-Test
Alternative   two-sided
Statistic     2.3059
p-value       0.02123
A mean        0.531
B mean        0.486
A normal?     True
A norm p      0.43254
A test        Shapiro
B normal?     True
B norm p      0.28463
B test        Shapiro
Var hom?      True
Var p         0.64102
Significant?  True
```

