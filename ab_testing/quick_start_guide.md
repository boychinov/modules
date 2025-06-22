## Quick Start

### Step 1 – Load Your Dataset

```python
from ab_testing import ABTester

ab = ABTester()

# Load your dataset
# You can load your data in two ways:
# 1. From a file (CSV or Excel):
ab.load_data(filepath='data.csv', col_name='purchase_amount', file_type='csv')

# 2. Or directly from an existing pandas DataFrame:
import pandas as pd
df = pd.read_csv('data.csv')
ab.load_data(df=df, col_name='clicks')

# - col_name: The name of the column that contains the metric you want to compare between groups.
#   This should be a numeric (e.g., revenue, number of clicks) or categorical column (e.g., purchase_made).
#   The column is used to calculate group statistics and determine if there is a significant difference.
#   Example values: 'purchase_amount', 'clicks', 'revenue', 'duration', 'score' (numeric)
#   or 'purchase_made', 'subscribed' (categorical).
```

###  Step 2 – Assign Groups

```python
# Method 1: Assign groups using an existing column (e.g., 'variant' containing values like 'A' and 'B')
# - group_col: The name of the column in your dataset that identifies group membership.
#   This column must contain the values you specify with label_a and label_b.
# - label_a / label_b: These are the exact values in the group_col that define Group A and Group B.
#   For example, if group_col='variant', and the column contains 'A' and 'B',
#   you can set label_a='A' and label_b='B'. Rows with 'A' will be used as Group A,
#   and rows with 'B' as Group B. Any other values in the column will be ignored.
ab.assign_groups(group_col='variant', label_a='A', label_b='B')

# Method 2: Assign groups using custom filtering functions
# - group_a_filter / group_b_filter: functions that return a boolean mask for filtering
# - label_a / label_b: what names to assign to filtered groups
# - Note: Even though you can assign custom labels like 'young' and 'adult',
# these are still internally treated as Group A and Group B respectively.
# All subsequent tests and summaries will treat 'young' as Group A and 'adult' as Group B
# based on the label_a and label_b values you define here.
ab.assign_groups(
    group_a_filter=lambda df: df['age'] < 30,
    group_b_filter=lambda df: df['age'] >= 30,
    label_a='young',
    label_b='adult'
)

# Method 3: Load groups from two separate files (CSV or Excel)
# - filepath_a / filepath_b: paths to files containing each group
# - col_name: column to analyze
# - label_a / label_b: labels for each group
# - Note: Each file is assumed to represent a single group entirely.
#   All rows in `filepath_a` are labeled as `label_a`, and all rows in `filepath_b` are labeled as `label_b`.
#   The input files do not need to contain a column for group membership.
#   However, both files must have the same column names and the same structure,
#   especially for the column specified in `col_name`, otherwise an error will occur.
ab.assign_groups_from_files(
    filepath_a='group_a.csv',
    filepath_b='group_b.csv',
    col_name='purchase_amount',
    file_type='csv',
    label_a='A',
    label_b='B'
)
```

### Step 3 – Run the Test

```python
# Run significance test
ab.is_ab_test_significant()
```

