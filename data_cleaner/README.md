# DataCleaner

`DataCleaner` is a lightweight and chainable Python class for preprocessing and cleaning pandas DataFrames. It includes utilities for handling missing values, standardizing column names, formatting values, detecting and transforming data types, and exporting cleaned data.

## Features
- ✅ Missing value imputation (mean/median/mode/constant)
- ✅ Detection of numeric or string columns with missing values
- ✅ Winsorization and outlier detection
- ✅ Standardization of column names and string formats
- ✅ Boolean and binary conversion
- ✅ Value casing (lower/upper/capitalize)
- ✅ Duplicate and null threshold removal
- ✅ Date parsing and conversion
- ✅ Export to Excel or CSV
- ✅ Pipeline support for chaining methods

##  Installation

Install required dependencies:

```bash
pip install pandas scipy
```

Optional: For ASCII support in `standardize_col_names`:
```bash
pip install unidecode
```

## Quick Start
```python
from data_cleaner import DataCleaner
import pandas as pd

df = pd.read_csv("raw_data.csv")
cleaner = DataCleaner(df)

cleaner.pipeline([
    ("remove_duplicates", {}),
    ("fill_missing", {"method": "median"}),
    ("convert_cols_yes_no_to_int", {}),
    ("standardize_col_names", {"ascii_only": True}),
    ("to_csv", {"filepath": "cleaned.csv"})
])

cleaned_df = cleaner.get_df()
```


## Key Methods

### Initialization
```python
DataCleaner(df=None)
```
- Initializes with optional `df`. If not provided, starts with empty DataFrame.

### Missing Value Handling
```python
.fill_missing(columns=None, method="mean")
.fill_missing_strings(method="mode", fill_value="Unknown", cols=None)
.drop_cols_with_missing(threshold=0.5)
.drop_rows_with_missing(threshold=0.5)
```

### Column & Value Formatting
```python
.remove_duplicates()
.upper_col_names()
.lower_col_names()
.capitalize_col_names()
.lower_values(col_list)
.upper_values(col_list)
.capitalize_values(col_list)
```

### Outlier Handling
```python
.detect_outliers(col_name, multiplier=1.5)
.analyze_outliers(col_name, multiplier=1.5)
.winsorize_column(col, lower_pct=0.01, upper_pct=0.01)
```

### Type Conversion & Detection
```python
.convert_cols_bool_to_int()
.convert_cols_yes_no_to_int(cols=None)
.detect_cols_yes_no_like()
.convert_cols_to_numeric(cols)
.detect_cols_to_numeric()
```

### Date Handling
```python
.convert_dates(date_col, new_col="converted_date")
```

### Column Name Cleaning
```python
.standardize_col_names(remove_special=True, ascii_only=False, case="lower", separator="_")
```

### Export Methods
```python
.to_excel(filepath="cleaned_data.xlsx")
.to_csv(filepath="cleaned_data.csv")
```

### Pipeline
```python
.pipeline(steps)
```
- Execute a sequence of transformations with optional kwargs per method.


## Return Convention
All transformation methods return `self`, enabling fluent method chaining:
```python
cleaner.fill_missing().drop_cols_with_missing().convert_cols_to_numeric(cols)
```

### 📜 Example Output

Before cleaning:

```plaintext
   name   age  salary   subscribed
0  John  25.0  50000.0        Yes
1  Anna   NaN     NaN         No
2  Mike  30.0  54000.0        yes
3  Sara   NaN  58000.0        NO
4  Emma  28.0     NaN         Y
```

Applied steps:

```python
clean_df = (DataCleaner(df)
    .fill_missing(method="mean")
    .convert_cols_yes_no_to_int(["subscribed"])
    .get_df())
```

After cleaning:

```plaintext
   name   age   salary  subscribed
0  John  25.0  50000.0           1
1  Anna  27.7  54000.0           0
2  Mike  30.0  54000.0           1
3  Sara  27.7  58000.0           0
4  Emma  28.0  54000.0           1
```



