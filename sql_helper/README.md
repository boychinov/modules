# SQLHelper 

A lightweight Python class to convert Excel or CSV datasets into SQL `CREATE TABLE` and `INSERT INTO` statements. Ideal for quickly prototyping, migrating, or exporting structured tabular data into relational databases.


##  Features

- ✅ Load data from Excel or CSV into pandas
- ✅ Auto-detect column types: `string`, `integer`, `float`, `date`
- ✅ Generate `CREATE TABLE` statements with intelligent type guessing
- ✅ Generate `INSERT INTO` statements (row-by-row)
- ✅ Export SQL files for both schema and data
- ✅ Supports null handling and SQL-safe string formatting

---

##  Installation

```bash
pip install pandas openpyxl
```


##  Quick Start

```python
from sql import SQLHelper

# Initialize the helper with the name of the SQL table you want to generate
helper = SQLHelper("my_table")

# Load data from an Excel or CSV file (Excel by default)
# Automatically fills missing values with NULL
helper.load_data("data.xlsx")

# Detect column data types from sample values
# Possible types: 'string', 'integer', 'float', 'date'
column_types = helper.detect_column_types()

# Export CREATE TABLE SQL statement to a file
# Adds ID INT IDENTITY(1,1) PRIMARY KEY by default
helper.export_create_table_sql(column_types)

# Export INSERT INTO statements row-by-row to a file
# Handles formatting and SQL escaping internally
helper.export_insert_statements_sql(column_types)
```


##  Available Methods

### `load_data(...)`

Loads Excel or CSV file into a pandas DataFrame and replaces missing values with `NULL`.

### `detect_column_types()`

Analyzes sample values to auto-detect column types (`string`, `integer`, `float`, `date`).

### `create_table_statement(...)`

Creates SQL `CREATE TABLE` statement based on given or detected types.

### `create_insert_statements(...)`

Generates SQL `INSERT INTO` statements row by row.

### `export_create_table_sql(...)`

Writes `CREATE TABLE` SQL to file.

### `export_insert_statements_sql(...)`

Writes all `INSERT INTO` statements to file.


##  Performance Notes

- Efficient up to **\~100,000 rows**
- For large datasets (>500,000 rows), exporting one INSERT per row may cause SQL tools (e.g., SSMS) to freeze
- Use batching or bulk insert tools for production-scale loads


##  Example Output (CREATE TABLE)

```sql
CREATE TABLE my_table (
    ID INT IDENTITY(1,1) PRIMARY KEY,
    name VARCHAR(50),
    age SMALLINT,
    salary FLOAT,
    join_date DATE
);
```


##  Example Output (INSERT INTO)

```sql
INSERT INTO my_table (name, age, salary, join_date) VALUES('Alice', 30, 55000.0, '2020-01-15');
INSERT INTO my_table (name, age, salary, join_date) VALUES('Bob', 45, 72000.0, '2018-06-01');
INSERT INTO my_table (name, age, salary, join_date) VALUES('Charlie', NULL, NULL, NULL);
```



