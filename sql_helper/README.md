# SQLHelper

A lightweight Python class for converting Excel or CSV data into SQL `CREATE TABLE` and `INSERT` statements.


---


## Features


-  Auto-detects column types (`string`, `integer`, `float`, `date`)
-  Generates `CREATE TABLE` SQL statement with correct data types and lengths
-  Generates `INSERT INTO` SQL statements for all rows
-  Handles `NULL` values and SQL-safe string formatting
-  Supports both Excel and CSV inputs
- 

---


## Usage


-Python

from sql_helper import SQLHelper

helper = SQLHelper("my_table")

df = helper.load_data("my_file.xlsx")

column_types = helper.detect_column_types()

#### Export CREATE TABLE SQL
helper.export_create_table_sql(column_types)

#### Export INSERT INTO statements
helper.export_insert_statements_sql(column_types)


---


-Output

Two .sql files will be created in the current directory:

•	create_table.sql
•	insert_statements.sql


---


-Requirements

•	Python 3.x
•	pandas

