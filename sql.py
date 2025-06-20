import pandas as pd

class SQLHelper:
    """
    A helper class for generating SQL statements (CREATE TABLE and INSERT) from Excel or CSV data.

    Notes:
        - The class processes all data in memory and generates one SQL INSERT per row.
        - It works efficiently with datasets up to ~100,000 rows.
        - For larger datasets (e.g., 500,000+ rows), the resulting SQL file may become too large
          for SQL clients like SSMS (SQL Server Management Studio) to execute reliably.
          This may lead to 'out of memory' or freezing issues during execution.
        - For big datasets, consider batching output or using bulk insert methods instead.

    Attributes:
        table_name (str): The name of the SQL table.
        df (DataFrame): The loaded dataset as a pandas DataFrame.
    """

    def __init__(self, table_name):
        """
        Initialize the SQLHelper with a target table name.

        Args:
            table_name (str): Name of the SQL table to generate statements for.
        """
        self.table_name = table_name
        self.df = None  # store dataframe once

    def load_data(self, file_path, sheet_name=None, file_type="excel", sep=","):
        """
        Load data from an Excel or CSV file into a DataFrame and fill NaN values with "NULL".

        Args:
            file_path (str): Path to the file.
            sheet_name (str, optional): Sheet name (only for Excel).
            file_type (str): Type of file: "excel" or "csv". Default is "excel".
            sep (str): Separator used in CSV file. Default is ",".

        Returns:
            DataFrame: The loaded and cleaned DataFrame.
        """
        if file_type == "csv":
            self.df = pd.read_csv(file_path, dtype=str, sep=sep)
        else:
            self.df = pd.read_excel(file_path, sheet_name=sheet_name, dtype=str)

        self.df.fillna("NULL", inplace=True)
        return self.df

    def _format_value(self, value, forced_type=None):
        """
        Format a single cell value into a valid SQL-compatible string, based on its type.

        Args:
            value: The cell value.
            forced_type (str, optional): One of ['string', 'integer', 'float', 'date'].

        Returns:
            str: SQL-formatted value (with quotes if needed).
        """
        if pd.isna(value) or str(value).strip().upper() == "NULL":
            return "NULL"

        if forced_type == "date":
            try:
                date_obj = pd.to_datetime(value, errors='coerce')
                if pd.notna(date_obj):
                    return f"'{date_obj.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}'"
            except:
                return "NULL"

        if forced_type == "integer":
            try:
                return str(int(float(value)))
            except:
                return "NULL"

        if forced_type == "float":
            try:
                return str(float(value))
            except:
                return "NULL"

        if forced_type == "string":
            return "'" + str(value).replace("'", "''") + "'"

        # fallback auto-detect
        try:
            date_obj = pd.to_datetime(value, errors='coerce')
            if pd.notna(date_obj):
                return f"'{date_obj.strftime('%Y-%m-%d')}'"
        except:
            pass

        try:
            num = float(value)
            return str(int(num)) if num.is_integer() else str(num)
        except:
            pass

        return "'" + str(value).replace("'", "''") + "'"

    def create_insert_statements(self, column_types=None):
        """
        Generate SQL INSERT statements for all rows in the DataFrame.

        Args:
            column_types (dict, optional): Dictionary mapping column names to types.

        Returns:
            list of str: SQL INSERT statements.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Use load_data() first.")

        df = self.df
        columns = list(df.columns)
        statements = []

        for _, row in df.iterrows():
            values = []
            for col_name, value in row.items():
                col_type = column_types.get(col_name) if column_types else None
                formatted = self._format_value(value, forced_type=col_type)
                values.append(formatted)
            sql = f"INSERT INTO {self.table_name} ({', '.join(columns)}) VALUES({', '.join(values)});"
            statements.append(sql)

        return statements

    def create_table_statement(self, column_types, add_auto_id_column=True):
        """
        Generate a SQL CREATE TABLE statement based on detected or given column types.

        Args:
            column_types (dict): Mapping of column names to data types.
            add_auto_id_column (bool): Whether to add an ID INT IDENTITY primary key.

        Returns:
            str: The CREATE TABLE SQL statement.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Use load_data() first.")

        df = self.df
        sql_lines = [f"CREATE TABLE {self.table_name} ("]
        lines = []

        if add_auto_id_column:
            lines.append("    ID INT IDENTITY(1,1) PRIMARY KEY")

        for column, col_type in column_types.items():
            col_data = df[column].dropna().astype(str)

            if col_type == "string":
                max_len = col_data.map(len).max() if not col_data.empty else 1
                sql_type = f"VARCHAR({max_len})"

            elif col_type == "integer":
                try:
                    nums = pd.to_numeric(col_data, errors="coerce").dropna()
                    if nums.empty:
                        sql_type = "INT"
                    elif nums.max() <= 255:
                        sql_type = "TINYINT"
                    elif nums.max() <= 32767:
                        sql_type = "SMALLINT"
                    else:
                        sql_type = "INT"
                except:
                    sql_type = "INT"

            elif col_type == "float":
                sql_type = "FLOAT"

            elif col_type == "date":
                if col_data.str.contains(r"\d{2}:\d{2}:\d{2}", regex=True).any():
                    sql_type = "DATETIME2(3)"
                else:
                    sql_type = "DATE"

            else:
                sql_type = "VARCHAR(255)"

            lines.append(f"    {column} {sql_type}")

        sql_lines.append(",\n".join(lines))
        sql_lines.append(");")
        return "\n".join(sql_lines)

    def detect_column_types(self):
        """
        Automatically detect column data types (string, integer, float, or date) based on sample values.

        Returns:
            dict: Column name to type mapping.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Use load_data() first.")

        column_types = {}

        for col in self.df.columns:
            sample_values = self.df[col].dropna().astype(str).head(10)
            guessed_type = "string"

            datetime_formats = [
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
                "%d/%m/%Y",
                "%m/%d/%Y",
                "%b %d, %Y"
            ]

            for fmt in datetime_formats:
                try:
                    parsed = pd.to_datetime(sample_values, format=fmt, errors="coerce")
                    if parsed.notna().sum() >= len(sample_values) * 0.8:
                        guessed_type = "date"
                        break
                except:
                    continue

            if guessed_type == "string":
                try:
                    nums = pd.to_numeric(sample_values, errors="coerce")
                    if nums.notna().sum() >= len(sample_values) * 0.8:
                        guessed_type = "integer" if (nums % 1 == 0).all() else "float"
                except:
                    pass

            column_types[col] = guessed_type

        return column_types

    def export_create_table_sql(self, column_types, filepath="create_table.sql", add_auto_id_column=True):
        """
        Export the CREATE TABLE SQL statement to a file.

        Args:
            column_types (dict): Column types to use in table definition.
            filepath (str): Output file path.
            add_auto_id_column (bool): Whether to add auto-increment ID column.
        """
        sql = self.create_table_statement(column_types, add_auto_id_column=add_auto_id_column)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(sql + "\n")

        print(f" CREATE TABLE SQL written to: {filepath}")

    def export_insert_statements_sql(self, column_types=None, filepath="insert_statements.sql"):
        """
        Export the generated INSERT INTO SQL statements to a file.

        Args:
            column_types (dict, optional): Column types to guide formatting.
            filepath (str): Output file path.
        """
        insert_statements = self.create_insert_statements(column_types=column_types)
        with open(filepath, "w", encoding="utf-8") as f:
            for stmt in insert_statements:
                f.write(stmt + "\n")

        print(f" INSERT statements written to: {filepath}")
