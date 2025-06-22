import pandas as pd
from scipy.stats.mstats import winsorize

class DataCleaner:
    def __init__(self, df=None):
        """
            Initialize a new DataCleaner instance with an optional DataFrame.

            If a DataFrame is provided, a deep copy is created to prevent modifying the original data.
            If no DataFrame is provided, an empty DataFrame is initialized.

            Parameters:
                df (pd.DataFrame, optional): The input DataFrame to be cleaned. Default is None.

            Returns:
                None
        """

        if df is not None:
            self.df = df.copy()
        else:
            self.df = pd.DataFrame()

    def set_data(self, df):
        """
            Set or replace the internal DataFrame used by the DataCleaner instance.

            This method allows assigning a new DataFrame to the cleaner, replacing any previously stored data.
            It creates a deep copy to avoid modifying the original DataFrame outside the class.

            Parameters:
                df (pd.DataFrame): The new DataFrame to be used for cleaning.

            Returns:
                self: Returns the DataCleaner instance for method chaining.
        """
        self.df = df.copy()

        return self

    def get_df(self):
        """
            Return the current internal DataFrame.

            This method provides access to the DataFrame being managed and transformed
            within the DataCleaner instance.

            Example:
                df_clean = cleaner.get_df()

            Returns:
                pd.DataFrame: The internal DataFrame.
        """

        return self.df

    def check_df(self, head=5):
        """
            Display a comprehensive overview of the DataFrame structure and contents.

            This method prints:
                - Shape of the DataFrame (rows, columns)
                - Data types of all columns
                - First and last `head` rows
                - Number of missing values per column
                - Descriptive statistics including extended quantiles (0%, 5%, 50%, 95%, 99%, 100%)

            Parameters:
                head (int): Number of rows to display from the top and bottom. Default is 5.

            Returns:
                None
        """

        print("--------------------- SHAPE ---------------------")
        print(self.df.shape)
        print("---------------------TYPES ---------------------")
        print(self.df.dtypes)
        print("--------------------- HEAD ---------------------")
        print(self.df.head(head))
        print("--------------------- TAIL ---------------------")
        print(self.df.tail(head))
        print("--------------------- NA ---------------------")
        print(self.df.isnull().sum())
        print("--------------------- QUANTILES ---------------------")
        print(self.df.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

    def report_missing(self):
        """
            Generate a summary of missing (null) values for each column.

            This method returns a Series where the index contains column names
            and the values indicate the number of missing entries in each column.

            Returns:
                pd.Series: A Series showing the count of missing values per column.
        """

        return self.df.isnull().sum()

    def detect_numeric_cols_with_missing(self):
        """
        Detect numeric columns that contain missing values.

        Returns:
            list: List of column names with numeric dtype and missing values.
        """
        numeric_cols = self.df.select_dtypes(include='number').columns
        missing_cols = [col for col in numeric_cols if self.df[col].isnull().any()]
        return missing_cols

    def fill_missing(self, columns=None, method="mean"):
        """
            Fill missing values in specified numeric columns using mean, median, or mode.

            Parameters:
                columns (list or None): List of column names to fill. If None, uses all numeric columns.
                method (str): 'mean', 'median', or 'mode'. Default is 'mean'.

            Example:
                cleaner = DataCleaner(df)
                cleaner.fill_missing(columns=["age", "salary"], method="median")

            Returns:
                self: Returns the DataCleaner instance for method chaining.
        """

        if columns is None:
            columns = self.df.select_dtypes(include='number').columns
        else:
            columns = [col for col in columns if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col])]

        for col in columns:
            if method == "mean":
                self.df[col].fillna(self.df[col].mean(), inplace=True)
            elif method == "median":
                self.df[col].fillna(self.df[col].median(), inplace=True)
            elif method == "mode":
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
            else:
                raise ValueError(f"Invalid method '{method}' for fill_missing. Use 'mean', 'median', or 'mode'.")

        return self

    def detect_string_cols_with_missing(self):
        """
        Detect object/string columns that contain missing values.

        Example:
            cleaner = DataCleaner(df)
            missing_str_cols = cleaner.detect_string_cols_with_missing()

        Returns:
            list: List of column names with string dtype and missing values.
        """

        string_cols = self.df.select_dtypes(include='object').columns
        return [col for col in string_cols if self.df[col].isnull().any()]

    def fill_missing_strings(self, method="mode", fill_value="Unknown", cols=None):
        """
            Fill missing values in string/categorical columns.

            This method fills missing (NaN) entries in string or object-type columns using the specified method:
                - "mode": fills with the most frequent value in the column.
                - "constant": fills with a user-defined value (e.g., "Unknown").
                - "ffill" / "bfill": forward-fill or backward-fill missing values.

            Parameters:
                method (str): Filling strategy. One of 'mode', 'constant', 'ffill', 'bfill'.
                fill_value (str): The constant value to use if method='constant'. Default is "Unknown".
                cols (list): Optional list of column names. If None, all string columns with missing values are targeted.

            Example:
                cleaner = DataCleaner(df)
                cleaner.fill_missing_strings(method="constant", fill_value="N/A", cols=["gender", "city"])

            Returns:
                self: Returns the DataCleaner instance for method chaining.
        """

        if cols is None:
            cols = self.detect_string_cols_with_missing()

        for col in cols:
            if col in self.df.columns and self.df[col].isnull().any():
                if method == "mode":
                    mode_series = self.df[col].mode()
                    mode_val = mode_series[0] if not mode_series.empty else fill_value
                    self.df[col].fillna(mode_val, inplace=True)
                elif method == "constant":
                    self.df[col].fillna(fill_value, inplace=True)
                elif method in ["ffill", "bfill"]:
                    self.df[col].fillna(method=method, inplace=True)
                else:
                    raise ValueError(
                        f"Invalid method '{method}' for fill_missing_strings. Use 'mode', 'constant', 'ffill' or 'bfill'.")

        return self


    def drop_cols_with_missing(self, threshold=0.5):
        """
            Drop columns that have more missing values than the specified threshold.

            This method removes columns from the DataFrame if the number of non-null values
            is less than the threshold proportion of total rows.

            Example:
                If threshold=0.5 and there are 100 rows, any column with fewer than 50 non-null values will be dropped.

            Parameters:
                threshold (float): Minimum proportion of non-null values a column must have to be kept (0 to 1). Default is 0.5.

            Returns:
                self: Returns the DataCleaner instance for method chaining.
        """

        limit = int(len(self.df) * threshold)
        before_cols = self.df.columns.tolist()

        self.df.dropna(thresh=limit, axis=1, inplace=True)

        after_cols = self.df.columns.tolist()
        dropped_cols = list(set(before_cols) - set(after_cols))

        print(f"Dropped {len(dropped_cols)} column(s) with more than {100 - threshold * 100:.0f}% missing values.")

        if dropped_cols:
            print(f"Dropped columns:", ", ".join(dropped_cols))
        else:
            print(f"No columns were dropped.")

        return self

    def drop_rows_with_missing(self, threshold=0.5):
        """
            Drop rows that have more missing values than the specified threshold.

            This method calculates the minimum number of non-null values required for a row to be retained.
            If a row contains more missing values than allowed by the threshold, it is dropped.

            Example:
                If threshold=0.5 and there are 10 columns, any row with fewer than 5 non-null values will be dropped.

            Parameters:
                threshold (float): Minimum proportion of non-null values a row must have to be kept (0 to 1). Default is 0.5.

            Returns:
                self: Returns the DataCleaner instance for method chaining.
        """

        limit = int(len(self.df.columns) * threshold)
        before_shape = self.df.shape[0]

        # Find which rows would be dropped
        rows_to_drop = self.df[self.df.count(axis=1) < limit].index.tolist()

        # Drop them
        self.df.drop(index=rows_to_drop, inplace=True)
        after_shape = self.df.shape[0]

        print(f"Dropped {before_shape - after_shape} row(s) with more than {100 - threshold * 100:.0f}% missing values.")

        if rows_to_drop:
            print(f"Dropped row indices: {rows_to_drop}")
        else:
            print(f"No rows were dropped.")

        return self

    def remove_duplicates(self):
        """
            Remove duplicate rows from the DataFrame.

            This method drops all rows that are exact duplicates across all columns,
            keeping the first occurrence by default. It modifies the DataFrame in place.

            Returns:
                self: Returns the DataCleaner instance for method chaining.
        """

        self.df.drop_duplicates(inplace=True)

        return self

    def upper_col_names(self):
        """
            Convert all column names in the DataFrame to uppercase.

            This method applies Python's `str.upper()` to each column name, which can be useful
            for formatting consistency or matching database schemas with uppercase conventions.

            Example:
                'user_id' → 'USER_ID'
                'Age' → 'AGE'

            Returns:
                self: Returns the DataCleaner instance for method chaining.
        """

        upper_list = [col.upper() for col in self.df.columns]
        self.df.columns = upper_list

        return self

    def lower_col_names(self):
        """
            Convert all column names in the DataFrame to lowercase.

            This method applies Python's `str.lower()` to each column name, which is useful
            for standardizing column names before data cleaning, merging, or exporting.

            Example:
                'User_ID' → 'user_id'
                'AGE' → 'age'

            Returns:
                self: Returns the DataCleaner instance for method chaining.
        """

        lower_list = [col.lower() for col in self.df.columns]
        self.df.columns = lower_list

        return self

    def capitalize_col_names(self):
        """
            Capitalize the first letter of each column name.

            This method applies Python's `str.capitalize()` to every column name in the DataFrame,
            making only the first character uppercase and the rest lowercase.

            Example:
                'user_id' → 'User_id'
                'AGE' → 'Age'

            Returns:
                self: Returns the DataCleaner instance for method chaining.
        """

        cap_list = [col.capitalize() for col in self.df.columns]
        self.df.columns = cap_list

        return self

    def lower_values(self, col_list):
        """
            Convert all string values in the specified columns to lowercase.

            This method applies Python's built-in `str.lower()` function to each string value
            in the given columns. Non-string values remain unchanged.

            Example:
                "JOHN" → "john"
                "New York" → "new york"

            Parameters:
                col_list (list): A list of column names to apply lowercase transformation.

            Returns:
                self: Returns the DataCleaner instance for method chaining.
        """

        self.df[col_list] = self.df[col_list].applymap(lambda x: x.lower() if isinstance(x, str) else x)

        return self

    def upper_values(self, col_list):
        """
            Convert all string values in the specified columns to uppercase.

            This method applies Python's built-in `str.upper()` function to each string value
            in the given columns. Non-string values are left unchanged.

            Example:
                "john" → "JOHN"
                "New York" → "NEW YORK"

            Parameters:
                col_list (list): A list of column names to apply uppercase transformation.

            Returns:
                self: Returns the DataCleaner instance for method chaining.
        """

        self.df[col_list] = self.df[col_list].applymap(lambda x: x.upper() if isinstance(x, str) else x)

        return self

    def capitalize_values(self, col_list):
        """
            Capitalize the first letter of string values in the specified columns.

            This method applies Python's built-in `str.capitalize()` function to each string value
            in the given columns. It leaves non-string values unchanged.

            Example:
                "john" → "John"
                "NEW YORK" → "New york"

            Parameters:
                col_list (list): A list of column names to apply capitalization on.

            Returns:
                self: Returns the DataCleaner instance for method chaining.
        """

        self.df[col_list] = self.df[col_list].applymap(lambda x: x.capitalize() if isinstance(x, str) else x)

        return self

    def convert_dates(self, date_col, new_col="converted_date"):
        """
            Convert a column containing various date formats or timestamps into standard datetime format.

            This method handles mixed formats such as:
                - ISO date strings (e.g., '2023-05-01')
                - Unix timestamps (seconds, milliseconds, microseconds, nanoseconds)
                - Numeric strings or integers
                - Empty or null values

            The method creates a new column (default name 'converted_date') containing the parsed datetime values.
            It intelligently detects the likely time unit based on the magnitude of numeric values.

            Parameters:
                date_col (str): The name of the column to convert.
                new_col (str): The name of the output column. Default is 'converted_date'.

            Returns:
                self: Returns the DataCleaner instance for chaining.
        """

        self.df.columns = self.df.columns.str.strip()
        self.df[date_col] = self.df[date_col].replace("", None)
        self.df[new_col] = None

        for i, value in self.df[date_col].items():
            if pd.isna(value):
                self.df.at[i, new_col] = pd.NaT

            elif isinstance(value, int) or (isinstance(value, str) and value.isnumeric()):
                value = int(value)

                if value > 1e18:
                    self.df.at[i, new_col] = pd.to_datetime(value, unit="ns", errors="coerce")
                elif value > 1e15:
                    self.df.at[i, new_col] = pd.to_datetime(value, unit="us", errors="coerce")
                elif value > 1e12:
                    self.df.at[i, new_col] = pd.to_datetime(value, unit="ms", errors="coerce")
                else:
                    self.df.at[i, new_col] = pd.to_datetime(value, unit="s", errors="coerce")

            elif isinstance(value, str):
                try:
                    self.df.at[i, new_col] = pd.to_datetime(value, errors="coerce")  # Convert date strings
                except:
                    self.df.at[i, new_col] = pd.NaT  # Handle invalid dates

        return self

    def detect_outliers(self, col_name, multiplier=1.5):
        """
            Calculate the lower and upper bounds for outliers in a numeric column using the IQR method.

            This method computes the interquartile range (IQR) and returns the threshold values
            for identifying outliers. Any value below the lower bound or above the upper bound
            is considered an outlier.

            Formula:
                IQR = Q3 - Q1
                Lower Bound = Q1 - (IQR * multiplier)
                Upper Bound = Q3 + (IQR * multiplier)

            Parameters:
                col_name (str): Name of the numeric column to analyze.
                multiplier (float): The IQR multiplier to determine the bounds (default is 1.5).

            Returns:
                tuple: (lower_bound, upper_bound) for detecting outliers.
        """

        Q1 = self.df[col_name].quantile(0.25)
        Q3 = self.df[col_name].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR

        return lower, upper

    def analyze_outliers(self, col_name, multiplier=1.5):
        """
            Analyze outliers in a numeric column using the IQR method.

            This method identifies values that fall outside the interquartile range (IQR),
            defined as below Q1 - (IQR * multiplier) or above Q3 + (IQR * multiplier).

            It prints a summary of:
                - Calculated lower and upper outlier thresholds
                - Total number of records
                - Number and percentage of outliers

            Parameters:
                col_name (str): Name of the numeric column to analyze.
                multiplier (float): IQR multiplier to determine outlier bounds (default is 1.5).

            Returns:
                DataFrame: A subset of the original DataFrame containing only the outlier rows.
        """

        lower, upper = self.detect_outliers(col_name, multiplier)

        total = self.df.shape[0]
        outliers = self.df[(self.df[col_name] < lower) | (self.df[col_name] > upper)]
        outlier_count = outliers.shape[0]
        outlier_ratio = outlier_count / total

        print(f"Outlier threshold: [{lower:.2f}, {upper:.2f}]")
        print(f"Total records: {total}")
        print(f"Number of outliers: {outlier_count} ({outlier_ratio:.2%})")

        return outliers

    def winsorize_column(self, col, lower_pct=0.01, upper_pct=0.01):
        """
            Apply winsorization to a numeric column to limit the influence of extreme values.

            This method replaces extreme values in the specified column with values at specified
            lower and upper percentiles. It creates a new column with the suffix "_winsor" that
            contains the winsorized values, while preserving the original column.

            Parameters:
                col (str): Name of the column to winsorize.
                lower_pct (float): Proportion of data to winsorize from the lower end (e.g., 0.01 = 1%).
                upper_pct (float): Proportion of data to winsorize from the upper end (e.g., 0.01 = 1%).

            Returns:
                self: Returns the DataCleaner instance for chaining.
        """

        original = self.df[col]
        winsorized = winsorize(original.dropna(), limits=(lower_pct, upper_pct))
        winsorized_series = pd.Series(data=winsorized, index=original.dropna().index)
        self.df[f"{col}_winsor"] = original.copy()
        self.df.loc[winsorized_series.index, f"{col}_winsor"] = winsorized_series
        
        return self

    def standardize_col_names(self, remove_special=True, ascii_only=False, case="lower", separator="_"):
        """
            Standardize column names by cleaning and formatting.

            Features:
                - Trims whitespace
                - Converts camelCase to snake_case
                - Removes special characters (optional)
                - Converts to ASCII equivalents (optional)
                - Changes case to lower/upper/capitalize
                - Replaces spaces and dashes with custom separator

            Parameters:
                remove_special (bool): Remove non-alphanumeric characters (except spaces). Default is True.
                ascii_only (bool): Convert Unicode characters to closest ASCII equivalents (e.g., 'ç' → 'c'). Requires `unidecode`.
                case (str): One of 'lower', 'upper', 'capitalize'. Controls case of output.
                separator (str): Character to use instead of spaces or dashes. Default is '_'.

            Example:
                cleaner = DataCleaner(df)
                cleaner.standardize_col_names(remove_special=True, ascii_only=True, case="lower", separator="_")

            Returns:
                self: For method chaining.
        """
        import re

        try:
            from unidecode import unidecode
        except ImportError:
            print("Warning: 'unidecode' package not found. ASCII conversion will be skipped.")
            def unidecode(x):
                return x  # fallback: do nothing

        def clean(col):
            col = str(col).strip()

            if ascii_only:
                col = unidecode(col)

            # Convert camelCase to snake_case
            col = re.sub(r'(?<!^)(?=[A-Z])', separator, col)

            # Remove special characters (except separator)
            if remove_special:
                col = re.sub(rf"[^\w\s{re.escape(separator)}]", "", col)

            # Replace spaces and dashes with separator
            col = col.replace(" ", separator).replace("-", separator)

            # Collapse multiple separators
            col = re.sub(rf"{separator}+", separator, col)

            # Final case formatting
            if case == "lower":
                col = col.lower()
            elif case == "upper":
                col = col.upper()
            elif case == "capitalize":
                col = col.capitalize()

            return col

        self.df.columns = [clean(col) for col in self.df.columns]
        return self

    def convert_cols_bool_to_int(self):
        """
            Convert all boolean columns in the DataFrame to integer type (True → 1, False → 0).

            This method automatically detects columns with boolean dtype and converts them to integers.
            Useful for preparing data for modeling or exporting to formats that do not support boolean types.

            Example:
                cleaner = DataCleaner(df)
                cleaner.convert_cols_bool_to_int()

            Returns:
                self: Returns the DataCleaner instance for method chaining.
        """

        self.df[self.df.select_dtypes('bool').columns] = self.df.select_dtypes('bool').astype(int)

        return self

    def convert_cols_yes_no_to_int(self, cols=None):
        """
        Convert binary categorical columns (e.g., yes/no, true/false, 1/0) to integers.

        Parameters:
            cols (list or None): List of columns to convert. If None, automatically detects binary-like columns.

        Example:
            cleaner = DataCleaner(df)
            cleaner.convert_cols_yes_no_to_int(cols=["subscribed", "active"])

        Returns:
                self: Returns the DataCleaner instance for method chaining.

        """
        if cols is None:
            cols = self.detect_cols_yes_no_like()

        for col in cols:
            if col in self.df.columns:
                mapped = (
                    self.df[col]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .map({
                        'yes': 1, 'no': 0,
                        'true': 1, 'false': 0,
                        'y': 1, 'n': 0,
                        '1': 1, '0': 0
                    })
                )
                self.df[col] = pd.to_numeric(mapped, errors="coerce").astype("Int64")

        return self

    def detect_cols_yes_no_like(self):
        """
        Detect columns with binary values that resemble yes/no semantics (e.g., yes/no, y/n, 1/0, true/false).

        Returns:
            list: Column names that contain only two unique values resembling binary yes/no logic.
        """
        yes_no_sets = [
            {"yes", "no"},
            {"y", "n"},
            {"1", "0"},
            {"true", "false"},
            {"t", "f"}
        ]
        detected_cols = []

        for col in self.df.columns:
            if self.df[col].dtype in ['object', 'bool', 'string']:
                # Normalize values
                values = (
                    self.df[col]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .unique()
                    .tolist()
                )
                value_set = set(values)

                for yn_set in yes_no_sets:
                    if value_set.issubset(yn_set):
                        detected_cols.append(col)
                        break

        return detected_cols

    def convert_cols_to_numeric(self, cols):
        """
        Convert specified columns to numeric type (int or float).

        This method attempts to convert the given columns to numeric using `pd.to_numeric()`.
        Any non-convertible values will be replaced with NaN (`errors='coerce'`).

        Parameters:
            cols (list): A list of column names to convert.

        Example:
            cleaner = DataCleaner(df)
            cleaner.convert_cols_to_numeric(["age", "price"])

        Returns:
            self: Returns the DataCleaner instance for chaining.
        """

        for col in cols:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        return self

    def detect_cols_to_numeric(self):
        """
            Detect columns that are likely convertible to numeric types.

            This method scans object or string-type columns and checks whether their values
            can be reliably converted to numeric (integer or float). It avoids columns that:
                - Contain alphabetic characters or currency symbols (e.g., €, $, %),
                - Appear to be categorical (low number of unique values),
                - Resemble date-like formats (e.g., contain dashes or slashes).

            Example:
                cleaner = DataCleaner(df)
                numeric_candidates = cleaner.detect_cols_to_numeric()

            Returns:
                list: A list of column names that are safe to convert to numeric using `pd.to_numeric()`.
        """

        convertible_cols = []

        for col in self.df.columns:
            if self.df[col].dtype.name in ['object', 'string']:
                sample = self.df[col].dropna().astype(str).head(10)

                contains_letters = sample.str.contains(r'[a-zA-Z€$%]', regex=True).any()
                few_unique_values = self.df[col].nunique() < 20
                is_likely_date = sample.str.contains(r"[-/]", regex=True).any()
                all_convertible = pd.to_numeric(sample.str.replace(r"[^\d.]", "", regex=True),
                                                errors="coerce").notna().all()

                if (contains_letters and not all_convertible) or few_unique_values or is_likely_date:
                    continue

            convertible_cols.append(col)

        return convertible_cols



    def pipeline(self, steps):
        """
        Apply a sequence of preprocessing steps to the DataFrame.

        Parameters:
            steps (list of tuples): Each tuple contains the method name (str)
                                    and a dictionary of kwargs (optional).

        Example: [("remove_duplicates", {}), ("fill_missing", {"method": "mean"})]

        Returns:
            self
        """

        for step in steps:
            if isinstance(step, str):
                method_name, kwargs = step, {}
            elif isinstance(step, tuple):
                method_name, kwargs = step
            else:
                raise ValueError("Each step must be a method name or (method_name, kwargs) tuple.")

            if hasattr(self, method_name):
                method = getattr(self, method_name)
                if callable(method):
                    result = method(**kwargs) if kwargs else method()
                    if result is not None:
                        self = result
                else:
                    raise AttributeError(f"'{method_name}' is not callable.")
            else:
                raise AttributeError(f"'{method_name}' is not a method of DataCleaner.")

        return self

    def to_excel(self, filepath="cleaned_data.xlsx", index=False, sheet_name="Sheet1"):
        """
        Export the cleaned DataFrame to an Excel file.

        Parameters:
            filepath (str): Destination file path.
            index (bool): Whether to write row indices.
            sheet_name (str): Excel sheet name.
        Example:
            cleaner.to_excel("output.xlsx")

        Returns:
            self: Returns the DataCleaner instance for method chaining.
        """

        self.df.to_excel(filepath, index=index, sheet_name=sheet_name)
        print(f"Data exported to Excel: {filepath}")

        return self


    def to_csv(self, filepath="cleaned_data.csv", index=False, sep=",", encoding="utf-8"):
        """
        Export the cleaned DataFrame to a CSV file.

        Parameters:
            filepath (str): Destination file path.
            index (bool): Whether to write row indices.
            sep (str): Column separator (default is comma).
            encoding (str): File encoding (default is 'utf-8').
        Example:
            cleaner.to_csv("cleaned.csv", sep=";")

        Returns:
            self: Returns the DataCleaner instance for method chaining.
        """

        self.df.to_csv(filepath, index=index, sep=sep, encoding=encoding)
        print(f"Data exported to CSV: {filepath}")

        return self
