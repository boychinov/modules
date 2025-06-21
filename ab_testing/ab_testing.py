import pandas as pd
from scipy.stats import shapiro, normaltest, levene, ttest_ind, mannwhitneyu, chi2_contingency
from tabulate import tabulate

class ABTester:
    def __init__(self):
        """
            Initialize an empty ABTester instance.
        """
        self.df = None
        self.col_name = None
        self.group_col = None
        self._group_a = None
        self._group_b = None

    def load_data(self, df=None, filepath=None, file_type='csv',
                  col_name=None, sep=',', sheet_name=0):
        """
            Load dataset from a file or directly from a DataFrame.

            Parameters:
                df (DataFrame): Optional, directly provided DataFrame.
                filepath (str): Optional, path to file (CSV or Excel).
                file_type (str): File type: 'csv' or 'excel'.
                col_name (str): Column name for A/B testing.
                sep (str): Separator used in CSV files.
                sheet_name (int/str): Sheet name or index for Excel files.
        """

        if df is None and filepath is not None:
            if file_type == 'csv':
                df = pd.read_csv(filepath, sep=sep)
            elif file_type == 'excel':
                df = pd.read_excel(filepath, sheet_name=sheet_name)
            else:
                raise ValueError("file_type must be either 'csv' or 'excel'")
        elif df is not None:
            pass
        else:
            raise ValueError("Either df or filepath must be provided.")

        if col_name is not None and col_name not in df.columns:
            raise KeyError(f"'{col_name}' column not found in data.")

        self.df = df.copy()
        self.col_name = col_name

    def assign_groups(self, group_col=None, group_a_filter=None, group_b_filter=None,label_a='A', label_b='B'):

        """
           Assign groups for A/B testing using either a column or filtering functions.

           Parameters:
               group_col (str): Column name with group labels.
               group_a_filter (function): Function to filter group A.
               group_b_filter (function): Function to filter group B.
               label_a (str): Label for group A.
               label_b (str): Label for group B.
               label_a and label_b are also applied when using custom filters.
        """

        if group_col is not None:
            if group_col not in self.df.columns:
                raise KeyError(f"'{group_col}' column not found in dataframe.")

            self.group_col = group_col
            self.df = self.df.dropna(subset=[group_col, self.col_name])
            self._group_a = self.df[self.df[group_col] == label_a][self.col_name]
            self._group_b = self.df[self.df[group_col] == label_b][self.col_name]

        elif group_a_filter is not None and group_b_filter is not None:
            self.df['Group'] = None
            self.df.loc[group_a_filter(self.df), 'Group'] = label_a
            self.df.loc[group_b_filter(self.df), 'Group'] = label_b
            self.df = self.df[self.df['Group'].notna()]
            self.group_col = 'Group'
            self._group_a = self.df[self.df['Group'] == label_a][self.col_name]
            self._group_b = self.df[self.df['Group'] == label_b][self.col_name]

        else:
            raise ValueError("Either group_col or both group_a_filter and group_b_filter must be provided.")

    def assign_groups_from_files(self, filepath_a, filepath_b, col_name,
                                 file_type='csv', sep=',',
                                 sheet_a=0, sheet_b=0,
                                 label_a='A', label_b='B'):

        """
            Load A and B groups from separate files or sheets and combine into one dataset.

            Parameters:
                filepath_a (str): Path to file for group A.
                filepath_b (str): Path to file for group B.
                col_name (str): Name of the column to compare.
                file_type (str): 'csv' or 'excel'.
                sep (str): Separator for CSV files.
                sheet_a (int/str): Sheet name/index for group A (Excel).
                sheet_b (int/str): Sheet name/index for group B (Excel).
                label_a (str): Label for group A.
                label_b (str): Label for group B.
        """

        self.col_name = col_name
        self.group_col = "Group"

        if file_type == 'csv':
            df_a = pd.read_csv(filepath_a, sep=sep)
            df_b = pd.read_csv(filepath_b, sep=sep)
        elif file_type == 'excel':
            df_a = pd.read_excel(filepath_a, sheet_name=sheet_a)
            df_b = pd.read_excel(filepath_b, sheet_name=sheet_b)
        else:
            raise ValueError("file_type must be either 'csv' or 'excel'")


        if col_name not in df_a.columns or col_name not in df_b.columns:
            raise KeyError(f"'{col_name}' column must be present in both files.")


        df_a = df_a[[col_name]].copy()
        df_b = df_b[[col_name]].copy()
        df_a["Group"] = label_a
        df_b["Group"] = label_b


        self.df = pd.concat([df_a, df_b], ignore_index=True)
        self._group_a = df_a[col_name]
        self._group_b = df_b[col_name]

    @property
    def group_a(self):
        """
                Returns:
                    Series: Values for group A.
        """
        return self._group_a

    @property
    def group_b(self):
        """
            Returns:
                Series: Values for group B.
        """
        return self._group_b

    def is_normal(self, sample_size=5000):
        """
            Test normality of both groups using Shapiro or D’Agostino test depending on sample size.

            Parameters:
                sample_size (int): Threshold to decide which test to use.

            Returns:
                dict: Normality status, p-values, and test types for both groups.
        """

        if not (
                pd.api.types.is_numeric_dtype(self.group_a) and
                pd.api.types.is_numeric_dtype(self.group_b)
        ):
            raise TypeError("Normality test can only be applied to numeric columns.")

        if len(self.group_a) <= sample_size:
            test_a = 'Shapiro'
            _, p1 = shapiro(self.group_a)
        else:
            test_a = 'Normaltest'
            _, p1 = normaltest(self.group_a)
        group_a_normal = p1 > 0.05

        if len(self.group_b) <= sample_size:
            test_b = 'Shapiro'
            _, p2 = shapiro(self.group_b)
        else:
            test_b = 'Normaltest'
            _, p2 = normaltest(self.group_b)
        group_b_normal = p2 > 0.05

        return {
            "group_a_normal": group_a_normal,
            "group_a_p": p1,
            "group_a_test": test_a,
            "group_b_normal": group_b_normal,
            "group_b_p": p2,
            "group_b_test": test_b
        }

    def is_variance_homogeneous(self):
        """
            Test variance homogeneity using Levene's test.

            Returns:
                tuple: (bool, p-value) indicating homogeneity and p-value.
        """

        if not (
                pd.api.types.is_numeric_dtype(self.group_a) and
                pd.api.types.is_numeric_dtype(self.group_b)
        ):
            raise TypeError("Variance homogeneity test can only be applied to numeric columns.")

        stat, p = levene(self.group_a, self.group_b)
        return (p > 0.05, p)

    def apply_ttest(self, alternative='two-sided'):
        """
            Apply independent t-test assuming equal variances.

            Parameters:
                alternative (str): Hypothesis type - 'two-sided', 'greater', or 'less'.

            Returns:
                tuple: Test statistic and p-value.
        """

        if not (
                pd.api.types.is_numeric_dtype(self.group_a) and
                pd.api.types.is_numeric_dtype(self.group_b)
        ):
            raise TypeError("T-Test can only be applied to numeric columns.")

        return ttest_ind(self.group_a, self.group_b, equal_var=True, alternative=alternative)

    def apply_welch_ttest(self, alternative='two-sided'):
        """
            Apply Welch’s t-test assuming unequal variances.

            Parameters:
                alternative (str): Hypothesis type - 'two-sided', 'greater', or 'less'.

            Returns:
                tuple: Test statistic and p-value.
        """

        if not (
                pd.api.types.is_numeric_dtype(self.group_a) and
                pd.api.types.is_numeric_dtype(self.group_b)
        ):
            raise TypeError("Welch's T-Test can only be applied to numeric columns.")

        return ttest_ind(self.group_a, self.group_b, equal_var=False, alternative=alternative)

    def apply_mann_whitney_u(self, alternative='two-sided'):
        """
            Apply non-parametric Mann-Whitney U test.

            Parameters:
                alternative (str): Hypothesis type - 'two-sided', 'greater', or 'less'.

            Returns:
                tuple: Test statistic and p-value.
        """

        if not (
                pd.api.types.is_numeric_dtype(self.group_a) and
                pd.api.types.is_numeric_dtype(self.group_b)
        ):
            raise TypeError("Mann-Whitney U test can only be applied to numeric columns.")

        return mannwhitneyu(self.group_a, self.group_b, alternative=alternative)

    def decide_alternative(self, threshold=0.02):
        """
            Automatically determine the alternative hypothesis based on mean difference.

            Parameters:
                threshold (float): Minimum difference to consider directional hypothesis.

            Returns:
                str: 'greater', 'less', or 'two-sided'.
        """

        if not (
                pd.api.types.is_numeric_dtype(self.group_a) and
                pd.api.types.is_numeric_dtype(self.group_b)
        ):
            raise TypeError("Alternative hypothesis decision requires numeric columns.")

        mean_a = self.group_a.mean()
        mean_b = self.group_b.mean()
        diff = abs(mean_a - mean_b)

        if diff < threshold:
            return 'two-sided'
        elif mean_a > mean_b:
            return 'greater'
        else:
            return 'less'

    def is_ab_test_significant(self, sample_size=5000, alternative='two-sided'):
        """
            Perform appropriate significance test depending on normality and variance homogeneity.

            Parameters:
                sample_size (int): Sample size threshold for normality test.
                alternative (str): Hypothesis type or 'auto'.

            Returns:
                dict: Test results and descriptive statistics.
        """

        if not pd.api.types.is_numeric_dtype(self.df[self.col_name]):
            raise TypeError("This method only supports numeric metrics. "
                            "Use apply_chi_square() for categorical metrics.")

        normality_result = self.is_normal(sample_size)

        group_a_normal = normality_result["group_a_normal"]
        p1 = normality_result["group_a_p"]
        test_a = normality_result["group_a_test"]

        group_b_normal = normality_result["group_b_normal"]
        p2 = normality_result["group_b_p"]
        test_b = normality_result["group_b_test"]

        var_homog, pvar = self.is_variance_homogeneous()

        mean_a = self.group_a.mean()
        mean_b = self.group_b.mean()
        normal = group_a_normal and group_b_normal
        homogeneous = var_homog
        p_var = pvar

        alt = self.decide_alternative() if alternative == 'auto' else alternative

        if normal:
            if homogeneous:
                test_name = "Independent T-Test"
                stat, p = self.apply_ttest(alt)
            else:
                test_name = "Welch’s T-Test"
                stat, p = self.apply_welch_ttest(alt)
        else:
            test_name = "Mann-Whitney U Test"
            stat, p = self.apply_mann_whitney_u(alt)

        result_table = [
            ["Test Used", test_name],
            ["Alternative", alt],
            ["Statistic", round(stat, 4)],
            ["p-value", round(p, 5)],
            ["A mean", round(mean_a, 3)],
            ["B mean", round(mean_b, 3)],
            ["A normal?", group_a_normal],
            ["A norm p", round(p1, 5)],
            ["A test", test_a],
            ["B normal?", group_b_normal],
            ["B norm p", round(p2, 5)],
            ["B test", test_b],
            ["Var hom?", homogeneous],
            ["Var p", round(p_var, 5)],
            ["Significant?", p <= 0.05]
        ]
        print(tabulate(result_table, headers=["Metric", "Value"], tablefmt="simple"))

        return {
            "test": test_name,
            "statistic": stat,
            "p_value": p,
            "alternative": alt,
            "group_a_mean": mean_a,
            "group_b_mean": mean_b,
            "group_a_normal": group_a_normal,
            "group_a_normality_p": p1,
            "group_a_test": test_a,
            "group_b_normal": group_b_normal,
            "group_b_normality_p": p2,
            "group_b_test": test_b,
            "normal_distribution": normal,
            "variance_homogeneous": homogeneous,
            "variance_homogeneity_p": p_var,
            "significant": p <= 0.05
        }

    def apply_chi_square(self, row_col="Group", col=None):
        """
            Perform Chi-Square test of independence between two categorical variables.

            Parameters:
                row_col (str): Name of the row variable.
                col (str): Name of the column variable. Defaults to col_name.

            Returns:
                dict: Chi-Square test results.
        """

        col_to_test = col if col is not None else self.col_name

        if row_col not in self.df.columns or col_to_test not in self.df.columns:
            raise KeyError("Row or Column variable not found in dataframe.")

        contingency_table = pd.crosstab(self.df[row_col], self.df[col_to_test])
        chi2, p, dof, expected = chi2_contingency(contingency_table)

        result_table = [
            ["Test Used", "Chi-Square Test"],
            ["Row Variable", row_col],
            ["Column Variable", col_to_test],
            ["Chi2 Statistic", round(chi2, 4)],
            ["Degrees of Freedom", dof],
            ["P-Value", round(p, 5)],
            ["Significant (p ≤ 0.05)", p <= 0.05]
        ]

        headers = ["METRIC", "VALUE"]
        print('---------------------------  ------------------')
        print(tabulate(result_table, headers=headers, tablefmt="simple"))
        print("\nContingency Table:\n")
        print(contingency_table.to_string())
        print("\nExpected Frequencies:\n")
        print(pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns).round(2))
        print('---------------------------  ------------------')

        return {
            "test_used": "Chi-Square Test",
            "row_variable": row_col,
            "column_variable": col_to_test,
            "chi2_statistic": chi2,
            "degrees_of_freedom": dof,
            "p_value": p,
            "significant": p <= 0.05,
            "expected_freqs": expected,
            "contingency_table": contingency_table
        }


    def auto_test(self):
        """
            Automatically choose and run the correct test (numeric → t-test/MWU, categorical → chi-square).

            Returns:
                dict: Test result.
        """

        if pd.api.types.is_numeric_dtype(self.df[self.col_name]):
            return self.is_ab_test_significant(alternative='auto')
        else:
            return self.apply_chi_square()
