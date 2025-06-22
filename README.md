#  Python Modules Collection

Modular Python tools for data scientists and engineers. Each module covers a specific task like cleaning data, running A/B tests, or generating SQL — all with clean code and clear docs.



## 📦 Modules

| Module         | Description |
|----------------|-------------|
| [`data_cleaner`](./data_cleaner) | A robust DataFrame cleaning and preprocessing class for pandas. Supports missing value handling, outlier detection, formatting, type conversion, and more. |
| [`ab_testing`](./ab_testing)     | A statistical AB testing toolkit to evaluate and compare experimental groups using t-tests, chi-square tests, and non-parametric tests. |
| [`sql_helper`](./sql_helper)     | Converts Excel or CSV data into SQL `CREATE TABLE` and `INSERT` statements. Useful for database seeding and ETL automation. |


##  Quick Start

Each module includes a README.md file with usage instructions, key method references, and example outputs to help you get started quickly.

To get started:

```bash
git clone https://github.com/boychinov/modules.git
cd modules/data_cleaner  # or ab_testing / sql_helper
```

##  Folder Structure

```bash
modules/
│
├── data_cleaner/                 # Data preprocessing tools
│   └── README.md
│
├── ab_testing/                   # AB test framework
│   └── README.md
│   └── quick_start_guide.md
│   
├── sql_helper/                   # SQL statement generator
│   └── README.md
│
└── README.md                     # <-- You are here
```


