import pandas as pd

def display_data_info(data, n=5):
    """
    Display various information about the dataset.

    Args:
        data (DataFrame): The dataset to analyze.
        n (int): Number of rows to display (default is 5).
    """
    
    # Display top rows
    print(f"Top {n} rows of data:")
    print(data.head(n))
    print("--------------------------------------------------------------------------------------------------------------")

    # Display bottom rows
    print(f"Bottom {n} rows of data:")
    print(data.tail(n))
    print("--------------------------------------------------------------------------------------------------------------")

    # Display random rows
    print(f"Random {n} rows of data:")
    print(data.sample(n))
    print("--------------------------------------------------------------------------------------------------------------")

    # Display shape
    print("Shape of the dataset:")
    print(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
    print("--------------------------------------------------------------------------------------------------------------")

    # Display columns
    print("Columns of the dataset:")
    print(data.columns)
    print("--------------------------------------------------------------------------------------------------------------")

    # Display info
    print("Info about the dataset:")
    print(data.info())
    print("--------------------------------------------------------------------------------------------------------------")

    # Display descriptive statistics
    print("Descriptive statistics of the dataset:")
    print(data.describe())
    print("--------------------------------------------------------------------------------------------------------------")

    # Display number of duplicate rows
    print("No of duplicated rows in the dataset:")
    print(data.duplicated().sum())
    print("--------------------------------------------------------------------------------------------------------------")

    # Calculate and display missing value percentages
    missing_percentage = data.isnull().mean() * 100
    columns_with_missing = missing_percentage[missing_percentage > 0]
    print("Percentage of missing values in each column:")
    print(columns_with_missing)


