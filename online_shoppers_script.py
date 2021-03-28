import pandas as pd
from pandas_profiling import ProfileReport


online_shoppers = pd.read_csv("online_shoppers_intention.csv")


def generate_EDA_report(df, report_title="EDA_report",
                        file_name="EDA Report.html"):
    """Automate EDA with Pandas Profiler."""
    profile = ProfileReport(df, title=report_title, explorative=True)
    profile.to_file(file_name)


def generate_dummy_columns(df, columns_to_dummy: list,
                           drop_in_place: bool = True):
    """Replaces given columns with dummy columns."""
    df = pd.get_dummies(df, columns=columns_to_dummy, drop_first=drop_in_place)
    return df


def generate_int_columns(df, columns_to_convert_to_int: list):
    """Convert a given list of columns to type int."""
    try:
        df[columns_to_convert_to_int] = df[columns_to_convert_to_int].astype(int)
        return df
    except ValueError:
        print("Error! Unable to convert a column to int.")


convert_to_dummy = ["Month", "VisitorType"]
#online_shoppers = generate_dummy_columns(online_shoppers, convert_to_dummy,
#                                         False)

convert_to_bool = ["Weekend", "Revenue", "VisitorType"]
online_shoppers = generate_int_columns(online_shoppers, convert_to_bool)


