import pandas as pd
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn import metrics

online_shoppers = pd.read_csv("online_shoppers_intention.csv")


def generate_EDA_report(df, report_title="EDA_report", file_name="EDA Report.html"):
    """Automate EDA with Pandas Profiler."""
    profile = ProfileReport(df, title=report_title, explorative=True)
    profile.to_file(file_name)


def generate_dummy_columns(df, columns_to_dummy: list, drop_in_place: bool = True):
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


def split_dataset(df, dependent_column, test_set_size_percent=0.20):
    X_train_set, X_test_set, y_train_set, y_test_set = train_test_split(
        df.loc[:, df.columns != dependent_column],
        df[dependent_column],
        test_size=test_set_size_percent,
    )
    split_data = {
        "X_train_set": X_train_set,
        "X_test_set": X_test_set,
        "y_train_set": y_train_set,
        "y_test_set": y_test_set,
    }
    return split_data


def upsample(X_df, y_df):
    sm = SMOTE()
    X_df_oversampled, y_df_oversampled = sm.fit_resample(X_df, y_df.ravel())
    return X_df_oversampled, y_df_oversampled


def create_random_forest(
    X_trn, X_tst, y_trn, y_tst, min_data_points_percent=0.05, estimators=100
):
    min_data_points_per_leaf = int(
        round(len(X_train_oversampled) * min_data_points_percent, 0)
    )
    clf = RandomForestClassifier(
        n_estimators=estimators, min_samples_leaf=min_data_points_per_leaf
    )
    clf.fit(X_trn, y_trn)
    y_predictions = clf.predict(X_tst)

    clf_accuracy = round(metrics.accuracy_score(y_tst, y_predictions) * 100, 2)
    print(f"Overall weighted model accuracy is: {clf_accuracy}%\n")

    clf_confusion_matrix = metrics.confusion_matrix(y_tst, y_predictions)
    clf_confusion_matrix_plot = metrics.plot_confusion_matrix(clf, X_tst, y_tst)
    print(metrics.classification_report(y_tst, y_predictions))


convert_to_dummy = ["Month", "VisitorType"]
online_shoppers = generate_dummy_columns(online_shoppers, convert_to_dummy, False)

convert_to_bool = ["Weekend", "Revenue"]
online_shoppers = generate_int_columns(online_shoppers, convert_to_bool)

split_sets = split_dataset(online_shoppers, "Revenue", test_set_size_percent=0.20)

X_train = split_sets["X_train_set"]
X_test = split_sets["X_test_set"]
y_train = split_sets["y_train_set"]
y_test = split_sets["y_test_set"]

X_train_oversampled, y_train_oversampled = upsample(X_train, y_train)

create_random_forest(X_train_oversampled, X_test, y_train_oversampled, y_test)
