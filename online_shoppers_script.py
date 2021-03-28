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


convert_to_dummy = ["Month", "VisitorType"]
online_shoppers = generate_dummy_columns(online_shoppers, convert_to_dummy, False)

convert_to_bool = ["Weekend", "Revenue"]
online_shoppers = generate_int_columns(online_shoppers, convert_to_bool)

X_train, X_test, y_train, y_test = train_test_split(
    online_shoppers.loc[:, online_shoppers.columns != "Revenue"],
    online_shoppers["Revenue"],
    test_size=0.20,
)

sm = SMOTE()
X_train_oversampled, y_train_oversampled = sm.fit_sample(
    X_train, y_train.ravel())

min_data_points_per_leaf = int(round(len(X_train_oversampled) * 0.05, 0))
clf = RandomForestClassifier(
    n_estimators=100, min_samples_leaf=min_data_points_per_leaf
)

clf.fit(X_train_oversampled, y_train_oversampled)

y_pred = clf.predict(X_test)

clf_accuracy = round(metrics.accuracy_score(y_test, y_pred) * 100,2)
print(f'Overall weighted model accuracy is: {clf_accuracy}%\n')

# Confusion matrix
clf_confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
clf_confusion_matrix_plot = metrics.plot_confusion_matrix(clf, X_test, y_test)

print(metrics.classification_report(y_test, y_pred))