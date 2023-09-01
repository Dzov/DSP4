from sklearn.base import BaseEstimator, TransformerMixin
import re
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
)
from sklearn.preprocessing import (
    RobustScaler,
    OneHotEncoder,
    TargetEncoder,
)
from sklearn.metrics import (
    make_scorer,
    mean_squared_error,
    r2_score,
    mean_absolute_error,
)
import scipy as sp

from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression

import time
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer, make_column_selector
import re


def split(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.unique_categories = [
            "Leisure",
            "MedicalFacility",
            "School",
            "ServiceFacility",
            "Store",
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoded_array = encoder.fit_transform(X)

        labels = [category for sublist in encoder.categories_ for category in sublist]

        df = pd.DataFrame(encoded_array, columns=labels)
        grouped = df.groupby(level=0, axis=1).sum()

        try:
            categorized_properties = grouped[self.unique_categories]
        except KeyError as e:
            key = re.findall(r"'(.*?)'", e.args[0])[0]
            grouped[key] = 0
            categorized_properties = grouped[self.unique_categories]

        return np.array(categorized_properties.applymap(lambda x: 1 if x > 1 else x))


def get_linear_regression_scores(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model = model.fit(X_train, y_train)

    train_score = cross_val_score(model, X_train, y_train, cv=5).mean()
    test_score = cross_val_score(model, X_test, y_test, cv=5).mean()
    print("Train R2 score: ", train_score)
    print(
        "Train RMSE score: ",
        np.sqrt(
            -(
                cross_val_score(
                    model, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
                ).mean()
            )
        ),
    )
    print("Test R2 score: ", test_score)
    print(
        "Test RMSE score: ",
        np.sqrt(
            -(
                cross_val_score(
                    model, X_test, y_test, cv=5, scoring="neg_mean_squared_error"
                ).mean()
            )
        ),
    )


def apply_custom_encoder(X_train, X_test, data):
    encoder = CustomOneHotEncoder()
    encoded_matrix = encoder.fit_transform(
        X_train[data.select_dtypes(exclude=np.number).columns]
    )
    labels = [category for category in encoder.unique_categories]

    df = pd.DataFrame(encoded_matrix, columns=labels)
    X_train = X_train.reset_index()
    X_train = X_train.join(df)
    X_train.drop(
        columns=[
            "index",
            "PrimaryPropertyType",
            "LargestPropertyUseType",
            "SecondLargestPropertyUseType",
            "ThirdLargestPropertyUseType",
        ],
        inplace=True,
    )

    encoded_matrix = encoder.transform(
        X_test[data.select_dtypes(exclude=np.number).columns]
    )
    labels = [category for category in encoder.unique_categories]

    df = pd.DataFrame(encoded_matrix, columns=labels)
    X_test = X_test.reset_index()
    X_test = X_test.join(df)
    X_test.drop(
        columns=[
            "index",
            "PrimaryPropertyType",
            "LargestPropertyUseType",
            "SecondLargestPropertyUseType",
            "ThirdLargestPropertyUseType",
        ],
        inplace=True,
    )

    return X_train, X_test


def apply_onehot_encoder(X_train, X_test, data):
    encoder = OneHotEncoder()
    encoded_matrix = encoder.fit_transform(
        X_train[data.select_dtypes(exclude=np.number).columns]
    ).toarray()

    labels = [category for sublist in encoder.categories_ for category in sublist]

    df = pd.DataFrame(encoded_matrix, columns=labels)

    X_train = X_train.reset_index()
    X_train = X_train.join(df)
    X_train.drop(
        columns=[
            "index",
            "PrimaryPropertyType",
            "LargestPropertyUseType",
            "SecondLargestPropertyUseType",
            "ThirdLargestPropertyUseType",
        ],
        inplace=True,
    )

    encoded_matrix = encoder.transform(
        X_test[data.select_dtypes(exclude=np.number).columns]
    ).toarray()

    labels = [category for sublist in encoder.categories_ for category in sublist]

    df = pd.DataFrame(encoded_matrix, columns=labels)

    X_test = X_test.reset_index()
    X_test = X_test.join(df)
    X_test.drop(
        columns=[
            "index",
            "PrimaryPropertyType",
            "LargestPropertyUseType",
            "SecondLargestPropertyUseType",
            "ThirdLargestPropertyUseType",
        ],
        inplace=True,
    )

    return X_train, X_test


def get_preprocessor(name):
    categorical_features = make_column_selector(dtype_include=object)

    preprocessors = {
        "onehot_encoder": (
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_features,
        ),
        "custom_onehot": (CustomOneHotEncoder(), categorical_features),
        "target_encoder": (TargetEncoder(), categorical_features),
    }

    return preprocessors.get(name)


def compare_pipelines(X, y, model):
    pipelines = {}
    scores = pd.DataFrame()

    for encoder in ["onehot_encoder", "target_encoder", "custom_onehot"]:
        for selector in ["KBest", "Variance"]:
            if selector == "KBest":
                selector_step = SelectKBest(k=10, score_func=f_regression)
            elif selector == "Variance":
                selector_step = VarianceThreshold(threshold=0.01)

            pipeline = make_pipeline(
                make_column_transformer(
                    get_preprocessor(encoder), remainder="passthrough"
                ),
                RobustScaler(),
                selector_step,
                model,
            )
            steps = f"{encoder}_{selector}"
            pipelines[steps] = pipeline

    for steps, pipeline in pipelines.items():
        score = cross_val_score(pipeline, X, y, cv=5).mean()
        scores = pd.concat(
            [
                scores,
                pd.DataFrame({"Pipleline Steps": steps, "R2": score}, index=[0]),
            ],
            ignore_index=True,
        ).sort_values(by="R2", ascending=False)
    ax = sns.barplot(x=scores["R2"], y=scores["Pipleline Steps"])
    for p in ax.patches:
        ax.annotate(
            format(p.get_width(), ".2f"),
            (
                p.get_x() + p.get_width() / 1.1,
                p.get_y() + p.get_height(),
            ),
            xytext=(0, 15),
            color="white",
            textcoords="offset points",
        )
    plt.show()


def build_pipeline(model, selector=None):
    if selector:
        return make_pipeline(
            make_column_transformer(
                get_preprocessor("onehot_encoder"), remainder="passthrough"
            ),
            RobustScaler(),
            selector,
            model,
        )
    return make_pipeline(
        make_column_transformer(
            get_preprocessor("onehot_encoder"), remainder="passthrough"
        ),
        RobustScaler(),
        model,
    )


def grid_search(X, y, model, params, comparison_df, selector=None):
    (X_train, X_test, y_train, y_test) = split(X, y)
    scoring = {
        "r2": make_scorer(r2_score),
        "mean_squared_error": make_scorer(mean_squared_error),
        "mean_absolute_error": make_scorer(mean_absolute_error),
    }

    start = time.time()
    grid = GridSearchCV(
        build_pipeline(model, selector),
        params,
        cv=5,
        scoring=scoring,
        refit="mean_squared_error",
    )
    grid.fit(X_train, y_train)
    y_predicted = grid.predict(X_test)
    elapsed_time = time.time() - start

    results = {
        "Model": str(model),
        "Selector": str(selector),
        "Test RMSE": round(np.sqrt(mean_squared_error(y_test, y_predicted)), 4),
        "Train Mean R2": round(grid.cv_results_["mean_test_r2"].mean(), 2),
        "Train Mean MAE": round(
            (grid.cv_results_["mean_test_mean_absolute_error"].mean()), 4
        ),
        "Train Mean RMSE": round(
            (np.sqrt((grid.cv_results_["mean_test_mean_squared_error"].mean()))), 4
        ),
        "Best params": str(grid.best_params_),
        "Mean Fit Time": round(grid.cv_results_["mean_fit_time"].mean(), 2),
        "Runtime": round(elapsed_time, 2),
    }

    comparison_df = pd.concat(
        [comparison_df, pd.DataFrame(results, index=[0])], ignore_index=True
    ).sort_values(by="Test RMSE")

    display(comparison_df)
    return comparison_df


def plot_predictions(y_test, predictions):
    actual_values = np.power(10, y_test) - 1
    predicted_values = np.power(10, predictions) - 1
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))

    axes[0].scatter(y_test, predictions, label="Predictions")
    axes[0].plot(
        [min(y_test), max(y_test)],
        [min(y_test), max(y_test)],
        color="red",
        linestyle="--",
        label="Ideal Model",
    )
    axes[0].set_xlabel("Actual (Log)")
    axes[0].set_ylabel("Predicted (Log)")
    axes[0].set_title("Predictions vs Actual Energy Use Log Values")
    axes[0].legend()

    axes[1].scatter(actual_values, predicted_values, label="Predictions")
    axes[1].plot(
        [min(actual_values), max(actual_values)],
        [min(actual_values), max(actual_values)],
        color="red",
        linestyle="--",
        label="Ideal Model",
    )
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predicted")
    axes[1].set_title("Predictions vs Actual Energy Use Values")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
