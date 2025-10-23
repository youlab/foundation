from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)
from sklearn.metrics import r2_score


def calc_accuracy(
    y_true,
    y_pred,
):
    return (y_true == y_pred).sum() / y_true.shape[0]


def fit_models(
    x_train,
    y_regr_train,
    y_class_train=None,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    n_estimators=100,
    max_features=1.0,
    bootstrap=False,
):
    regr = ExtraTreesRegressor(
        criterion="friedman_mse",
        random_state=42,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        n_estimators=n_estimators,
        max_features=max_features,
        bootstrap=bootstrap,
    )
    
    regr.fit(
        x_train,
        y_regr_train,
    )

    if y_class_train is not None:
        clf = ExtraTreesClassifier(
            criterion="entropy",
            random_state=42,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_estimators=n_estimators,
            max_features=max_features,
            bootstrap=bootstrap,
        )
        clf.fit(
            x_train,
            y_class_train,
        )

        return regr, clf
    
    return regr, None


def predict(
    x_train,
    x_test,
    regr,
    clf,
):
    y_regr_train_pred = regr.predict(x_train)
    y_regr_test_pred = regr.predict(x_test)

    if clf is not None:
        y_class_train_pred = clf.predict(x_train)
        y_class_test_pred = clf.predict(x_test)
    else:
        y_class_train_pred, y_class_test_pred = None, None

    return y_regr_train_pred, y_regr_test_pred, y_class_train_pred, y_class_test_pred


def calc_metrics(
    y_regr_train,
    y_regr_train_pred,
    y_regr_test,
    y_regr_test_pred,
    y_class_train=None,
    y_class_train_pred=None,
    y_class_test=None,
    y_class_test_pred=None,
):
    r2_train = r2_score(
        y_true=y_regr_train,
        y_pred=y_regr_train_pred,
    )

    r2_test = r2_score(
        y_true=y_regr_test,
        y_pred=y_regr_test_pred,
    )

    if y_class_train is not None:
        accuracy_train = calc_accuracy(
            y_true=y_class_train,
            y_pred=y_class_train_pred,
        )

        accuracy_test = calc_accuracy(
            y_true=y_class_test,
            y_pred=y_class_test_pred,
        )
    else:
        accuracy_train, accuracy_test = None, None

    return r2_train, r2_test, accuracy_train, accuracy_test
