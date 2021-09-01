import pandas as pd
import numpy as np

import config
from ensemble_model import ClassifierModel

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier



if __name__ == "__main__":
    data = pd.read_csv(config.dataset_dir)
    # Encode Categorical Columns
    labelencoder = LabelEncoder()
    data[config.category_columns] = data[config.category_columns].apply(labelencoder.fit_transform)


    # XGBoost Classifier
    xgb_clf = XGBClassifier
    xgb_model = ClassifierModel(df=data, classifier=xgb_clf, num_fold=config.NUM_FOLD, parameters=config.xgb_params)
    xgb_model.train_model()
    # Evaluation model
    xgb_metrics = xgb_model.evaluate_model()
    xgb_metrics.index = ["XGBoost"]
    results = xgb_metrics

    # GBoost Classifier
    gb_clf = GradientBoostingClassifier
    gb_model = ClassifierModel(df=data, classifier=gb_clf, num_fold=config.NUM_FOLD, parameters=config.gb_params)
    gb_model.train_model()
    # Evaluation model
    gb_metrics = gb_model.evaluate_model()
    gb_metrics.index = ["GBoost"]
    results = results.append(gb_metrics)

    # AdaBoost Classifier
    ada_clf = AdaBoostClassifier
    ada_model = ClassifierModel(df=data, classifier=ada_clf, num_fold=config.NUM_FOLD, parameters=config.ada_params)
    ada_model.train_model()
    # Evaluation model
    ada_metrics = ada_model.evaluate_model()
    ada_metrics.index = ["AdaBoost"]
    results = results.append(ada_metrics)

    # Random Forest Classifier
    rf_clf = RandomForestClassifier
    rf_model = ClassifierModel(df=data, classifier=rf_clf, num_fold=config.NUM_FOLD, parameters=config.rf_params)
    rf_model.train_model()
    # Evaluation model
    rf_metrics = rf_model.evaluate_model()
    rf_metrics.index = ["Random Forest"]
    results = results.append(rf_metrics)

    # Decision Tree Classifier
    dt_clf = DecisionTreeClassifier
    dt_model = ClassifierModel(df=data, classifier=dt_clf, num_fold=config.NUM_FOLD, parameters=config.dt_params)
    dt_model.train_model()
    # Evaluation model
    dt_metrics = dt_model.evaluate_model()
    dt_metrics.index = ["Decision Tree"]
    results = results.append(dt_metrics)

    # ExtraTreesClassifier
    et_clf = ExtraTreesClassifier
    et_model = ClassifierModel(df=data, classifier=et_clf, num_fold=config.NUM_FOLD, parameters=config.et_params)
    et_model.train_model()
    # Evaluation model
    et_metrics = et_model.evaluate_model()
    et_metrics.index = ["ExtraTrees"]
    results = results.append(et_metrics)
    print(results.head())



