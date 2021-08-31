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

    # GBoost Classifier
    gb_clf = GradientBoostingClassifier
    gb_model = ClassifierModel(df=data, classifier=gb_clf, num_fold=config.NUM_FOLD, parameters=config.gb_params)
    gb_model.train_model()

    # AdaBoost Classifier
    ada_clf = AdaBoostClassifier
    ada_model = ClassifierModel(df=data, classifier=ada_clf, num_fold=config.NUM_FOLD, parameters=config.ada_params)
    ada_model.train_model()

    # Random Forest Classifier
    rf_clf = RandomForestClassifier
    rf_model = ClassifierModel(df=data, classifier=rf_clf, num_fold=config.NUM_FOLD, parameters=config.rf_params)
    rf_model.train_model()

    # Decision Tree Classifier
    dt_clf = DecisionTreeClassifier
    dt_model = ClassifierModel(df=data, classifier=dt_clf, num_fold=config.NUM_FOLD, parameters=config.dt_params)
    dt_model.train_model()

    # ExtraTreesClassifier
    et_clf = ExtraTreesClassifier
    et_model = ClassifierModel(df=data, classifier=et_clf, num_fold=config.NUM_FOLD, parameters=config.et_params)
    et_model.train_model()




