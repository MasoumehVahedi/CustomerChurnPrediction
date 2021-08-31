import time
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold




class ClassifierModel():
    def __init__(self, df, classifier, num_fold, parameters=None):
        self.df = df
        self.classifier = classifier
        self.num_fold = num_fold
        self.parameters = parameters

        self.X_train, self.X_test, self.Y_train, self.Y_test = self.load_data()

    def load_data(self):
        self.x = self.df.drop(["Churn", "customerID"], axis=1)
        self.Y = self.df["Churn"]

        # Normalize data
        scaler = MinMaxScaler()
        scaler.fit(self.x)
        self.X = scaler.transform(self.x)

        # Separate train and test data
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=0.1, random_state=42)

        return X_train, X_test, Y_train, Y_test

    def train_model(self):
        start = time.time()
        # Build the model
        model = self.classifier(**self.parameters)

        # Fit the model
        model.fit(self.X_train, self.Y_train)

        # Get feature importances
        feature_importance = model.feature_importances_
        # Set pandas series to see feature importance
        model_importances = pd.Series(feature_importance,
                                      index=self.x.columns.values)  # x is the first one before normalizing
        print(model_importances)

        end = time.time() - start
        print("Elapsed time to tarin model = {} seconds".format(end))

        # Predict the model
        predictions = model.predict(self.X_test)

        # Evaluation model
        cv = KFold(n_splits=self.num_fold, random_state=42, shuffle=True)
        recall = cross_val_score(model, self.X, self.Y, cv=cv, scoring="recall", n_jobs=-1)
        precision = cross_val_score(model, self.X, self.Y, cv=cv, scoring="precision", n_jobs=-1)
        accuracy = cross_val_score(model, self.X, self.Y, cv=cv, scoring="accuracy", n_jobs=-1)
        f1_score = cross_val_score(model, self.X, self.Y, cv=cv, scoring="f1_macro", n_jobs=-1)
        print("Accuracy = {}".format(accuracy))
        print("Accuracy in average = {}".format(np.mean(accuracy)))
        print("Recall = {}".format(recall))
        print("Precision = {}".format(precision))
        print("f1_score = {}".format(f1_score))


