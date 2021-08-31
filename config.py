

dataset_dir = "/input/WA_Fn-UseC_-Telco-Customer-Churn.csv"
category_columns = ["Contract", "gender", "Partner",	"Dependents", "PhoneService", "MultipleLines",
                    "InternetService", "OnlineSecurity","OnlineBackup", "DeviceProtection",	"TotalCharges",
                    "TechSupport",	"StreamingTV", "StreamingMovies" , "PaperlessBilling", "PaymentMethod", "Churn"]
NUM_FOLD = 5
xgb_params = {
    "n_estimators": 300,
    "max_depth ": 5,
    "learning_rate": 0.001,
    "min_child_weight": 1,
    "base_score": 0.5,
    "gamma": 0,
    "min_child_weight": 1
}
gb_params = {
    "n_estimators": 300,
    "learning_rate" : 0.008,
}
ada_params = {
    "n_estimators": 400,
    "learning_rate" : 0.01,
}
rf_params = {
    "n_estimators": 300,
    "min_samples_leaf" : 3,
    "max_features" : "sqrt",
}
dt_params = {
    "min_samples_leaf" : 2,
}
et_params = {
    "n_estimators": 300,
    "min_samples_leaf" : 5,
    "min_samples_leaf" : 2,
    "n_jobs" : -1
}