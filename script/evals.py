# evals.py

# numpy + dataframe
import numpy as np
import pandas as pd
import xgboost as xgb

# classic sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score , f1_score , classification_report


# --------------------------
# dataset + feature tweaks
# --------------------------



target = "surge_multiplier"
df_sorted = df.sort_values(["month","day","hour"]).reset_index(drop=True)

# route feature join
df_sorted["route"] = df_sorted["source"] + "_" + df_sorted["destination"]

# weekday
df_sorted["dayofweek"] = pd.to_datetime(
    df_sorted.assign(year=2023)[["year","month","day","hour"]]
).dt.dayofweek

# cyclic encodings
df_sorted["hour_sin"] = np.sin( 2*np.pi*df_sorted["hour"]/24 )
df_sorted["hour_cos"] = np.cos( 2*np.pi*df_sorted["hour"]/24 )
df_sorted["dow_sin"]  = np.sin( 2*np.pi*df_sorted["dayofweek"]/7 )
df_sorted["dow_cos"]  = np.cos( 2*np.pi*df_sorted["dayofweek"]/7 )


# --------------------------
# turn surge into categories
# --------------------------

def surge_to_class(x):
    if x == 1.0: return "no_surge"
    elif x <= 1.25: return "low_surge"
    elif x <= 2.0: return "med_surge"
    else: return "high_surge"

df_sorted["surge_class"] = df_sorted["surge_multiplier"].apply(surge_to_class)


# --------------------------
# features
# --------------------------

num_feats = [
    "distance","temperature","visibility","pressure","cloudCover",
    "distance_log","is_weekend","is_peak_hour",
    "hour_sin","hour_cos","dow_sin","dow_cos"
]

cat_feats = ["cab_type","route"]

X = df_sorted[num_feats + cat_feats]
y = df_sorted["surge_class"]

split_idx = int(0.8*len(df_sorted))
X_train , X_test = X.iloc[:split_idx] , X.iloc[split_idx:]
y_train , y_test = y.iloc[:split_idx] , y.iloc[split_idx:]


# --------------------------
# preprocess + model
# --------------------------

num_pipe = Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
])

cat_pipe = Pipeline([
    ("imputer",SimpleImputer(strategy="constant",fill_value="missing")),
    ("ohe",OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num",num_pipe,num_feats),
    ("cat",cat_pipe,cat_feats)
])


xgb_clf = xgb.XGBClassifier(
    n_estimators = 800 ,
    learning_rate = 0.05 ,
    max_depth = 5 ,
    subsample = 0.8 ,
    colsample_bytree = 0.8 ,
    tree_method = "hist" ,
    random_state = 42 ,
    n_jobs = -1
)


pipeline = Pipeline([
    ("preproc",preprocessor),
    ("xgb",xgb_clf)
])


# --------------------------
# train + eval
# --------------------------

pipeline.fit(X_train,y_train)
preds = pipeline.predict(X_test)

print("Accuracy :" , accuracy_score(y_test,preds))
print("Macro F1 :" , f1_score(y_test,preds,average="macro"))
print(classification_report(y_test,preds))