import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error, r2_score

import statsmodels.api as sm


def split_data(df:pd.DataFrame,target:str= "MedHouseVal",test_size:float =0.2, random_state: int = 42):
    
    
    X=df.drop(columns=[target])
    y=df[target]

    return train_test_split(X,y, test_size=test_size,random_state=random_state)


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
  
    df = df.copy()

    if "Population" in df.columns:
        lower_pop = np.percentile(df["Population"], 1)
        upper_pop = np.percentile(df["Population"], 99)
        df["Population"] = df["Population"].clip(lower=lower_pop, upper=upper_pop)


    if "AveOccup" in df.columns:
        df["AveOccup"] = df["AveOccup"].clip(lower=1, upper=10)


    if "AveRooms" in df.columns:
        df["AveRooms"] = df["AveRooms"].clip(lower=1, upper=20)

    if "AveBedrms" in df.columns:
        df["AveBedrms"] = df["AveBedrms"].clip(lower=1, upper=5)

    return df

def calculate_vif (df: pd.DataFrame)->pd.DataFrame:

    df=df.copy()

    vif_data=[]

    for i in range(df.shape[1]):
        x = df.iloc[:, i]
        X = df.drop(df.columns[i], axis=1)
        X = sm.add_constant(X)  
        model = sm.OLS(x, X).fit()
        r2 = model.rsquared
        vif = 1 / (1 - r2) if r2 < 1 else float("inf")
        vif_data.append({"variable": df.columns[i], "VIF": vif})
    
    return pd.DataFrame(vif_data)

def remove_multicolinearity (df:pd.DataFrame, threshold: float = 10.0)->pd.DataFrame:

    df=df.copy()
    dropped=True

    while dropped:

        dropped=False
        vif=[variance_inflation_factor(df.values,i)for i in range(df.shape[1])]
        max_vif=max(vif)


        if max_vif>threshold:

            max_index=vif.index(max_vif)

            print(f"Eliminado {df.columns[max_index]} con VIF = {max_vif:.2f}")
            df=df.drop(df.columns[max_index], axis=1)

            dropped=True
        
    return df

def evaluate_train_test(pipeline,X_train,X_test, y_train, y_test):


    y_predict_train=pipeline.predict(X_train)
    y_predict_test=pipeline.predict(X_test)

    mse_train=mean_squared_error(y_train,y_predict_train)
    rmse_train=np.sqrt(mse_train)
    r2_train=r2_score(y_train,y_predict_train)

    mse_test=mean_squared_error(y_test,y_predict_test)
    rmse_test=np.sqrt(mse_test)
    r2_test=r2_score(y_test,y_predict_test)

    metrics = {
        "Train": {"MSE": mse_train, "RMSE": rmse_train, "R2": r2_train},
        "Test": {"MSE": mse_test, "RMSE": rmse_test, "R2": r2_test}
    }
    return metrics, y_predict_train, y_predict_test

