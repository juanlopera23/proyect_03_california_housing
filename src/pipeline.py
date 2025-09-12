import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import FunctionTransformer
from src.preprocessing import handle_outliers
from sklearn.linear_model import RidgeCV, LassoCV


def build_pipeline():

    pipeline=Pipeline(steps=[
        ("outliers",FunctionTransformer(handle_outliers,validate=False)),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    return pipeline

def build_pipeline_ridge():
    ridge=Ridge(alpha=0.1)

    ridge_pipe=Pipeline(steps=[
        ("outliers", FunctionTransformer(handle_outliers, validate=False)),
        ("scaler", StandardScaler()),
        ("model", ridge)
    ])
    return ridge_pipe

def build_pipeline_lasso():
    lasso = Lasso(alpha=0.1)
    lasso_pipe = Pipeline(steps=[
        ("outliers", FunctionTransformer(handle_outliers, validate=False)),
        ("scaler", StandardScaler()),
        ("model", lasso)
    ])
    return lasso_pipe

def train_and_evaluate(pipeline,X_train,X_test,y_train,y_test):

    pipeline.fit(X_train,y_train)

    y_pred=pipeline.predict(X_test)

    mse=mean_squared_error(y_test,y_pred)
    rmse=np.sqrt(mse)
    r2=r2_score(y_test,y_pred)

    metrics={
        "MSE": mse,
        "RMSE":rmse,
        "R2":r2
    }

    return metrics, pipeline


def validation_ridge(X_train, y_train):
    ridge_cv = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5).fit(X_train, y_train)
    print("Mejor alpha Ridge:", ridge_cv.alpha_)
    return ridge_cv

def validation_lasso(X_train, y_train):
    y_train = y_train.values.ravel()  
    lasso_cv = LassoCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5, max_iter=5000).fit(X_train, y_train)
    print("Mejor alpha Lasso:", lasso_cv.alpha_)
    return lasso_cv