from sklearn.datasets import fetch_california_housing
import pandas as pd

def load_california_housing(as_frame: bool= True)-> pd.DataFrame:
    california=fetch_california_housing(as_frame=as_frame)

    return california.frame