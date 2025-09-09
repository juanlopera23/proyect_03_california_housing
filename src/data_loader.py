from sklearn.datasets import fetch_california_housing
import pandas as pd

def load_california_housing():
    california=fetch_california_housing(as_frame=True)

    return california.frame