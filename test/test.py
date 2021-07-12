import pandas as pd


if __name__ == '__main__':
    df_data = pd.read_excel("./LoanStats_2018Q2.csv")
    df_data.head()