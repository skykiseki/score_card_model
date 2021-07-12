import pandas as pd
from score_card_model.score_card_func import ScoreCardModel

if __name__ == '__main__':
    df_data = pd.read_excel("./LoanStats_2018Q2.xlsx",
                            nrows=100

    scm_obj = ScoreCardModel(df_data)