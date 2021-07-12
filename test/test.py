import pandas as pd
from score_card_model.score_card_func import ScoreCardModel

if __name__ == '__main__':
    df_data = pd.read_excel("./test.xlsx")

    scm_obj = ScoreCardModel(df=df_data,
                             const_cols_ratio=0.9)

    print(scm_obj.df.columns.shape)

    scm_obj.add_pinepine('Check_Const_Cols')
    scm_obj.model_pineline_proc()

    print(scm_obj.df_res.columns.shape)

