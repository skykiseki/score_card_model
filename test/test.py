import pandas as pd
from score_card_model.score_card_func import ScoreCardModel
from utils import bin_badrate

if __name__ == '__main__':
    df_data = pd.read_excel("./test.xlsx")

    scm_obj = ScoreCardModel(df=df_data,
                             target='loan_status',
                             const_cols_ratio=0.9,
                             cols_disc_ord=['emp_length']

                             )

    print(scm_obj.df.columns.shape)

    scm_obj.add_pinepine('Check_Const_Cols')
    scm_obj.model_pineline_proc()

    print(bin_badrate(df=df_data, col_name='emp_length', target='loan_status'))



