import pandas as pd
from score_card_model.score_card_func import ScoreCardModel

if __name__ == '__main__':
    df_data = pd.read_excel("./test.xlsx")

    scm_obj = ScoreCardModel(df_data)

    print(scm_obj.add_pinepine('Constant_Columns'))
    print(scm_obj.model_pineline_proc())
    print(scm_obj.pinelines)

