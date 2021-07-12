import pandas as pd
from score_card_model.score_card_func import ScoreCardModel
from utils import chi2_cutting_discrete

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

    discrete_order = {'emp_length': {'00': 0, '01': 1, '02': 2, '03': 3, '04': 4,
                                     '05': 5, '06': 6, '07': 7, '08': 8, '09': 9,
                                     '10': 10}}

    a, b, c = chi2_cutting_discrete(df_data=df_data,
                                    feat_list=scm_obj.cols_disc_disord_less + scm_obj.cols_disc_ord,
                                    target='loan_status',
                                    special_feat_val={},
                                    max_intervals=scm_obj.max_intervals,
                                    min_pnt=scm_obj.min_pnt,
                                    discrete_order=discrete_order,
                                    mono_expect={'emp_length': {'shape': 'mono','u': False}})





