import pandas as pd
from score_card_model.score_card_func import ScoreCardModel
from utils import chi2_cutting_discrete, chi2_cutting_continuous, feat_bins_split

if __name__ == '__main__':
    df_data = pd.read_excel("./test.xlsx")
    df_data = df_data.drop('id', axis=1)

    scm_obj = ScoreCardModel(df=df_data, target='loan_status')

    """
    'sp_vals_cols_disc': {{'sec_app_revol_util': [-1]},
                           'const_cols_ratio': 0.9,
                           'max_intervals': 5,
                           'min_pnt': 0.05,
                           'idx_cols_disc_ord': {'emp_length': {'00': 0, '01': 1, '02': 2, '03': 3, '04': 4,
                                                                '05': 5, '06': 6, '07': 7, '08': 8, '09': 9,
                                                                '10': 10}},
    """
    pipe_config = {'sp_vals_cols': {'id': [-1], 'dti': [-1],
                                    'mths_since_last_delinq': [-1],
                                    'mths_since_last_record': [-1],
                                    'mths_since_last_major_derog': [-1]},
                   'const_cols_ratio': 0.9,
                   'max_intervals': 5,
                   'min_pnt': 0.05,
                   'idx_cols_disc_ord': {'emp_length': {'00': 0, '01': 1, '02': 2, '03': 3, '04': 4,
                                                        '05': 5, '06': 6, '07': 7, '08': 8, '09': 9,
                                                        '10': 10}},
                          }
    scm_obj.model_pineline_proc(pipe_config=pipe_config)

    df_woe = scm_obj.df_woe
    print(df_woe.shape)

    a = scm_obj.filter_df_woe_iv(df_woe=df_woe)
    df_woe = df_woe.drop(a, axis=1)
    print(df_woe.shape)

    a = scm_obj.filter_df_woe_corr(df_woe=df_woe)
    df_woe = df_woe.drop(a, axis=1)
    print(df_woe.shape)

    a = scm_obj.filter_df_woe_vif(df_woe=df_woe)
    df_woe = df_woe.drop(a, axis=1)
    print(df_woe.shape)

    a = scm_obj.filter_df_woe_pvalue(df_woe=df_woe)
    df_woe = df_woe.drop(a, axis=1)
    print(df_woe.shape)






