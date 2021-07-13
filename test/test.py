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
    pipe_config = {'sp_vals_cols': {'dti': [-1], 'mths_since_last_delinq': [-1],
                                    'mths_since_last_record': [-1], 'mths_since_last_major_derog': [-1],
                                    'annual_inc_joint': [-1], 'dti_joint': [-1], 'mths_since_rcnt_il': [-1],
                                    'il_util': [-1], 'all_util': [-1], 'bc_open_to_buy': [-1],
                                    'bc_util': [-1], 'mo_sin_old_il_acct': [-1], 'mths_since_recent_bc': [-1],
                                    'mths_since_recent_bc_dlq': [-1], 'mths_since_recent_inq': [-1],
                                    'mths_since_recent_revol_delinq': [-1], 'num_tl_120dpd_2m': [-1],
                                    'percent_bc_gt_75': [-1], 'revol_bal_joint': [-1], 'sec_app_inq_last_6mths': [-1],
                                    'sec_app_mort_acc': [-1], 'sec_app_open_acc': [-1], 'sec_app_revol_util': [-1],
                                    'sec_app_open_act_il': [-1], 'sec_app_num_rev_accts': [-1],
                                    'sec_app_chargeoff_within_12_mths': [-1], 'sec_app_collections_12_mths_ex_med': [-1],
                                    'sec_app_mths_since_last_major_derog': [-1]},
                   'const_cols_ratio': 0.9,
                   'max_intervals': 5,
                   'min_pnt': 0.05,
                   'idx_cols_disc_ord': {'emp_length': {'00': 0, '01': 1, '02': 2, '03': 3, '04': 4,
                                                        '05': 5, '06': 6, '07': 7, '08': 8, '09': 9,
                                                        '10': 10}},
                          }
    scm_obj.model_pineline_proc(pipe_config=pipe_config)

    print(scm_obj.sp_vals_cols)
    print(scm_obj.const_cols_ratio, scm_obj.max_intervals, scm_obj.min_pnt)
    print(scm_obj.idx_cols_disc_ord)
    print(scm_obj.cols_disc_ord)
    print(scm_obj.pinelines)
    print(scm_obj.df.shape)

    print(scm_obj.mono_expect)

    print(scm_obj.dict_disc_cols_to_bins)
    print(scm_obj.dict_disc_iv)
    print(scm_obj.dict_disc_woe)

    print('-' * 80)
    print(scm_obj.dict_cont_cols_to_bins)
    print(scm_obj.dict_cont_iv)
    print(scm_obj.dict_cont_woe)






