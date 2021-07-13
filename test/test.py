import pandas as pd
from score_card_model.score_card_func import ScoreCardModel
from utils import chi2_cutting_discrete, chi2_cutting_continuous, feat_bins_split

if __name__ == '__main__':
    df_data = pd.read_excel("./test.xlsx")
    df_data = df_data.drop('id', axis=1)

    scm_obj = ScoreCardModel(df=df_data, target='loan_status')

    """
    'sp_vals_cols_disc': {},
                           'const_cols_ratio': 0.9,
                           'max_intervals': 5,
                           'min_pnt': 0.05,
                           'idx_cols_disc_ord': {'emp_length': {'00': 0, '01': 1, '02': 2, '03': 3, '04': 4,
                                                                '05': 5, '06': 6, '07': 7, '08': 8, '09': 9,
                                                                '10': 10}},
    """
    pipe_config = {'sp_vals_cols_disc': {},
                           'const_cols_ratio': 0.9,
                           'max_intervals': 5,
                           'min_pnt': 0.05,
                           'idx_cols_disc_ord': {'emp_length': {'00': 0, '01': 1, '02': 2, '03': 3, '04': 4,
                                                                '05': 5, '06': 6, '07': 7, '08': 8, '09': 9,
                                                                '10': 10}},
                          }
    scm_obj.model_pineline_proc(pipe_config=pipe_config)

    print(scm_obj.sp_vals_cols_disc)
    print(scm_obj.const_cols_ratio, scm_obj.max_intervals, scm_obj.min_pnt)
    print('ss' + str(scm_obj.max_intervals))
    print(scm_obj.idx_cols_disc_ord)
    print(scm_obj.cols_disc_ord)
    print(scm_obj.pinelines)
    print(scm_obj.df.shape)

    # a, b, c = chi2_cutting_discrete(df_data=scm_obj.df,
    #                                 feat_list=scm_obj.cols_disc_disord_less + scm_obj.cols_disc_ord,
    #                                 target='loan_status',
    #                                 special_feat_val={},
    #                                 max_intervals=scm_obj.max_intervals,
    #                                 min_pnt=scm_obj.min_pnt,
    #                                 discrete_order=scm_obj.idx_cols_disc_ord,
    #                                 mono_expect={'emp_length': {'shape': 'mono','u': False}})
    #
    # print(a)
    # print(b)
    # print(c)
    #
    # mono_contin = {i: {'shape': 'mono', 'u': False} for i in scm_obj.cols_cont}
    #
    # dict_aa = {}
    # for col in scm_obj.cols_cont:
    #     if -1 in set(df_data[col]):
    #         dict_aa[col] = [-1]
    #
    # d, e, f = chi2_cutting_continuous(df_data=scm_obj.df,
    #                                   feat_list=scm_obj.cols_disc_disord_more + scm_obj.cols_cont,
    #                                   discrete_more_feats=scm_obj.cols_disc_disord_more,
    #                                   target='loan_status',
    #                                   special_feat_val=dict_aa,
    #                                   max_intervals=scm_obj.max_intervals,
    #                                   min_pnt=scm_obj.min_pnt,
    #                                   mono_expect=mono_contin)
    #
    # print(d)
    # print(e)
    # print(f)






