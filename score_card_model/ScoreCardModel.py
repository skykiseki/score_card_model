import statsmodels.api as sm
import numpy as np
import pandas as pd
import warnings
import dill
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from . import utils
from tqdm import tqdm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
warnings.filterwarnings('ignore')


class ScoreCardModel(object):
    """
    评分卡模型建模类

    Attributes:
    ----------
    cols: list, 建模过程中涉及到的columns名称
    target: str, Y标特征

    cols_disc: list,离散型特征
    sp_vals_cols: dict, 某个特征下不参与分箱的特殊值,具体格式如下:
                      {特征名1: [特殊值1...特殊值r], 特征名2: [特殊值1...特殊值o, ......], 特征名k: [特殊值1...特殊值n]}

    cols_disc_ord: list,有序离散型特征列表,

    cols_disc_disord: list,无序离散型特征列表
    cols_disc_disord_less: list, 无序离散型分箱少(小于等于阈值箱数)特征列表
    cols_disc_disord_more: list, 无序离散型分箱多(大于阈值箱数)特征列表

    cols_cont: list, 连续型特征

    max_intervals: int, 最大分箱数
    min_pnt: float, 特征单属性样本最小占比
    mono_expect: dict, 特征的单调性要求, {‘col’: {'shape': 'mono', 'u':False}}

    pipe_options: list, 分别有:
    'Check_Target': 检查Y标
    'Check_None': 检查空值
    'Check_Const_Cols': 剔除常值特征
    'Check_Cols_Types': 获取字段类型

    pinelines: list, 流水线处理列表

    const_cols_ratio: float, 常值字段的阈值%
    const_cols: list, 常值字段

    disc_cols_cut: list, 参与无序分箱少离散 & 有序离散分箱的特征,
    cont_cols_cut: list, 参与无序分箱多离散 & 连续型分箱的特征

    dict_disc_cols_to_bins: dict, 离散特征的分组取值
    dict_disc_iv: dict, 离散特征的woe编码后的IV
    dict_disc_woe: dict, 离散特征分组后的woe值

    dict_cont_cols_to_bins: dict, 连续特征的分组取值
    dict_cont_iv: dict, 连续特征的woe编码后的IV
    dict_cont_woe: dict, 连续特征分组后的woe值

    dict_cols_to_bins: dict, 所有特征的分组取值
    dict_iv: dict, 所有特征的IV
    dict_woe: dict, 所有特征的woe值

    md_feats: list, 入模特征

    estimator: model,模型对象,必须包含predict_proba, 当前版本只支持LR

    estimator_is_fit: bool, 模型是否已经fit

    _coefs: dict, 对应的LR系数, 其中包含截距项


    """
    def __init__(self, target, **estimator_kwargs):

        self.cols = None
        self.target = target

        self.cols_disc = []
        self.sp_vals_cols = {}

        self.cols_disc_ord = []
        self.idx_cols_disc_ord = {}

        self.cols_disc_disord = []
        self.cols_disc_disord_less = []
        self.cols_disc_disord_more = []

        self.cols_cont = []

        self.const_cols_ratio = 0.9
        self.max_intervals = 5
        self.min_pnt = 0.05

        self.mono_expect = None

        self.pipe_options = ['Check_Target', 'Check_None', 'Check_Const_Cols', 'Check_Cols_Types',
                             'Add_Mono_Expect','Chi2_Cutting']
        self.pinelines = []

        self.const_cols = []

        self.disc_cols_cut = []
        self.cont_cols_cut = []

        self.dict_disc_cols_to_bins = {}
        self.dict_disc_iv = {}
        self.dict_disc_woe = {}

        self.dict_cont_cols_to_bins = {}
        self.dict_cont_iv = {}
        self.dict_cont_woe = {}

        self.dict_cols_to_bins = {}
        self.dict_iv = {}
        self.dict_woe = {}

        self.md_feats = None

        # 如果传入的模型对象是None, 则默认用sklearn的逻辑回归
        self.estimator = LogisticRegression(random_state=0,
                                            fit_intercept=True,
                                            n_jobs=-1,
                                            **estimator_kwargs)

        self.estimator_is_fit = False

        self._coefs = {}


    def _add_min_pnt(self, min_pnt):
        """
        添加分箱的样本最小占比

        Parameters:
        ----------
        min_pnt: float, 分箱的样本最小占比

        Returns:
        -------
        self
        """
        self.min_pnt = min_pnt


    def _add_max_intervals(self, max_intervals):
        """
        添加分箱最大分箱数

        Parameters:
        ----------
        max_intervals: int, 分箱的最大分箱数

        Returns:
        -------
        self
        """
        self.max_intervals = max_intervals

    def _add_const_cols_ratio(self, const_cols_ratio):
        """

        添加特征常值占比

        Parameters:
        ----------
        const_cols_ratio:

        Returns:
        -------
        self
        """
        self.const_cols_ratio = const_cols_ratio

    def _add_cols_disc_ord(self, idx_cols_disc_ord):
        """

        添加有序特征, 含其排序

        Parameters:
        ----------
        discrete_order: dict

        e.g.
        discrete_order = {'emp_length': {'00': 0, '01': 1, '02': 2, '03': 3,
                                         '04': 4, '05': 5, '06': 6, '07': 7,
                                         '08': 8, '09': 9, '10': 10}}

        Returns:
        -------
        self
        """
        if len(idx_cols_disc_ord) > 0:
            for k in idx_cols_disc_ord:
                self.cols_disc_ord.append(k)
                self.idx_cols_disc_ord = idx_cols_disc_ord

    def _add_disc_sp_vals(self, sp_vals_cols):
        """
        添加离散型特征的特殊值

        Paramters:
        ---------
        sp_vals_cols: dict, 某个特征下不参与分箱的特殊值,具体格式如下:
                      {特征名1: [特殊值1...特殊值r], 特征名2: [特殊值1...特殊值o, ......], 特征名k: [特殊值1...特殊值n]}

        Returns:
        -------
        self
        """
        self.sp_vals_cols = sp_vals_cols

    def _get_cols_type(self, df):
        """
        对特征进行分类, 当前为识别col类型进行识别, 但是有序与无序的类别性特征需要手工进行输入,
        object类为类别型特征, 其余为数值型特征

        Parameters:
        ----------
        df: dataframe, 待处理的dataframe

        Returns:
        -------
        self

        """
        for col in df.columns:
            # 注意剔除Y标
            if col != self.target:
                col_dtype = df[col].dtype
                ## 先整理类别型特征和连续型特征
                if col_dtype == 'O':
                    self.cols_disc.append(col)

                    ### 再整理类别型特征中是属于有序还是无序
                    if col not in self.cols_disc_ord:
                        self.cols_disc_disord.append(col)

                        #### 再整理类别型特征中是属于无序分箱少, 还是属于无序分箱多
                        if len(set(df[col])) <= self.max_intervals:
                            self.cols_disc_disord_less.append(col)
                        else:
                            self.cols_disc_disord_more.append(col)
                else:
                    ## 其余的是连续型
                    self.cols_cont.append(col)

    def _add_mono_expect(self):
        """
        对有序离散型、连续性特征进行单调性参数整理

        PS: 当前只做单调增或者单调减, U型和倒U暂时不支持(虽然功能我已经开发了)

        Parameters:
        ----------
        self

        Returns:
        -------
        self
        """
        list_cols = self.cols_disc_ord + self.cols_cont

        self.mono_expect = {col:{'shape': 'mono', 'u': False} for col in list_cols}


    def _check_target(self, df):
        """
        简单检查一下Y标的分布是否正确



        Parameters:
        ----------
        df: dataframe, 待处理的dataframe

        """
        if len(df[self.target].unique()) <= 1:
            print('The proportion of target is >= 1')
            raise TypeError

    @staticmethod
    def _check_if_has_null(df):
        """
        用于检查输入的dataframe是否有空值
        (原理上不允许出现空值)

        Parameters:
        ----------
        df: dataframe, 待处理的dataframe

        Returns:
        -------

        """
        if df.isnull().sum().sum() > 0:
            print('Checking None values: None value exists.Please fix your data.')
            raise TypeError
        else:
            print('Checking None values: No None value exists.')


    def _get_const_cols(self, df):
        """
        获得常值特征, 即特征列某个属性占比超阈值%

        Parameters:
        ----------
        df: dataframe, 待处理的dataframe

        Returns:
        -------
        df: dataframe, drop了常值特征后的dataframe
        """
        for col in df.columns:
            if col != self.target:
                if any(df[col].value_counts(normalize=True) >= self.const_cols_ratio):
                    self.const_cols.append(col)

        df = df.drop(self.const_cols, axis=1)

        # 注意这里需要更新涉及的列
        self.cols = [col for col in self.cols if col not in self.const_cols]

        return df

    def chi2_cutting(self, df):
        """
        卡方分箱



        Parameters:
        ----------
        df: dataframe, 待处理的dataframe


        Returns:
        -------

        """

        # 先处理无序分箱少离散特征 & 有序离散特征
        self.disc_cols_cut = self.cols_disc_disord_less + self.cols_disc_ord

        ## 特殊值
        disc_special_cols_vals = {k:v for k,v in self.sp_vals_cols.items() if k in self.disc_cols_cut}

        ## 单调性要求
        disc_mono_expect = {k:v for k,v in self.mono_expect.items() if k in self.disc_cols_cut}

        ## 开始分箱
        self.dict_disc_cols_to_bins, self.dict_disc_iv, self.dict_disc_woe = utils.chi2_cutting_discrete(df_data=df,
                                                                                                         feat_list=self.disc_cols_cut,
                                                                                                         target=self.target,
                                                                                                         special_feat_val=disc_special_cols_vals,
                                                                                                         max_intervals=self.max_intervals,
                                                                                                         min_pnt=self.min_pnt,
                                                                                                         discrete_order=self.idx_cols_disc_ord,
                                                                                                         mono_expect=disc_mono_expect)


        # 开始处理无序分箱多离散特征 & 连续特征
        self.cont_cols_cut = self.cols_disc_disord_more + self.cols_cont

        ## 特殊值
        cont_special_cols_vals = {k:v for k,v in self.sp_vals_cols.items() if k in self.cont_cols_cut}

        ## 单调性要求
        cont_mono_expect = {k:v for k,v in self.mono_expect.items() if k in self.cont_cols_cut}

        ## 开始分箱
        self.dict_cont_cols_to_bins, self.dict_cont_iv, self.dict_cont_woe = utils.chi2_cutting_continuous(df_data=df,
                                                                                                           feat_list=self.cont_cols_cut,
                                                                                                           target=self.target,
                                                                                                           discrete_more_feats=self.cols_disc_disord_more,
                                                                                                           special_feat_val=cont_special_cols_vals,
                                                                                                           max_intervals=self.max_intervals,
                                                                                                           min_pnt=self.min_pnt,
                                                                                                           mono_expect=cont_mono_expect)

        # 分组取值
        self.dict_cols_to_bins.update(self.dict_disc_cols_to_bins)
        self.dict_cols_to_bins.update(self.dict_cont_cols_to_bins)

        # woe
        self.dict_woe.update(self.dict_disc_woe)
        self.dict_woe.update(self.dict_cont_woe)

        # iv
        self.dict_iv.update(self.dict_disc_iv)
        self.dict_iv.update(self.dict_cont_iv)

    def trans_df_to_bins(self, df):
        """
        对样本进行分组

        Parameters:
        ----------
        df:需要做分组转化的dataframe(注意, 这里会仅筛选出进入了分箱过程的特征)


        Returns:
        -------
        df_bins: bins后的dataframe
        """
        # 检查在df中的特征, 仅选取进入了分箱过程的列
        cols = [col for col in df.columns if col in self.cols]

        # 注意把target加回去
        cols.append(self.target)

        df_bins = df.loc[:, cols]

        for col in tqdm(df_bins.columns, desc="cutting bins"):
            # 遍历处理特征, 注意排除target
            if col == self.target:
                continue

            # 特征的映射字典和映射的woe, {val1: 分组序值1, val2: 分组序值2}
            # 分组序值对应的woe值 {分组序值1: woe1, 分组序值2: woe2}
            dict_col_to_bins = self.dict_cols_to_bins[col]

            # 开始转化特征, 离散型直接转化, 连续型则需要做个转化
            if col in self.cols_cont:
                df_bins[col] = df_bins[col].apply(
                    lambda x: utils.value_to_intervals(value=x, dict_valstoinv=dict_col_to_bins))

            df_bins[col] = df_bins[col].map(dict_col_to_bins)

        return df_bins



    def trans_df_to_woe(self, df):
        """
        对样本进行woe转化

        Parameters:
        ----------
        df:需要做woe转化的dataframe(注意, 这里会仅筛选出进入了分箱过程的特征)

        Returns:
        -------
        df_woe: woe编码后的dataframe, 也会保留target

        """
        # 检查在df中的特征, 仅选取进入了分箱过程的列
        df_woe = self.trans_df_to_bins(df=df)

        for col in tqdm(df_woe.columns, desc="transforming the woe dataframe"):
            # 遍历处理特征, 注意排除target
            if col == self.target:
                continue

            dict_bins_to_woe = self.dict_woe[col]

            df_woe[col] = df_woe[col].map(dict_bins_to_woe)

        return df_woe

    def _add_pinepine(self, pipe_name):
        """
        向流水线列表中添加流程名称

        PS: 添加时会自动识别序号, 自0开始计算开始执行

        Parameters:
        ----------
        pipe_name: 流水线处理名称,值为self.pipe_options之一

        Returns:
        -------
        self
        """
        if pipe_name in self.pipe_options:
            self.pinelines.append((len(self.pinelines), pipe_name))
        else:
            print('Back pipeline option.')
            raise TypeError

    def save_model(self, save_path='Score_card_model.pkl'):
        """
        保存模型
        当前只支持用dill进行pkl的封装

        Parameters:
        ----------
        save_path: str, 保存路径

        Returns:
        -------
        """

        dill.dump(self, file=open(save_path, 'wb'))

    def model_pineline_proc(self, df, pipe_config=None):
        """
        对设定的流水线过程进行逐步操作

        Parameters:
        ----------
        df: dataframe, 待处理的dataframe

        pipe_config:dict, {'sp_vals_cols': {},
                           'const_cols_ratio': 0.9,
                           'max_intervals': 5,
                           'min_pnt': 0.05,
                           'idx_cols_disc_ord': {'emp_length': {'00': 0, '01': 1, '02': 2, '03': 3, '04': 4,
                                                                '05': 5, '06': 6, '07': 7, '08': 8, '09': 9,
                                                                '10': 10}},
                          }

        """
        if pipe_config is None:
            pipe_config = {}

        # 处理输入参数
        for config in pipe_config.keys():
            if config == 'const_cols_ratio' and 0 < pipe_config[config] < 1:
                self._add_const_cols_ratio(const_cols_ratio=pipe_config[config])

            elif config == 'min_pnt' and 0 < pipe_config[config] < 1:
                self._add_min_pnt(min_pnt=pipe_config[config])

            elif config == 'idx_cols_disc_ord' and isinstance(pipe_config[config], dict):
                self._add_cols_disc_ord(idx_cols_disc_ord=pipe_config[config])

            elif config == 'sp_vals_cols' and isinstance(pipe_config[config], dict):
                self._add_disc_sp_vals(sp_vals_cols=pipe_config[config])

            elif config == 'max_intervals' and isinstance(pipe_config[config], int):
                self._add_max_intervals(max_intervals=pipe_config[config])

        # 获取列
        self.cols = [col for col in df.columns.tolist() if col != self.target]

        # 当前设定第一步检查Y标是否唯一的错误
        self._add_pinepine('Check_Target')

        # 当前设定第二步检查是否为非空
        self._add_pinepine('Check_None')

        # 当前设定第三步检查常值特征
        self._add_pinepine('Check_Const_Cols')

        # 当前设定第四步为获取特征的类型
        self._add_pinepine('Check_Cols_Types')

        # 当前设定第五步为处理特征的单调性要求
        self._add_pinepine('Add_Mono_Expect')

        # 第五步开始卡方分箱
        self._add_pinepine('Chi2_Cutting')


        # 开始遍历流程处理
        for proc in self.pinelines:
            proc_name = proc[1]

            if proc_name == 'Check_Target':
                self._check_target(df=df)
            elif proc_name == 'Check_None':
                self._check_if_has_null(df=df)
            elif proc_name == 'Check_Const_Cols':
                df = self._get_const_cols(df=df)
            elif proc_name == 'Check_Cols_Types':
                self._get_cols_type(df=df)
            elif proc_name == 'Add_Mono_Expect':
                self._add_mono_expect()
            elif proc_name == 'Chi2_Cutting':
                self.chi2_cutting(df=df)

    def filter_df_woe_iv(self, iv_thres=0.02):
        """
        选出基于iv阈值需要踢出的特征名

        经验而言，iv<=0.02:认为其没有预测性;0.02<iv<=0.1,弱预测性;0.1<iv<=0.2,有一定预测性;iv>0.2,强预测性
        PS:如果一个特征的IV特别高, 这个时候必须要注意检查一下了

        Parameters:
        ----------
        iv_thres: float, 最小的IV阈值

        Returns:
        -------
        cols_filter: set, 小于IV阈值的特征集合

        """
        cols_filter = set()

        for col in self.cols:
            # 先获取特征的iv值
            iv_col = self.dict_iv[col]
            # 是否小于阈值
            if iv_col < iv_thres:
                cols_filter.add(col)

        return cols_filter


    def filter_df_woe_corr(self, df_woe, corr_thres=0.7, frac=0.3):
        """
        基于两个特征的斯皮尔逊相关系数进行剔除
        若两个特征之间的相关系数大于阈值(默认为0.7), 则剔除IV较低的那个

        Parameters:
        ----------
        df_woe: dataframe, 输入的训练集(含target)

        corr_thres: float, 相关系数阈值

        frac: float, 抽样比率

        Returns:
        -------
        cols_filter: set, 返回需要剔除的特征集合

        """
        cols_filter = set()
        dict_feat_corr = {}

        # 先进行抽样
        df_pos = df_woe.loc[df_woe[self.target] == 1].sample(random_state=0, frac=frac)
        df_neg = df_woe.loc[df_woe[self.target] == 0].sample(random_state=0, frac=frac)

        df = pd.concat([df_pos, df_neg]).drop(self.target, axis=1)

        # 计算相关系数, 记得剔除feat本身组合
        # 注意这里含(A, B) 和(B, A）的重复组合
        corr_feat = df.corr()

        for col in corr_feat.columns:
            for idx in corr_feat.index:
                if col != idx:
                    dict_feat_corr[(idx, col)] = corr_feat.loc[idx, col]

        # 找出大于等于相关系数阈值的组合
        # 通过比较两者的IV插入list
        for key, val in dict_feat_corr.items():
            if abs(val) >= corr_thres:
                iv_feat_0 = self.dict_iv[key[0]]
                iv_feat_1 = self.dict_iv[key[1]]
                # 插入IV较小的值
                if iv_feat_0 <= iv_feat_1:
                    cols_filter.add(key[0])
                else:
                    cols_filter.add(key[1])

        return cols_filter



    def filter_df_woe_vif(self, df_woe, vif_thres=10, frac=0.1):
        """
        基于statsmodels.stats.outliers_influence.variance_inflation_factor进行vif分析
        对于共线性问题, 这里使用排列组合的方法, 对所有的特征集合进行遍历, 取出符合不存在共线性的集合

        而当特征个数相等时, 则使用iv平均最大进行取集合

        Parameters:
        ----------
        df_woe: dataframe, 输入的训练集(含target)

        vif_thres: float, vif系数阈值

        frac: float, 小于1的比例, 用于抽样计算加速

        Returns:
        -------
        cols_filter: set, 返回需要剔除的特征集合

        """

        # 原始特征名
        _feats = df_woe.columns.drop(self.target).tolist()

        # 先进行抽样
        df_pos = df_woe.loc[df_woe[self.target] == 1].sample(random_state=0, frac=frac)
        df_neg = df_woe.loc[df_woe[self.target] == 0].sample(random_state=0, frac=frac)

        df = pd.concat([df_pos, df_neg]).drop(self.target, axis=1)

        # 符合vif的iv值计算结果
        dict_vif_iv = {}

        # 符合vif的特征列表计算结果
        feats_vif_lower = []

        # 开始排列组合
        for _n in range(len(_feats), 0, -1):
            for _feats_com in tqdm(combinations(_feats, _n), desc="combinations, n={0}".format(_n)):
                df_feats_com = add_constant(df.loc[:, _feats_com])

                vif_feats_com = pd.Series([variance_inflation_factor(df_feats_com.values, i) for i in range(df_feats_com.shape[1])],
                                          index=df_feats_com.columns)

                # 如果该排列组合符合vif条件, 则计算平均iv后纳入
                if vif_feats_com[list(_feats_com)].max() < vif_thres:
                    iv_feats_com = np.average([self.dict_iv[feat] for feat in _feats_com])

                    dict_vif_iv[len(dict_vif_iv)] = {'feats': _feats_com,
                                                     'iv': np.round(iv_feats_com, 4)}







            if len(dict_vif_iv.keys()) > 0:
                break

        if len(dict_vif_iv.keys()) > 0:
            feats_vif_lower = sorted(dict_vif_iv.items(), key=lambda x: x[1]['iv'], reverse=True)[0][1]['feats']

        cols_filter = [col for col in _feats if col not in feats_vif_lower]

        # # 如果list_feats_h_vif有值, 即存在vif>10的情况, 则进入循环
        # while len(list_feats_h_vif) > 0 and df.shape[1] > 2:
        #     # 先重置list_feats_h_vif
        #     list_feats_h_vif = []
        #
        #     # dict记录可以剔除的候选特征以及其IV
        #     dict_feats_candi = {}
        #
        #     # 基于IV对特征进行排序
        #     dict_feativ_order = {k: v for k, v in sorted(self.dict_iv.items(), key=lambda x: x[1]) if
        #                          k in _feats}
        #     # 开始遍历每个特征
        #     for feat in dict_feativ_order.keys():
        #         # 剔除这个特征
        #         df0 = df.drop(feat, axis=1)
        #
        #         # 重新计算vif
        #         # mat_df0 = df0.as_matrix()
        #         mat_df0 = df0.values
        #         list_vif = [variance_inflation_factor(mat_df0, i) for i in range(df0.shape[1])]
        #
        #         # 如果list_vif中不存在大于阈值vif的情况, 则表示剔除这个特征有效解决共线性
        #         if max(list_vif) < vif_thres:
        #             # 找出该特征的iv
        #             iv_feat_candi = dict_feativ_order[feat]
        #             # 插入候选字典
        #             dict_feats_candi[feat] = iv_feat_candi
        #
        #     # 取得候选中最小iv的特征
        #     # 如果遍历了全部特征仍没有办法排除共线性, 则剔除最小iv的特征
        #     if len(dict_feats_candi.keys()) > 0:
        #         feat_candi_miniv = min(dict_feats_candi, key=dict_feats_candi.get)
        #     else:
        #         feat_candi_miniv = min(dict_feativ_order, key=dict_feativ_order.get)
        #
        #     # 插入返回的set中
        #     cols_filter.add(feat_candi_miniv)
        #
        #     # 剔除该特征
        #     df = df.drop(feat_candi_miniv, axis=1)
        #
        #     # mat_vif = df.as_matrix()
        #     mat_vif = df.values
        #
        #     # 重新计算
        #     list_featnames = list(df.columns)
        #     for i in range(len(list_featnames)):
        #         # 获取特征名
        #         featname = list_featnames[i]
        #
        #         # 获取特征对应的vif
        #         feat_vif = variance_inflation_factor(mat_vif, i)
        #
        #         # 如果vif大于阈值, 则写入这个特征
        #         if feat_vif >= vif_thres:
        #             list_feats_h_vif.append(featname)

        return cols_filter

    def filter_df_woe_pvalue(self, df_woe, pval_thres=0.05, frac=0.1):
        """
        对回归模型系数进行显著性检验
        类似vif的处理方法逐步回归,先按p_value最高的特征进行剔除,再进行回归,直到所有的系数显著

        Parameters:
        ----------
        df_woe: dataframe, 输入的训练集(含target)

        pval_thres: float, p_value阈值

        frac: float, 小于1的比例, 用于抽样计算加速

        Returns:
        -------
        cols_filter: set, 返回需要剔除的特征集合

        """
        # 原始特征名
        _feats = df_woe.columns.drop(self.target).tolist()

        # 先进行抽样
        df_pos = df_woe.loc[df_woe[self.target] == 1].sample(random_state=0, frac=frac)
        df_neg = df_woe.loc[df_woe[self.target] == 0].sample(random_state=0, frac=frac)

        df = pd.concat([df_pos, df_neg])

        # 符合p值的iv值计算结果
        dict_pvalue_iv = {}

        # 符合p值的特征列表计算结果
        feats_pvalue_lower = []

        # 开始排列组合
        for _n in range(len(_feats), 0, -1):
            for _feats_com in tqdm(combinations(_feats, _n), desc="combinations, n={0}".format(_n)):
                # 预建模, 注意加入常数项
                x = df.loc[:, _feats_com]
                x['intercept'] = [1] * x.shape[0]

                y = df[self.target]

                model = sm.Logit(y, x)
                results = model.fit(disp=0)

                # 判断系数是否为正
                is_params_pos = results.params[list(_feats_com)].min() > 0

                # 判断是否都显著
                is_pvalue_pos = results.pvalues[list(_feats_com)].max() < pval_thres

                # 如果该排列组合符合p值和系数条件, 则计算平均iv后纳入
                if is_params_pos and is_pvalue_pos:
                    iv_feats_com = np.average([self.dict_iv[feat] for feat in _feats_com])

                    dict_pvalue_iv[len(dict_pvalue_iv)] = {'feats': _feats_com,
                                                           'iv': np.round(iv_feats_com, 4)}

            if len(dict_pvalue_iv.keys()) > 0:
                break

        if len(dict_pvalue_iv.keys()) > 0:
            feats_pvalue_lower = sorted(dict_pvalue_iv.items(), key=lambda x: x[1]['iv'], reverse=True)[0][1]['feats']

        cols_filter = [col for col in _feats if col not in feats_pvalue_lower]

        # #copy
        # df = df_woe.copy()
        #
        # # 待删除特征set
        # cols_filter = set()
        #
        # # 初始建模, 注意加入常数项
        # x = df.drop(self.target, axis=1)
        # x['intercept'] = [1] * x.shape[0]
        #
        # y = df[self.target]
        #
        # model = sm.Logit(y, x)
        # results = model.fit(disp=0)
        #
        # # 初始化对应的pvalue字典
        # dict_feats_pvalue = results.pvalues.to_dict()
        #
        # # 注意要删除截距项
        # del dict_feats_pvalue['intercept']
        #
        # # 如果存在不显著的系数
        # while max(dict_feats_pvalue.values()) > pval_thres and len(dict_feats_pvalue) > 1:
        #     # 取得最不显著的特征
        #     feat_max_pval = max(dict_feats_pvalue, key=dict_feats_pvalue.get)
        #
        #     # 插入待删除列表
        #     cols_filter.add(feat_max_pval)
        #
        #     # 剔除该特征
        #     df0 = df.drop(feat_max_pval, axis=1)
        #
        #     # 重新建模
        #     x = df0.drop(self.target, axis=1)
        #     x['intercept'] = [1] * x.shape[0]
        #
        #     y = df0[self.target]
        #     model = sm.Logit(y, x)
        #     results = model.fit(disp=0)
        #
        #     # 重置赋值dict_feats_pvalue
        #     dict_feats_pvalue = results.pvalues.to_dict()
        #
        #     # 删除截距项
        #     del dict_feats_pvalue['intercept']
        #
        #     # 彻底删除这个特征
        #     df = df.drop(feat_max_pval, axis=1)

        return cols_filter

    def set_md_features(self, md_feats, df_woe):
        """

        设置入模特征, 同时会启动训练

        Parameters:
        ----------
        md_feats: list, 入模特征列表

        df_woe: dataframe, 训练使用的df_woe

        Returns:
        -------
        self

        """
        if not isinstance(md_feats, list):
            raise Exception('设置的入模特征必须为列表.')

        if len(md_feats) > 0:
            for feat in md_feats:
                if feat == self.target:
                    raise Exception('设置的入模特征不允许为目标变量(Y标).')
                else:
                    if feat not in self.cols:
                        raise Exception('{0}不存在于候选特征中.'.format(feat))

            self.md_feats = md_feats
            print('设置入模特征{0}个.'.format(len(md_feats)))
        else:
            raise Exception('设置的入模特征个数必须大于0.')

        # 这里开始训练
        self.estimator.fit(X=df_woe.loc[:, self.md_feats], y=df_woe[self.target])

        self.estimator_is_fit = True

        # 这里要抓系数
        self._coefs['const'] = self.estimator.intercept_[0]

        for _idx, _feat in enumerate(self.md_feats):
            self._coefs[_feat] = list(self.estimator.coef_[0])[_idx]


    def _check_if_has_md_feats(self):
        """

        检查是否已设置入模特征(列表)

        Parameters:
        ----------

        Returns:
        -------

        """
        if self.md_feats:
            pass
        else:
            raise Exception('未设置入模特征.')

    def gen_feat_to_score(self, md_feats, path_file='Score_bins.xlsx', base_score=500, pdo=20):
        """

        用来生成入模变量的单箱分数表, 最终用于业务使用

        Parameters:
        ----------
        base_score: int, 基础分

        pdo: int, odds提高rate(这里是2)倍时变化的分数

        md_feats: list, 入模特征列表
        """
        # 检查模型是否已经训练过了
        if not self.estimator_is_fit:
            raise Exception('模型还未经过输入,请先使用set_md_features')

        # 计算常数项分数
        const_score = int(base_score - pdo * self._coefs['const'] / np.log(2))

        # 整合各组的woe, iv等
        score_res = []

        for _feat, _bin_group in self.dict_cols_to_bins.items():
            if _feat not in md_feats:
                continue

            woe_group = self.dict_woe[_feat]
            iv_feat = self.dict_iv[_feat]
            coef_feat = self._coefs[_feat]

            for _value, _group_no in _bin_group.items():
                _woe = woe_group[_group_no]

                score_data = {'featname': _feat,
                              'value': _value,
                              'group_no': _group_no,
                              'group_woe': _woe,
                              'group_score': round(-pdo / np.log(2) * coef_feat * _woe),
                              'const_score': const_score,
                              'iv': iv_feat}

                score_res.append(score_data)

        df_score_res = pd.DataFrame(score_res)

        df_score_res.to_excel(path_file, index=False)

        return df_score_res
      
    
    @ staticmethod
    def proba_to_score(proba, base_score=500, pdo=20):
        """
        概率转化为分数

        注意, 这里默认pdo的rate为2倍, 不设置offset,

        Parameters:
        ----------
        proba: float, 正例的概率

        base_score: int, 基础分

        pdo: int,odds提高rate(这里是2)倍时变化的分数

        Returns:
        -------
        score: int, 分数

        """
        # 1e-6是为了平滑分母, 避免分母为0或者分子为0
        if proba == 0:
            proba += 1e-6

        if proba == 1:
            proba -= 1e-6

        odds = proba / (1 - proba + 1e-9)
        score = int(base_score - pdo / np.log(2) * np.log(odds))

        return score

    def get_df_scores(self, df_woe, base_score=500, pdo=20):
        """
        基于入模特征计算dataframe的分数

        Parameters:
        ----------
        df_woe: dataframe,训练集, 注意列必须包含所有的入模特征, 且默认经过了woe编码了

        base_score: int,基础分

        pdo: int, 20

        Returns:
        -------
        labels: list, 拟合的标签列表

        probas: list, 拟合的概率列表

        scores: list, 拟合的分数列表


        """
        # 先检查是否有设置入模特征
        self._check_if_has_md_feats()

        # 检查模型是否已经训练过了
        if not self.estimator_is_fit:
            raise Exception('模型还未经过输入,请先使用set_md_features')

        # 再检查df是否含有所有的入模特征
        if not all([col in df_woe.columns for col in self.md_feats]):
            raise Exception('输入的dataframe没有包含所有入模特征.')

        # 选择入模特征
        df = df_woe.loc[:, self.md_feats]

        # 计算proba
        probas = [p[1] for p in self.estimator.predict_proba(df)]

        # 计算score
        scores = [self.proba_to_score(proba=p, base_score=base_score, pdo=pdo) for p in probas]

        return probas, scores

    def plot_feats_iv(self, feats=[], iv_thres=0.02):
        """
        基于IV进行绘图, 仅绘制特定列的iv

        Parameters:
        ----------
        feats: list, 待绘制的特征了列表

        iv_thres: float, iv基准线

        """
        # 找出iv 大于（含）阈值的特征
        _feats_iv = {_feat:self.dict_iv[_feat] for _feat in feats}

        # 绘图
        fig, ax = plt.subplots(figsize=(10, 10))
        fontdict = {'fontsize': 15}

        dict_feativ_order = {k: v for k, v in sorted(_feats_iv.items(), key=lambda x: x[1])}
        list_featnames = list(dict_feativ_order.keys())
        list_iv = list(dict_feativ_order.values())

        ax.barh(list_featnames, list_iv)
        ax.set_xlabel('IV', fontdict=fontdict)
        ax.set_ylabel('featname', fontdict=fontdict)
        ax.set_title("Feats' IV", fontsize=15)
        ax.tick_params(labelsize='large')
        ax.vlines(x=iv_thres, ymin=-1, ymax=len(list_featnames),
                  linestyles='--', colors='r',
                  label='IV=%.2f' % iv_thres)
        ax.legend()

        # 铺满整个画布
        fig.tight_layout()


    def plot_feats_badrate(self, df, use_cols=None, dict_plot_params=None, factor=None):
        """
        对数据集的各个col绘制badrate分布图

        Parameters:
        ----------
        df: dataframe, 输入的数据集

        use_cols: list, 选择使用的列, 如果为None, 则为全部进行分箱和统计绘图

        dict_plot_params: dict, 绘图的参数

        factor: float

        Returns:
        -------
        Just plot

        """
        # 处理参数
        if dict_plot_params is None:
            dict_plot_params = {'fontsize': 12,
                                'figsize': (15, 8),
                                'linewidth': 3,
                                'markersize': 12,
                                'markeredgewidth': 6}

        if 'fontsize' in dict_plot_params.keys():
            fontsize = dict_plot_params['fontsize']
        else:
            fontsize = 12

        if 'figsize' in dict_plot_params.keys():
            figsize = dict_plot_params['figsize']
        else:
            figsize = (15, 8)

        if 'linewidth' in dict_plot_params.keys():
            linewidth = dict_plot_params['linewidth']
        else:
            linewidth = 3

        if 'markersize' in dict_plot_params.keys():
            markersize = dict_plot_params['markersize']
        else:
            markersize = 12

        if 'markeredgewidth' in dict_plot_params.keys():
            markeredgewidth = dict_plot_params['markeredgewidth']
        else:
            markeredgewidth = 6


        # 先检查target变量是否存在
        if self.target not in df.columns:
            raise Exception('输入的Dataframe不存在目标变量{0}.'.format(self.target))

        # 筛选出使用的特征列
        if use_cols:
            if len(use_cols) == 0:
                raise Exception('输入的参数use_cols为空.')
            elif self.target not in use_cols:
                raise Exception('输入的参数use_cols不包含目标变量{0}.'.format(self.target))
            else:
                df = df.loc[:, use_cols]

        # 对df进行分组
        df_bins = self.trans_df_to_bins(df=df)

        cols = df_bins.columns.drop(self.target).tolist()

        # 初始化画布
        if not factor:
            factor = len(cols) * 0.8

        fontsize *= factor
        figsize = (figsize[0], figsize[1] * factor)
        linewidth *= factor
        markersize *= factor
        markeredgewidth *= factor

        if len(cols) > 1:
            fig, ax = plt.subplots(len(cols), 1, figsize=figsize)
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax = [ax]

        # 开始遍历
        for i in range(len(cols)):
            col = cols[i]

            if col == self.target:
                continue

            # 创建regroup_badrate
            regroup_badrate = utils.bin_badrate(df_bins, col_name=col, target=self.target)

            # 创建分组编号
            regroup_badrate['bin_no'] = regroup_badrate.index

            # 创建样本占比
            regroup_badrate['pnt_feat_vals'] = regroup_badrate['num_feat_vals'] / df.shape[0]

            # 连续型特征需要做区间转换
            if col in self.cols_cont:
                dict_no_to_group = dict(map(reversed, self.dict_cols_to_bins[col].items()))
                regroup_badrate['bins'] = regroup_badrate['bin_no'].map(dict_no_to_group)
            else:
                regroup_badrate['bins'] = regroup_badrate['bin_no']

            # 反转index
            regroup_badrate['bins'] = regroup_badrate['bins'].astype(str)
            regroup_badrate = regroup_badrate.set_index('bins')

            # 注意regroup_badrate需要排序
            regroup_badrate = regroup_badrate.sort_values(by='bin_no')

            # 开始绘图
            fontdict = {'fontsize': fontsize}

            ax[i].bar(regroup_badrate.index, regroup_badrate['pnt_feat_vals'])
            ax[i].set_ylim((0, 1))
            ax[i].set_title("{0}' Badrate".format(col), fontdict=fontdict)

            ax[i].set_xticks(regroup_badrate.index)

            ax[i].set_xlabel('Group', fontdict=fontdict)
            ax[i].set_ylabel('Pct. of bins', fontdict=fontdict)
            ax[i].tick_params(labelsize=fontsize)

            for x, y, z in zip(list(regroup_badrate.index),
                               list(regroup_badrate['pnt_feat_vals'] / 2),
                               list(regroup_badrate['pnt_feat_vals'])):

                ax[i].text(x, y, '{0:.2f}%'.format(z * 100),
                           ha='center',
                           va='center',
                           fontdict=fontdict,
                           color='white')

            # 另外一轴
            ax_twin = ax[i].twinx()
            ax_twin.plot(regroup_badrate.index, regroup_badrate['bad_rate'],
                         linewidth=linewidth,
                         linestyle='--',
                         color='r',
                         marker='x',
                         markersize=markersize,
                         markeredgewidth=markeredgewidth,
                         label='Badrate')
            for x, y, z in zip(list(regroup_badrate.index),
                               list(regroup_badrate['bad_rate']),
                               list(regroup_badrate['bad_rate'])):

                ax_twin.text(x, y + 0.02, '{0:.2f}%'.format(z * 100),
                             ha='center',
                             va='center',
                             fontdict=fontdict,
                             color='r')

            ax_twin.set_ylim((0, regroup_badrate['bad_rate'].max() + 0.1))
            ax_twin.set_ylabel('Badrate', fontdict=fontdict)
            ax_twin.legend(loc='upper left', fontsize=fontsize)

            ax_twin.tick_params(labelsize=fontsize)

        # 铺满整个画布
        fig.tight_layout()



