import statsmodels.api as sm
import numpy as np
import warnings
import matplotlib.pyplot as plt
from . import utils
from tqdm import tqdm
from statsmodels.stats.outliers_influence import variance_inflation_factor
warnings.filterwarnings('ignore')


class ScoreCardModel(object):
    """
    评分卡模型建模类

    Attributes:
    ----------
    df: dataframe,输入的训练集
    df_woe: dataframe, 输出的woe训练集

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

    self.disc_cols_cut: list, 参与无序分箱少离散 & 有序离散分箱的特征,
    self.cont_cols_cut: list, 参与无序分箱多离散 & 连续型分箱的特征

    self.dict_disc_cols_to_bins: dict, 离散特征的分组取值
    self.dict_disc_iv: dict, 离散特征的woe编码后的IV
    self.dict_disc_woe: dict, 离散特征分组后的woe值

    self.dict_cont_cols_to_bins: dict, 连续特征的分组取值
    self.dict_cont_iv: dict, 连续特征的woe编码后的IV
    self.dict_cont_woe: dict, 连续特征分组后的woe值

    self.dict_cols_to_bins: dict, 所有特征的分组取值
    self.dict_iv: dict, 所有特征的IV
    self.dict_woe: dict, 所有特征的woe值

    self.md_feats: list, 入模特征

    """
    def __init__(self, df, target):
        self.df = df
        self.df_woe = None

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
                             'Add_Mono_Expect','Chi2_Cutting', 'Woe_Transform']
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

    def _get_cols_type(self):
        """
        对特征进行分类, 当前为识别col类型进行识别, 但是有序与无序的类别性特征需要手工进行输入,
        object类为类别型特征, 其余为数值型特征

        Parameters:
        ----------

        Returns:
        -------
        self

        """
        for col in self.df.columns:
            # 注意剔除Y标
            if col != self.target:
                col_dtype = self.df[col].dtype
                ## 先整理类别型特征和连续型特征
                if col_dtype == 'O':
                    self.cols_disc.append(col)

                    ### 再整理类别型特征中是属于有序还是无序
                    if col not in self.cols_disc_ord:
                        self.cols_disc_disord.append(col)

                        #### 再整理类别型特征中是属于无序分箱少, 还是属于无序分箱多
                        if len(set(self.df[col])) <= self.max_intervals:
                            self.cols_disc_disord_less.append(col)
                        else:
                            self.cols_disc_disord_more.append(col)
                else:
                    ## 其余的是连续型
                    self.cols_cont.append(col)

    def add_mono_expect(self):
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


    def _check_target(self):
        """
        简单检查一下Y标的分布是否正确

        """
        if len(self.df[self.target].unique()) <= 1:
            print('Bad Target!!!')
            raise TypeError

    def _check_if_has_null(self):
        """
        用于检查输入的dataframe是否有空值
        (原理上不允许出现空值)

        Parameters:
        ----------

        Returns:
        -------
        bool, 是否含有空值元素
        """
        if self.df.isnull().sum().sum() > 0:
            print('Checking None values: None value exists.Please fix your data.')
            raise TypeError
        else:
            print('Checking None values: No None value exists.')

    def _get_const_cols(self):
        """
        获得常值特征, 即特征列某个属性占比超阈值%

        Parameters:
        ----------

        Returns:
        -------
        self
        """
        for col in self.df.columns:
            if col != self.target:
                if any(self.df[col].value_counts(normalize=True) >= self.const_cols_ratio):
                    self.const_cols.append(col)

        self.df = self.df.drop(self.const_cols, axis=1)

    def chi2_cutting(self):

        # 先处理无序分箱少离散特征 & 有序离散特征
        self.disc_cols_cut = self.cols_disc_disord_less + self.cols_disc_ord

        ## 特殊值
        disc_special_cols_vals = {k:v for k,v in self.sp_vals_cols.items() if k in self.disc_cols_cut}

        ## 单调性要求
        disc_mono_expect = {k:v for k,v in self.mono_expect.items() if k in self.disc_cols_cut}

        ## 开始分箱
        self.dict_disc_cols_to_bins, self.dict_disc_iv, self.dict_disc_woe = utils.chi2_cutting_discrete(df_data=self.df,
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
        self.dict_cont_cols_to_bins, self.dict_cont_iv, self.dict_cont_woe = utils.chi2_cutting_continuous(df_data=self.df,
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
        cols = [col for col in df.columns if col in self.df.columns]

        df_bins = df.loc[:, cols]

        for col in df_bins.columns:
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
        df_woe: woe编码后的dataframe
        """
        # 检查在df中的特征, 仅选取进入了分箱过程的列
        df_woe = self.trans_df_to_bins(df=df)

        for col in df_woe.columns:
            # 遍历处理特征, 注意排除target
            if col == self.target:
                continue

            dict_bins_to_woe = self.dict_woe[col]

            df_woe[col] = df_woe[col].map(dict_bins_to_woe)

        return df_woe

    def add_pinepine(self, pipe_name):
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

    def model_pineline_proc(self, pipe_config=None):
        """
        对设定的流水线过程进行逐步操作

        Parameters:
        ----------

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

        # 当前设定第一步检查Y标是否唯一的错误
        self.add_pinepine('Check_Target')

        # 当前设定第二步检查是否为非空
        self.add_pinepine('Check_None')

        # 当前设定第三步检查常值特征
        self.add_pinepine('Check_Const_Cols')

        # 当前设定第四步为获取特征的类型
        self.add_pinepine('Check_Cols_Types')

        # 当前设定第五步为处理特征的单调性要求
        self.add_pinepine('Add_Mono_Expect')

        # 第五步开始卡方分箱
        self.add_pinepine('Chi2_Cutting')

        # 第六步进行woe转换
        self.add_pinepine('Woe_Transform')


        # 开始遍历流程处理
        for proc in self.pinelines:
            proc_name = proc[1]

            if proc_name == 'Check_Target':
                self._check_target()
            elif proc_name == 'Check_None':
                self._check_if_has_null()
            elif proc_name == 'Check_Const_Cols':
                self._get_const_cols()
            elif proc_name == 'Check_Cols_Types':
                self._get_cols_type()
            elif proc_name == 'Add_Mono_Expect':
                self.add_mono_expect()
            elif proc_name == 'Chi2_Cutting':
                self.chi2_cutting()
            elif proc_name == 'Woe_Transform':
                self.df_woe = self.trans_df_to_woe(df=self.df)

    def filter_df_woe_iv(self, df_woe, iv_thres=0.01):
        """
        选出基于iv阈值需要踢出的特征名

        经验而言，iv<=0.02:认为其没有预测性;0.02<iv<=0.1,弱预测性;0.1<iv<=0.2,有一定预测性;iv>0.2,强预测性
        PS:如果一个特征的IV特别高, 这个时候必须要注意检查一下了

        Parameters:
        ----------
        df_woe: dataframe, 输入的训练集(含target)
        iv_thres: float, 最小的IV阈值

        Returns:
        cols_filter: set, 小于IV阈值的特征集合

        """
        cols_filter = set()

        # 剔除target
        df = df_woe.drop(self.target, axis=1)

        for col in df.columns:
            # 先获取特征的iv值
            iv_col = self.dict_iv[col]
            # 是否小于阈值
            if iv_col < iv_thres:
                cols_filter.add(col)

        return cols_filter


    def filter_df_woe_corr(self, df_woe, corr_thres=0.7):
        """
        基于两个特征的斯皮尔逊相关系数进行剔除
        若两个特征之间的相关系数大于阈值(默认为0.7), 则剔除IV较低的那个

        Parameters:
        ----------
        df_woe: dataframe, 输入的训练集(含target)
        corr_thres: float, 相关系数阈值

        Returns:
        -------
        cols_filter: set, 返回需要剔除的特征集合

        """
        cols_filter = set()
        dict_feat_corr = {}

        # 剔除target
        df = df_woe.drop(self.target, axis=1)

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



    def filter_df_woe_vif(self, df_woe, vif_thres=10):
        """
        基于statsmodels.stats.outliers_influence.variance_inflation_factor进行vif分析
        对于共线性问题,可以使用逐一剔除的方法,即先遍历全部特征,尝试去剔除一个特征, 再去统计剔除后的vif是否小于阈值
        如果小于阈值, 则说明剔除该特征有效
        如果存在有多个特征可以进行剔除, 则剔除IV较小的那个特征
        如果遍历了之后,无法找到一个适合的特征进行剔除,则将最小IV的特征进行剔除, 保留高IV特征

        Parameters:
        ----------
        df_woe: dataframe, 输入的训练集(含target)
        vif_thres: float, vif系数阈值

        Returns:
        -------
        cols_filter: set, 返回需要剔除的特征集合

        """
        cols_filter = set()


        # list记录vif大于阈值的特征名称
        list_feats_h_vif = []
        # copy
        df = df_woe.drop(self.target, axis=1)
        # 特征名
        list_featnames = list(df.columns)
        # df转化为矩阵
        ## pandas 1.0.0开始剔除as_matrix
        # mat_vif = df.as_matrix()
        mat_vif = df.values
        # 初始化计算dict_feats_iv_vif & list_feats_h_vif
        for i in range(len(list_featnames)):
            # 获取特征名
            featname = list_featnames[i]
            # 获取特征对应的vif
            feat_vif = variance_inflation_factor(mat_vif, i)
            # 如果vif大于阈值, 则写入这个特征
            if feat_vif >= vif_thres:
                list_feats_h_vif.append(featname)

        # 如果list_feats_h_vif有值, 即存在vif>10的情况, 则进入循环
        while len(list_feats_h_vif) > 0:
            # 先重置list_feats_h_vif
            list_feats_h_vif = []
            # dict记录可以剔除的候选特征以及其IV
            dict_feats_candi = {}
            # 基于IV对特征进行排序
            dict_feativ_order = {k: v for k, v in sorted(self.dict_iv.items(), key=lambda x: x[1]) if
                                 k in list_featnames}
            # 开始遍历每个特征
            for feat in dict_feativ_order.keys():
                # 剔除这个特征
                df0 = df.drop(feat, axis=1)
                # 重新计算vif
                # mat_df0 = df0.as_matrix()
                mat_df0 = df0.values
                list_vif = [variance_inflation_factor(mat_df0, i) for i in range(df0.shape[1])]
                # 如果list_vif中不存在大于阈值vif的情况, 则表示剔除这个特征有效解决共线性
                if max(list_vif) < vif_thres:
                    # 找出该特征的iv
                    iv_feat_candi = dict_feativ_order[feat]
                    # 插入候选字典
                    dict_feats_candi[feat] = iv_feat_candi

            # 取得候选中最小iv的特征
            # 如果遍历了全部特征仍没有办法排除共线性, 则剔除最小iv的特征
            if len(dict_feats_candi.keys()) > 0:
                feat_candi_miniv = min(dict_feats_candi, key=dict_feats_candi.get)
            else:
                feat_candi_miniv = min(dict_feativ_order, key=dict_feativ_order.get)

            # 插入返回的set中
            cols_filter.add(feat_candi_miniv)
            # 剔除该特征
            df = df.drop(feat_candi_miniv, axis=1)
            # mat_vif = df.as_matrix()
            mat_vif = df.values

            # 重新计算
            list_featnames = list(df.columns)
            for i in range(len(list_featnames)):
                # 获取特征名
                featname = list_featnames[i]
                # 获取特征对应的vif
                feat_vif = variance_inflation_factor(mat_vif, i)
                # 如果vif大于阈值, 则写入这个特征
                if feat_vif >= vif_thres:
                    list_feats_h_vif.append(featname)

        return cols_filter

    def filter_df_woe_pvalue(self, df_woe, pval_thres=0.05):
        """
        对回归模型系数进行显著性检验
        类似vif的处理方法逐步回归,先按p_value最高的特征进行剔除,再进行回归,直到所有的系数显著

        Parameters:
        ----------
        df_woe: dataframe, 输入的训练集(含target)
        pval_thres: float, p_value阈值

        Returns:
        -------
        cols_filter: set, 返回需要剔除的特征集合

        """
        cols_filter = set()

        # copy
        df = df_woe.copy()

        # 初始建模, 注意加入常数项
        df['intercept'] = [1] * df.shape[0]
        x = df.drop(self.target, axis=1)
        y = df[self.target]
        model = sm.Logit(y, x)
        results = model.fit()

        # 初始化对应的pvalue字典
        dict_feats_pvalue = results.pvalues.to_dict()
        # 注意要删除截距项
        del dict_feats_pvalue['intercept']

        # 如果存在不显著的系数
        while max(dict_feats_pvalue.values()) > pval_thres:
            # 取得最不显著的特征
            feat_max_pval = max(dict_feats_pvalue, key=dict_feats_pvalue.get)
            # 插入待删除列表
            cols_filter.add(feat_max_pval)
            # 剔除该特征
            df0 = df.drop(feat_max_pval, axis=1)
            # 重新建模
            df0['intercept'] = [1] * df0.shape[0]
            x = df0.drop(self.target, axis=1)
            y = df0[self.target]
            model = sm.Logit(y, x)
            results = model.fit()
            # 重置赋值dict_feats_pvalue
            dict_feats_pvalue = results.pvalues.to_dict()
            # 删除截距项
            del dict_feats_pvalue['intercept']
            # df剔除该特征
            df = df.drop(feat_max_pval, axis=1)

        return cols_filter

    def set_md_features(self, md_feats):
        """

        设置入模特征

        Parameters:
        ----------
        md_feats: list, 入模特征列表

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
                    if feat not in self.df.columns:
                        raise Exception('{0}不存在于候选特征中.'.format(feat))

            self.md_feats = md_feats
            print('设置入模特征{0}个.'.format(len(md_feats)))
        else:
            raise Exception('设置的入模特征个数必须大于0.')


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

    def get_df_scores(self, df_woe, estimator, base_score=500, pdo=20):
        """
        基于入模特征计算dataframe的分数

        Parameters:
        ----------
        df_woe: dataframe,训练集, 注意列必须包含所有的入模特征, 且默认经过了woe编码了

        estimator: model,模型对象,必须包含predict_proba

        base_score: int,基础分

        pdo: int, 20

        Returns:
        -------
        probas: list, 拟合的概率列表

        scores: list, 拟合的分数列表


        """
        # 先检查是否有设置入模特征
        self._check_if_has_md_feats()

        # 再检查df是否含有所有的入模特征
        if not all([col in df_woe.columns for col in self.md_feats]):
            raise Exception('输入的dataframe没有包含所有入模特征.')

        # 选择入模特征
        df = df_woe.loc[:, self.md_feats]

        # 计算proba
        probas = [p[1] for p in estimator.predict_proba(df)]

        # 计算分数
        scores = [self.proba_to_score(proba=p, base_score=base_score, pdo=pdo) for p in probas]

        return probas, scores

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



