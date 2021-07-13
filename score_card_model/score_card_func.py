import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import warnings
from tqdm import tqdm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib.ticker import MultipleLocator
from utils import chi2_cutting_discrete, chi2_cutting_continuous, value_to_intervals
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

    def add_min_pnt(self, min_pnt):
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


    def add_max_intervals(self, max_intervals):
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

    def add_const_cols_ratio(self, const_cols_ratio):
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

    def add_cols_disc_ord(self, idx_cols_disc_ord):
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

    def add_disc_sp_vals(self, sp_vals_cols):
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

    def get_cols_type(self):
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


    def check_target(self):
        """
        简单检查一下Y标的分布是否正确

        """
        if len(self.df[self.target].unique()) <= 1:
            print('Bad Target!!!')
            raise TypeError

    def check_if_has_null(self):
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

    def get_const_cols(self):
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
        self.dict_disc_cols_to_bins, self.dict_disc_iv, self.dict_disc_woe = chi2_cutting_discrete(df_data=self.df,
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
        self.dict_cont_cols_to_bins, self.dict_cont_iv, self.dict_cont_woe = chi2_cutting_continuous(df_data=self.df,
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

    def trans_df_to_woe(self):
        """
        对样本进行woe转化

        Parameters:
        ----------
        self

        Returns:
        -------
        self
        """
        df_woe = self.df.copy()

        for col in tqdm(df_woe.columns, desc='Woe Transforming'):
            # 遍历处理特征, 注意排除target
            if col == self.target:
                continue

            # 特征的映射字典和映射的woe, {val1: 分组序值1, val2: 分组序值2}
            # 分组序值对应的woe值 {分组序值1: woe1, 分组序值2: woe2}
            dict_col_to_bins = self.dict_cols_to_bins[col]
            dict_bins_to_woe = self.dict_woe[col]

            # 开始转化特征
            # 离散型直接转化, 连续型则需要做个转化
            if col in self.cols_cont:
                df_woe[col] = df_woe[col].apply(lambda x: value_to_intervals(value=x, dict_valstoinv=dict_col_to_bins))

            df_woe[col] = df_woe[col].map(dict_col_to_bins).map(dict_bins_to_woe)

        self.df_woe = df_woe

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
                self.add_const_cols_ratio(const_cols_ratio=pipe_config[config])

            elif config == 'min_pnt' and 0 < pipe_config[config] < 1:
                self.add_min_pnt(min_pnt=pipe_config[config])

            elif config == 'idx_cols_disc_ord' and isinstance(pipe_config[config], dict):
                self.add_cols_disc_ord(idx_cols_disc_ord=pipe_config[config])

            elif config == 'sp_vals_cols' and isinstance(pipe_config[config], dict):
                self.add_disc_sp_vals(sp_vals_cols=pipe_config[config])

            elif config == 'max_intervals' and isinstance(pipe_config[config], int):
                self.add_max_intervals(max_intervals=pipe_config[config])

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
                self.check_target()
            elif proc_name == 'Check_None':
                self.check_if_has_null()
            elif proc_name == 'Check_Const_Cols':
                self.get_const_cols()
            elif proc_name == 'Check_Cols_Types':
                self.get_cols_type()
            elif proc_name == 'Add_Mono_Expect':
                self.add_mono_expect()
            elif proc_name == 'Chi2_Cutting':
                self.chi2_cutting()
            elif proc_name == 'Woe_Transform':
                self.trans_df_to_woe()

    def filter_df_woe_iv(self, df_woe, iv_thres=0.01):
        """
        选出基于iv阈值需要踢出的特征名

        Parameters:
        ----------
        iv_thres: float, 最小的IV阈值

        Returns:
        cols_filter: list, 小于IV阈值的特征列表

        """
        cols_filter = []
        for col in df_woe.columns:
            if col != self.target
                # 先获取特征的iv值
                iv_col = self.dict_iv[col]
                # 是否小于阈值
                if iv_col < iv_thres:
                    cols_filter.append(col)

        return cols_filter


    def filter_df_woe_corr(self, corr_thres=0.7):
        pass

    def filter_
