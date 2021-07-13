import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib.ticker import MultipleLocator

class ScoreCardModel(object):
    """
    评分卡模型建模类

    Attributes:
    ----------
    df: dataframe,输入的训练集
    df_res: dataframe, 输出的训练集

    target: str, Y标特征

    cols_disc: list,离散型特征

    cols_disc_ord: list,有序离散型特征列表,

    cols_disc_disord: list,无序离散型特征列表
    cols_disc_disord_less: list, 无序离散型分箱少(小于等于阈值箱数)特征列表
    cols_disc_disord_more: list, 无序离散型分箱多(大于阈值箱数)特征列表

    cols_cont: list, 连续型特征

    max_intervals: int, 最大分箱数
    min_pnt: float, 特征单属性样本最小占比

    pipe_options: list, 分别有:
    'Check_None': 检查空值
    'Check_Const_Cols': 剔除常值特征
    'Check_Cols_Types': 获取字段类型

    pinelines: list, 流水线处理列表

    const_cols_ratio: float, 常值字段的阈值%
    const_cols: list

    """
    def __init__(self,
                 df,
                 target,
                 const_cols_ratio,
                 max_intervals=5,
                 min_pnt=0.05):
        self.df = df
        self.df_res = None

        self.target = target

        self.cols_disc = []

        self.cols_disc_ord = []
        self.idx_cols_disc_ord = {}

        self.cols_disc_disord = []
        self.cols_disc_disord_less = []
        self.cols_disc_disord_more = []

        self.cols_cont = []

        self.max_intervals = max_intervals
        self.min_pnt = min_pnt

        self.pipe_options = ['Check_Target', 'Check_None', 'Check_Const_Cols', 'Check_Cols_Types']
        self.pinelines = []

        self.const_cols_ratio = const_cols_ratio
        self.const_cols = []

        # 当前设定第一步必须检查Y标是否唯一的错误
        self.add_pinepine('Check_Target')

        # 当前设定第二步必须检查是否为非空
        self.add_pinepine('Check_None')

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

    def model_pineline_proc(self):
        """
        对设定的流水线过程进行逐步操作


        """
        self.df_res = self.df.copy()

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
