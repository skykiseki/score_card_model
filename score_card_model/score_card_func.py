import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import feat_bincutting_func as fbf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib.ticker import MultipleLocator

class ScoreCardModel(object):
    """
    评分卡模型建模类

    Attributes:
    ----------
    df: dataframe,输入的训练集
    df_res: dataframe, 输出的训练集

    pipe_options: list, 分别有:
    'Check_None': 检查空值
    'Constant_Columns': 剔除常值特征

    pinelines: list, 流水线处理列表

    const_cols_ratio: float, 常值字段的阈值%
    const_cols: list

    """
    def __init__(self,
                 df: pd.DataFrame,
                 const_cols_ratio: float):
        self.df = df
        self.df_res = None


        self.pipe_options = ['Check_None', 'Constant_Columns']
        self.pinelines = []

        self.const_cols_ratio = const_cols_ratio
        self.const_cols = []

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
        print('Checking if exists None value...')
        if self.df.isnull().sum().sum() > 0:
            print('None value exists.Please fix your data.')
            raise TypeError
        else:
            print('No None value exists.')

    def get_const_cols(self, df):
        """
        获得常值特征, 即特征列某个属性占比超阈值%

        Parameters:
        ----------
        df: dataframe,输入的dataframe

        Returns:
        -------
        df_res: dataframe, 剔除常值特征后的dataframe
        """
        for col in self.df.columns:
            if sum(self.df[col].value_counts(normalize=True) >= self.const_cols_ratio) >= 1:
                self.const_cols.append(col)

        df_res = df.drop(self.const_cols, axis=1)

        return df_res


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
        # 当前设定第一步必须检查是否为非空
        self.add_pinepine('Check_None')

        df_res = self.df.copy()

        for proc in self.pinelines:
            proc_name = proc[1]
            if proc_name == 'Check_None':
                self.check_if_has_null()
            elif proc_name == 'Constant_Columns':
                df_res = self.get_const_cols(df_res)
