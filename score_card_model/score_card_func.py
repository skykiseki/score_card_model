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
    df:输入的训练集
    df_res:

    pipe_options: 分别有:
    'check_null': 检查空值

    pinelines: 流水线处理列表

    """
    def __init__(self, df):
        self.df = df
        self.df_res = None
        self.pipe_options = ['Check_None']
        self.pinelines = []

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

        for proc in self.pinelines:
            if proc[1] == 'Check_None':
                self.check_if_has_null()
