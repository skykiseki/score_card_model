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

    pinelines_list: 流水线处理列表, 分别有:
    'check_null': 检查空值

    """
    def __init__(self):
        self.df = None
        self.df_res = None
        self.pinelines_list = []

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
        print('Checking if exists None value.')
        if self.df.isnull().sum().sum() == 0:
            return True
        else:
            return False

    def add_pinepine(self, pipe_name):
        """
        向流水线列表中添加流程名称

        Parameters:
        ----------
        pipe_name: 流水线名称,分别有:
        'check_null': 检查空值

        Returns:
        -------
        self
        """
        if pipe_name in ['check_null']:
            self.pinelines_list.append(pipe_name)


    def model_pineline_proc(self):
        """

        :return:
        """
        if self.check_if_has_null():
            print('None value exists.Please fix your data.')
            raise TypeError
