score_card_model
================

    "风险评分卡模型开发" 
    基于最早的FICO风险评分卡逻辑进行优化

    当前只包含主干部分, 即特征分箱、IV值计算、Woe转化等。
    不包含后续进行建模的部分
    
    特征的分箱使用的方法是卡方分箱, 整个流程为:
    整理特征类型(离散、连续) -> 初始化分箱 -> 卡方合并 -> 单调性检验 -> 特殊值处理 


安装说明
======

```shell
pip install score-card-model
```

使用方法
====


1.初始化:
------

代码示例:

```python
## 加载
import pandas as pd
from score_card_model.ScoreCardModel import ScoreCardModel

# 读取数据
df_data = pd.read_excel("./test.xlsx")

# 创建和初始化类, 前提数据已经预处理完毕, 没有缺失值
scm_obj = ScoreCardModel(df=df_data, target='loan_status')

```

2.设定pipeline参数:
---------------

代码示例:

```python

# sp_vals_cols, 特殊值字典, 格式为{'特征名':[特征值]}
#PS:注意,当前的版本仅支持数值类以-1作为特殊值,且数值型特征必须大于等于0(有时间再修改)

# const_cols_ratio, 判断常值特征的阈值
# max_intervals, 最大分箱数(含特殊值箱, 特殊值单独成箱)
# min_pnt, 分箱的最小样本数占比
# idx_cols_disc_ord, 有序离散特征及其排序idx
pipe_config = {'sp_vals_cols': {'id': [-1], 
                                'dti': [-1],
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
```

3.开始分箱:
-------

代码示例:

```python
scm_obj.model_pineline_proc(pipe_config=pipe_config)

```

4.主要属性:
-------

代码示例:

```python
# 获取流水线处理列表
print(scm_obj.pinelines)

# 获取所有特征的分组取值
dict_cols_to_bins = scm_obj.dict_cols_to_bins

# 获取所有特征的IV
dict_iv = scm_obj.dict_iv

# 获取所有特征的woe值
dict_woe = scm_obj.dict_woe

# 获取woe转化后的dataframe
df_woe = scm_obj.df_woe

# 以下筛选过程顺序可以随意安排, 也非必须调用的内容

## 基于iv进行特征筛选
cols_iv_lower = scm_obj.filter_df_woe_iv(df_woe=df_woe, iv_thres=0.02)
df_woe = df_woe.drop(cols_iv_lower, axis=1)

## 基于相关系数进行特征筛选
cols_corr_higher = scm_obj.filter_df_woe_corr(df_woe=df_woe, corr_thres=0.7)
df_woe = df_woe.drop(cols_corr_higher, axis=1)

## 基于膨胀因子进行特征筛选
cols_vif_higher = scm_obj.filter_df_woe_vif(df_woe=df_woe, vif_thres=10)
df_woe = df_woe.drop(cols_vif_higher, axis=1)

## 基于显著性进行特征筛选
cols_pval_higher = scm_obj.filter_df_woe_pvalue(df_woe=df_woe, pval_thres=0.05)
df_woe = df_woe.drop(cols_pval_higher, axis=1)

```

5.模型评估:
--------
用随机数代表序列:
```python
import  random

n = 50
y_true = [random.randint(0,1) for i in range(n)]
y_pred = [random.randint(0,1) for i in range(n)]
y_proba = [random.random() for i in range(n)]
```

roc曲线:
```python
from score_card_model.utils import model_roc_auc

model_roc_auc(y_true=y_true, y_proba=y_proba, is_plot=True)
```

![roc曲线](https://github.com/skykiseki/score_card_model/blob/main/pics/model_roc_auc.png)


ks曲线:
```python
from score_card_model.utils import model_ks

model_ks(y_true=y_true, y_pred=y_pred, y_proba=y_proba, is_plot=True)
```

![roc曲线](https://github.com/skykiseki/score_card_model/blob/main/pics/model_ks.png)