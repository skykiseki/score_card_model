#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import feat_bincutting_func as fbf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from feat_bincutting_func import *
from matplotlib.ticker import MultipleLocator


# 创建返回汇总表
def dataSetSummary(df_train, targetname='y'):
    # 复制一个df
    df = df_train.copy()
    if targetname in df.columns:
        df = df.drop(targetname, axis=1)
    # 缺失值剔除阈值
    val_invarRtoThres = 0.95
    # 计算特征个数
    num_features = df.shape[1]
    # 特征名
    list_feature = list(df.columns)

    list_featureType = []
    list_featureRowCnt = [df.shape[0]] * num_features
    list_featureNullCnt = []
    list_featureIsInvar = []
    for feature in list_feature:
        list_featureType.append(df[feature].dtype)
        list_featureNullCnt.append(df[feature].isnull().sum())

    for feature in list_feature:
        featureVarRto = ((df[feature].value_counts() / df.shape[0]) > val_invarRtoThres).sum()
        list_featureIsInvar.append(np.where(featureVarRto >= 1, 1, 0))
    dict_summary = {'featureName': list_feature, 'featureType': list_featureType,
                    'featureRowCnt': list_featureRowCnt, 'featureNullCnt': list_featureNullCnt,
                    'featureIsInvar': list_featureIsInvar}
    df_summary = pd.DataFrame(dict_summary)
    df_summary['featureNullCntRto'] = df_summary['featureNullCnt'] / df_summary['featureRowCnt']
    return df_summary


# 剔除90%空缺的特征
def dataSetElimNullFeat(df_train, df_summary, eliminateThresRto=0.9):
    list_featureElim = list(df_summary.loc[df_summary['featureNullCntRto'] > eliminateThresRto, 'featureName'])
    for feature in list_featureElim:
        print('Drop the feature: {0}     (featureNullCntRto: {1:.4f})'.format(
            feature, df_summary.loc[df_summary['featureName'] == feature, 'featureNullCntRto'].values[0]))
    return df_train.drop(list_featureElim, axis=1)


# 剔除95%以上同类的特征
def dataSetElimInvarFeat(df_train, df_summary):
    list_featureElim = list(df_summary.loc[df_summary['featureIsInvar'] == 1, 'featureName'])
    for feature in list_featureElim:
        print("Drop feature: %s" % feature)
    return df_train.drop(list_featureElim, axis=1)


# 返回两个时间戳之间的月份差值
def get_monthsbetween(datetime_s, datetime_e):
    return (datetime_e.year - datetime_s.year) * 12 + (datetime_e.month - datetime_s.month)


# 类别型特征数值分布、类别型特征与y值关联分布
def plot_discrete(df_train, cols_discrete, col_target, subfigsize=(15, 25)):
    # 特征个数 
    feat_cnt = len(cols_discrete)
    # 创建一个cnt * 2 的subplots
    fig, axes = plt.subplots(feat_cnt, 2, figsize=subfigsize)
    for i in range(feat_cnt):
        # 先画左侧的特征分布图
        vals_cnt = df_train[cols_discrete[i]].value_counts(dropna=False)
        list_vals = list(vals_cnt)
        cnt_row = df_train.shape[0]
        vals_cnt.plot(kind='barh', ax=axes[i][0], rot=0, title=cols_discrete[i])
        # 贴标签
        for j in range(len(list_vals)):
            axes[i][0].text(list_vals[j] / 2, j, r'{0}-{1:.1f}%'.format(list_vals[j], list_vals[j] / cnt_row * 100),
                            ha='center', va='center')
        # 画右侧的关联分布图
        ct_featY = pd.crosstab(df_train[cols_discrete[i]], df_train[col_target], normalize='index')
        ct_featY.plot(kind='barh', stacked=True, rot=0, title=cols_discrete[i] + ' VS y', ax=axes[i][1])
        for j in range(len(ct_featY.index)):
            axes[i][1].text(ct_featY.iloc[j, 0] / 2, j, '%.1f' % (ct_featY.iloc[j, 0] * 100), ha='center', va='center')
            axes[i][1].text(ct_featY.iloc[j, 0] + ct_featY.iloc[j, 1] / 2, j,
                            '{0:.1f}%'.format(ct_featY.iloc[j, 1] * 100),
                            ha='center', va='center')
    # 平铺
    fig.tight_layout()


# 连续型特征绘图,
# 特征箱线图分布, 特征与y值的箱线图, 特征的频率直方图，特征与y值的频率直方图
# df_train: 训练集, cols_continuous: 连续型特征名称列表, col_target: 目标变量y值
def plot_continuous(df_train, cols_continuous, col_target):
    # 特征个数
    cnt_features = len(cols_continuous)
    # 创建画布
    fig, axes = plt.subplots(cnt_features, 4, figsize=(25, 120))
    for i in range(cnt_features):
        fontdict = {'fontsize': 'large'}
        # 第一列，画特征的箱线图
        sns.boxplot(df_train[cols_continuous[i]], ax=axes[i][0], orient='v')
        axes[i][0].set_ylabel('')
        axes[i][0].set_xlabel(cols_continuous[i], fontdict=fontdict)
        # 第二列，画特征与y值的箱线图
        sns.boxplot(x=df_train[col_target], y=df_train[cols_continuous[i]], ax=axes[i][1], orient='v')
        axes[i][1].set_ylabel('')
        axes[i][1].set_xlabel(cols_continuous[i] + ' group by y', fontdict=fontdict)
        # 第三列，画特征的频率分布直方图
        sns.distplot(df_train[cols_continuous[i]].dropna(), ax=axes[i][2])
        axes[i][2].set_xlabel(cols_continuous[i], fontdict=fontdict)
        # 第四列，画特征与y值的频率直方图
        sns.distplot(df_train.loc[df_train[col_target] == 1, cols_continuous[i]].dropna(),
                     ax=axes[i][3], color='r', label='1')
        sns.distplot(df_train.loc[df_train[col_target] == 0, cols_continuous[i]].dropna(),
                     ax=axes[i][3], color='b', label='0')
        axes[i][3].legend(loc='best')
        axes[i][3].set_xlabel(cols_continuous[i], fontdict=fontdict)
    #  平铺
    fig.tight_layout()


# 经验而言，iv<=0.02:认为其没有预测性;0.02<iv<=0.1,弱预测性;0.1<iv<=0.2,有一定预测性;iv>0.2,强预测性
# 注意, 如果一个特征的IV特别高, 这个时候必须要注意了
# 输入dict_featIv: dict, 特征的IV {featname1: iv_val1, featname2: iv_val2}
# 输入iv_thres: 剔除特征所基于的最小IV阈值
# 输入dataSetSummary的返回值, 新增一个IV字段
# 输入if_plotted是否打印
# 返回需要剔除的特征名list和绘图
def feats_iv_proc(dict_feativ, iv_thres=0.1, is_plot=True):
    list_featDropped = []
    # 找出iv 小于阈值的特征
    for k, v in dict_feativ.items():
        if v < iv_thres:
            list_featDropped.append(k)
    # 绘图
    if is_plot:
        fig, ax = plt.subplots(figsize=(10, 10))
        fontdict = {'fontsize': 15}
        dict_feativ_order = {k: v for k, v in sorted(dict_feativ.items(), key=lambda x: x[1])}
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
    return list_featDropped


# 基于两个特征的斯皮尔逊相关系数进行剔除
# 若两个特征之间的相关系数大于阈值(默认为0.7), 则剔除IV较低的那个
# 输入X: 训练集
# 输入dict_feativ: 特征对应的IV
# 输入targetname: 目标变量名
# 输入corr_thres: 相关系数的阈值
# 输入is_plot: 是否绘图(热力图)
# 返回需要剔除的特征list
def feats_corr_proc(X, dict_feativ, targetname='y', corr_thres=0.7, is_plot=True):
    dict_feat_corr = {}
    set_featDropped = set()
    # copy
    df = X.drop(targetname, axis=1)
    # 计算相关系数, 记得剔除feat本身组合
    # 注意这里含(A, B) 和(B, A）的重复组合
    corr_feat = df.corr()
    for col in corr_feat.columns:
        for idx in corr_feat.index:
            if col != idx:
                dict_feat_corr[(idx, col)] = corr_feat.loc[idx, col]
    # return corr_feat
    # 找出大于等于相关系数阈值的组合
    # 通过比较两者的IV插入list
    for key, val in dict_feat_corr.items():
        if abs(val) >= corr_thres:
            iv_feat_0 = dict_feativ[key[0]]
            iv_feat_1 = dict_feativ[key[1]]
            # 插入IV较小的值
            if iv_feat_0 <= iv_feat_1:
                set_featDropped.add(key[0])
            else:
                set_featDropped.add(key[1])
    # 绘制
    if is_plot:
        fig, ax = plt.subplots(figsize=(15, 15))
        sns.heatmap(corr_feat, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax, linewidths=0.5, square=True)
    return list(set_featDropped)


# 基于statsmodels.stats.outliers_influence.variance_inflation_factor进行vif分析
# 对于共线性问题,可以使用逐一剔除的方法,即先遍历全部特征,尝试去剔除一个特征, 再去统计剔除后的vif是否小于阈值
# 如果小于阈值, 则说明剔除该特征有效
# 如果存在有多个特征可以进行剔除, 则剔除IV较小的那个特征
# 如果遍历了之后,无法找到一个适合的特征进行剔除,则将最小IV的特征进行剔除, 保留高IV特征
# 输入X_woe: 经过woe转化后的df
# 输入dict_feativ: 特征的iv字典
# 输入vif_thres: vif阈值, 大于这个阈值表示有严重的共线性问题
# 输入targetname: 目标特征名称
# 输出list待剔除的特征
def feats_vif_proc(X_woe, dict_feativ, vif_thres=10, targetname='y'):
    # list记录最后要剔除的特征
    list_featDropped = []
    # list记录vif大于阈值的特征名称
    list_feats_h_vif = []
    # copy
    df = X_woe.drop(targetname, axis=1)
    # 特征名
    list_featnames = list(df.columns)
    # df转化为矩阵
    mat_vif = df.as_matrix()
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
        dict_feativ_order = {k: v for k, v in sorted(dict_feativ.items(), key=lambda x: x[1]) if k in list_featnames}
        # 开始遍历每个特征
        for feat in dict_feativ_order.keys():
            # 剔除这个特征
            df0 = df.drop(feat, axis=1)
            # 重新计算vif
            mat_df0 = df0.as_matrix()
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

        # 插入返回的list中
        list_featDropped.append(feat_candi_miniv)
        # 剔除该特征
        df = df.drop(feat_candi_miniv, axis=1)
        mat_vif = df.as_matrix()

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
    return list_featDropped


# 对回归模型系数进行显著性检验
# 类似vif的处理方法,先按p_value最高的特征进行剔除,再进行回归,直到所有的系数显著
# 输入X_woe, 经过woe转化的训练集
# 输入pval_thres, 显著性因子
# 输入targetname, 目标变量名
# 输出待剔除的特征
def feats_pvalue_proc(X_woe, pval_thres=0.05, targetname='y'):
    # list记录最后要剔除的特征
    list_featDropped = []
    # copy
    df = X_woe.copy()
    # 初始建模, 注意加入常数项
    df['intercept'] = [1] * df.shape[0]
    x = df.drop(targetname, axis=1)
    y = df[targetname]
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
        list_featDropped.append(feat_max_pval)
        # 剔除该特征
        df0 = df.drop(feat_max_pval, axis=1)
        # 重新建模
        df0['intercept'] = [1] * df0.shape[0]
        x = df0.drop(targetname, axis=1)
        y = df0[targetname]
        model = sm.Logit(y, x)
        results = model.fit()
        # 重置赋值dict_feats_pvalue
        dict_feats_pvalue = results.pvalues.to_dict()
        # 删除截距项
        del dict_feats_pvalue['intercept']
        # df剔除该特征
        df = df.drop(feat_max_pval, axis=1)
    return list_featDropped


# 基于statsmodel模块进行预logit建模, 输出各系数或者模型(如F检验)显著性检验结果
# ****注意, 该函数因为涉及到fit建模, 所以可能速度会比较慢, 建议测试的时候才用****
# 输入X_woe, 经过WOE转化后的X
# 输入targername, 目标变量名
def logit_summary(X_woe, targetname='y'):
    # copy
    df_woe = X_woe.copy()
    # 注意创建常数项
    df_woe['intercept'] = [1] * df_woe.shape[0]
    x = df_woe.drop(targetname, axis=1)
    y = df_woe[targetname]
    model = sm.Logit(y, x)
    results = model.fit()
    return print(results.summary())


# 基于样本的proba计算得分
# B = PDO / ln(2)
# score = base_score - B * ln(odds)
# score = base_score - PDO / ln(2) * ln(odds)
# 输入proba, 模型最终产出的概率
# 输出score, 对应的评分
def proba_to_score(proba, base_score=500, pdo=20):
    odds = proba / (1 - proba)
    score = int(base_score - pdo / np.log(2) * np.log(odds))
    return score


# 绘制score的频数分布, score的01频数分布, badrate的曲线
# 输入X_woe, 经过WOE转化后的X
# 输入targername, 目标变量名
def plot_score_badrate(X_woe, targetname='y'):
    # copy
    df = X_woe.copy()
    # 绘制一个3X1的画布
    fig, ax = plt.subplots(3, figsize=(15, 20))

    score_min = df['score'].min() // 100 * 100
    score_max = df['score'].max() // 100 * 100
    x_range = range(score_min, score_max, 20)
    x_linrange = np.linspace(df['score'].min(), df['score'].max(), 20)

    fontsize = 15
    linewidth = 3
    markersize = 12
    fontdict = {'fontsize': fontsize}

    # 第一个绘制评分频数分布
    sns.distplot(df['score'], ax=ax[0])
    ax[0].set_title('Distribution of scores', fontsize=fontsize)
    ax[0].set_xticks(x_range)
    ax[0].set_xlabel('score', fontdict=fontdict)

    # 第二个绘制0,1两类的频数分布
    df['lin_bins'] = pd.cut(df['score'], x_linrange)
    sns.distplot(df.loc[df[targetname] == 0, 'score'], ax=ax[1], hist=False, label='0', kde_kws={'lw': linewidth})
    sns.distplot(df.loc[df[targetname] == 1, 'score'], ax=ax[1], hist=False, label='1', kde_kws={'lw': linewidth})
    ax[1].set_title('Distibution of scores(0,1)', fontsize=fontsize)
    ax[1].set_xticks(x_range)
    ax[1].set_xlabel('score', fontdict=fontdict)
    ax[1].legend(fontsize='large')

    # 第三个绘制badrate的频数分布
    regroup_badrate = fbf.bin_badrate(df, featname='lin_bins', targetname=targetname)
    twinx_ax2 = ax[2].twinx()
    sns.distplot(df['score'], bins=x_linrange, kde=False, ax=ax[2], hist_kws={'rwidth': 0.9})
    twinx_ax2.plot(x_linrange[1:], regroup_badrate['bad_rate'].to_list(), linestyle='--', linewidth=linewidth,
                   marker='x', markersize=markersize, markeredgewidth=5, c='r', label='badrate')
    ax[2].set_title('Distibution of scores & badrates', fontsize=fontsize)
    ax[2].set_xlabel('score', fontdict=fontdict)
    twinx_ax2.set_ylabel('badrate', fontdict=fontdict)
    # 铺满整个画布
    fig.tight_layout()


# 绘图特征分组的badrate分布
# 输入X_train, Woe转换前的训练集
# 输入list_feats_inmod, 入模特征list
# 输入cols_discrete, 离散型特征list
# 输入dict_feattobins, 特征的取值dict
# 输入targetname, 目标类别名
def plot_feat_badrate(X_train, list_feats_inmod, cols_discrete, dict_feattobins,
                      targetname='y', subfigsize=(15, 80)):
    # 入模特征数
    cnt_feats_inmod = len(list_feats_inmod)
    # 样本数
    cnt_total = X_train.shape[0]
    # 创建1 X N画布
    fig, ax = plt.subplots(cnt_feats_inmod, 1, figsize=subfigsize)

    # 开始处理
    for i in range(cnt_feats_inmod):
        # 取得特征名
        feat = list_feats_inmod[i]
        # copy
        df = X_train.loc[:, [feat, targetname]]
        # 如果是离散型特征, 则直接映射, 如果是连续型特征, 则先划分bins再映射
        if feat in cols_discrete:
            df[feat + '_groupno'] = df[feat].map(dict_feattobins[feat])
        else:
            df[feat + '_interval'] = df[feat].apply(lambda x: fbf.value_to_intervals(x, dict_feattobins[feat]))
            df[feat + '_groupno'] = df[feat + '_interval'].map(dict_feattobins[feat])
        # 创建regroup_badrate
        regroup_badrate = fbf.bin_badrate(df, featname=feat + '_groupno', targetname=targetname)
        # 创建样本占比
        regroup_badrate['pnt_feat_vals'] = regroup_badrate['num_feat_vals'] / cnt_total
        # 注意regroup_badrate需要排序
        regroup_badrate = regroup_badrate.sort_index()
        # 绘图
        fontsize = 15
        markersize = 15
        markeredgewidth = 6
        fontdict = {'fontsize': fontsize}
        ax[i].bar(regroup_badrate.index, regroup_badrate['pnt_feat_vals'])
        ax[i].set_ylim((0, 1))
        ax[i].set_title("%s' Badrate" % feat, fontdict=fontdict)
        ax[i].set_xticks(regroup_badrate.index)
        for x, y, z in zip(list(regroup_badrate.index),
                           list(regroup_badrate['pnt_feat_vals'] / 2),
                           list(regroup_badrate['pnt_feat_vals'])):
            ax[i].text(x, y, '{0:.2f}%'.format(z * 100),
                       ha='center', va='center', fontdict=fontdict, color='white')
        if feat not in cols_discrete:
            list_xlabel = [k for k, v in sorted(dict_feattobins[feat].items(), key=lambda x: x[1])]
            ax[i].set_xticklabels(list_xlabel, fontdict=fontdict)
        ax[i].set_xlabel('Group', fontdict=fontdict)
        ax[i].set_ylabel('Pct. of feat value', fontdict=fontdict)

        ax_twin = ax[i].twinx()
        ax_twin.plot(regroup_badrate.index, regroup_badrate['bad_rate'],
                     linewidth=3, linestyle='--', color='r',
                     marker='x', markersize=markersize, markeredgewidth=markeredgewidth,
                     label='Badrate')
        for x, y, z in zip(list(regroup_badrate.index),
                           list(regroup_badrate['bad_rate']),
                           list(regroup_badrate['bad_rate'])):
            ax_twin.text(x, y + 0.02, '{0:.2f}%'.format(z * 100),
                         ha='center', va='center', fontdict=fontdict, color='r')
        ax_twin.set_ylim((0, regroup_badrate['bad_rate'].max() + 0.1))
        ax_twin.set_ylabel('Badrate', fontdict=fontdict)
        ax_twin.legend(loc='best', fontsize='xx-large')
    fig.tight_layout()


# 测试机的woe转化、预测、评分
# 输入X_test, 原始的训练集样本, 注意这里的样本是含y值的
# 输入estimator, 训练所得的模型
# 输入list_feats_inmod, 入模的特征list
# 输入dict_feattobins, 取值分组字典
# 输入dict_featwoe, 特征取值分组对应的woe转化
# 输入targetname, 类别名称
# 输入cols_discrete, 离散特征名称list
# 这一步可以不要, 实际在对于T1数据完全可以在数仓做
def test_trans_pred(X_test, estimator, list_feats_inmod, dict_feattobins, dict_featwoe, cols_discrete, targetname='y'):
    # copy
    df = X_test.copy()
    # dict, 保存着新出现的特征取值
    dict_inv_featvals = {}

    # 剔除入模变量以外的特征
    df = df.loc[:, list_feats_inmod + [targetname]]
    # 对入模特征进行woe转换
    for feat in list_feats_inmod:
        # 区分离散型和连续型来处理
        # 离散型的需要考虑测试集中可能存在有新的特征取值, 这个时候就以最大的badrate组进行代替(未知的才是最可怕的)
        # 以最大的badrate进行代替, 意味着woe也为最大值(带符号)
        if feat in cols_discrete:
            # 取得测试集的特征取值
            list_feat_vals = list(set(df[feat]))
            list_feat_vals_inval = [i for i in list_feat_vals if i not in dict_feattobins[feat].keys()]
            if len(list_feat_vals_inval) > 0:
                dict_inv_featvals[feat] = list_feat_vals_inval
            # 取得最大的woe值
            max_feat_woe = max(dict_featwoe[feat].values())
            # 开始处理
            df[feat + '_woe'] = df[feat].map(dict_feattobins[feat]).map(dict_featwoe[feat])
            df[feat + '_woe'] = df[feat + '_woe'].fillna(max_feat_woe)
            df = df.drop(feat, axis=1)
        else:
            # 先对连续值进行区间转化再做映射
            df[feat + '_interv'] = df[feat].apply(lambda x: fbf.value_to_intervals(x, dict_feattobins[feat]))
            df[feat + '_woe'] = df[feat + '_interv'].map(dict_feattobins[feat]).map(dict_featwoe[feat])
            df = df.drop([feat + '_interv', feat], axis=1)
    # copy woe
    df_woe = df.drop(targetname, axis=1)
    # 预测
    df[targetname + '_pred'] = estimator.predict(df_woe)
    df[targetname + '_proba'] = [i[1] for i in estimator.predict_proba(df_woe)]
    df['score'] = df[targetname + '_proba'].apply(lambda x: proba_to_score(x))
    return df, dict_inv_featvals


# 整理特征、特征分组、特征分组对应的woe、特征分组对应的分数
# 输入X_woe, woe转化后的训练集(不含y值)
# 输入list_feats_inmod, list, 入模变量
# 输入dict_feattoobins, 特征的分组取值字典
# 输入dict_featwoe, 特征的分组woe
# 输入if_saved, 是否保存为文件
# 输入file_saved, 保存的文件名
# 输出df保存特征各自的属性&分值
# 保存成excel文件
def model_feats_score(X_woe,
                      estimator,
                      dict_feattoobins, dict_featwoe,
                      base_score=500, pdo=20,
                      if_saved=True, file_saved='feats_group_dict.xlsx'):
    # copy
    df_woe = X_woe.copy()
    # 取出训练特征名列表
    list_featnames = []
    for feat in df_woe.columns:
        list_featnames.append(feat[:len(feat) - 4])
    # 取出模型的回归系数
    list_coefs = list(estimator.coef_[0])
    # 做一个对应的dict
    dict_feats_coef = {list_featnames[i]: list_coefs[i] for i in range(len(list_featnames))}
    # 取出截距
    model_intercept = estimator.intercept_[0]
    # 创建一个空的df
    df = pd.DataFrame(columns=['featname', 'feat_value', 'group_no', 'group_woe', 'group_score'])
    # 开始插入
    for k1, v1 in dict_feattoobins.items():
        # 仅考虑入模特征
        if k1 in list_featnames:
            for k2, v2 in v1.items():
                # 取得特征名称
                featname = k1
                # 取得特征取值
                feat_value = k2
                # 取得特征分组no
                group_no = v2
                # 取得特征分组对应的woe
                group_woe = dict_featwoe[featname][group_no]
                # 计算得分
                # 先取得权值
                feat_coef = dict_feats_coef[featname]
                group_score = round(-pdo / np.log(2) * feat_coef * group_woe)
                # 准备行数据
                row_data = {'featname': featname,
                            'feat_value': feat_value,
                            'group_no': group_no,
                            'group_woe': group_woe,
                            'group_score': group_score}
                # 插入
                df = df.append(row_data, ignore_index=True)
        else:
            continue
    # 插入常数项得分
    const_score = int(base_score - pdo * model_intercept / np.log(2))
    df['const_score'] = [const_score] * df.shape[0]
    # 如果保存,则在本目录生成一个xlsx
    if if_saved:
        df.to_excel(excel_writer=file_saved, sheet_name='feats_group', index=False)
    return df


# 绘制roc曲线
# 输入X_train_woe, 含预测结果和预测概率的训练集
# 输入is_plot, 是否绘图
def model_roc_auc(X_train_woe, targetname='y', is_plot=True):
    # 计算auc
    auc = roc_auc_score(y_true=X_train_woe[targetname], y_score=X_train_woe[targetname + '_proba'])
    if is_plot:
        fontsize = 15
        fontdict = {'fontsize': fontsize}
        # 绘制roc曲线
        fpr, tpr, thresholds = roc_curve(y_true=X_train_woe[targetname],
                                         y_score=X_train_woe[targetname + '_proba'],
                                         pos_label=1)
        fig, ax = plt.subplots(figsize=(15, 8))

        # 绘制roc
        ax.plot(fpr, tpr, label='ROC Curve(AUC=%.2F)' % auc, linewidth=5)
        # 绘制(0,0) (1,1)直线
        ax.plot([0, 1], [0, 1], linestyle='--', c='r', linewidth=2)

        ax.set_title('Receiver Operating Characteristic', fontdict=fontdict)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel('False Positive Rate', fontdict=fontdict)
        ax.set_ylabel('True Positive Rate', fontdict=fontdict)
        ax.legend(loc='lower right', fontsize='x-large')
    # 返回结果
    return auc


# 计算模型的ks(Kolmogorov-Smirnov)
# ks: <0 模型错误, 0~0.2 模型较差, 0.2~0.3 模型可用, >0.3 模型预测性较好
# ks计算公式为 ks = max(cum(Bi)/ Bad(total) - cum(Gi) / Good(total))
# 输入X 输入的含y, y_pred, y_proba的集合
# 输入targetname, 目标变量名
# 输入is_plot, 是否绘制ks曲线
def distin_ks(X, targetname='y', is_plot=True):
    # 取得原始y值和预测的概率值
    df = X.loc[:, [targetname, targetname + '_proba']]
    # 计算总样本数、bad样本数、good样本数
    cnt_total = df.shape[0]
    cnt_bad = df[df[targetname] == 1].shape[0]
    cnt_good = cnt_total - cnt_bad
    # 对df基于proba进行降序排序
    df = df.sort_values(by=targetname + '_proba', ascending=False)
    # 计算总样本(排序后的)下的样本数占比
    df['cum_pnt_total'] = [i / cnt_total for i in list(range(1, cnt_total + 1))]
    # 计算好样本(排序后的)下的样本数占比
    df['cum_pnt_good'] = (df[targetname] == 0).cumsum() / cnt_good
    # 计算坏样本(排序后的)下的样本数占比
    df['cum_pnt_bad'] = (df[targetname] == 1).cumsum() / cnt_bad
    # 计算差值的绝对值
    df['cum_diff_value'] = abs(df['cum_pnt_bad'] - df['cum_pnt_good'])
    # 计算ks值
    ks = df['cum_diff_value'].max()
    # 取得ks下的index, 计算在该idx下的cum_pnt_total
    idxmax = df['cum_diff_value'].idxmax()
    x_ks = df.loc[idxmax, 'cum_pnt_total']
    # 绘制ks曲线
    if is_plot:
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.plot(df['cum_pnt_total'], df['cum_pnt_bad'], linewidth=2, label='Bad')
        ax.plot(df['cum_pnt_total'], df['cum_pnt_good'], linewidth=2, label='Good')
        ax.plot(df['cum_pnt_total'], df['cum_diff_value'], linewidth=2, label='K-S Curve')
        ax.plot([x_ks], [ks], 'o')
        ax.annotate('K-S Statistic(%.2f)' % ks, xy=(x_ks, ks), xytext=(x_ks, ks / 1.5),
                    fontsize=20, arrowprops=dict(arrowstyle='fancy', facecolor='red'))
        ax.grid(linewidth=0.5, linestyle='--')
        ax.legend(loc='left upper', fontsize='large')
        ax.set_title('K-S Curve', fontsize=15)
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
    return ks


# 基于ROC计算KS
# 事实上, ROC曲线第一步即是基于score进行降序排序
# TPR即是阈值前的bad的个数, 同等与cum(Bi), FPR即是阈值前的good的个数, 同等与cum(Gi)
# TP+FN即为实际的badtotal, FP+TN 即为实际的goodtotal
# 输入X 输入的含y, y_pred, y_proba的集合
# 输入targetname, 目标变量名
# 输入is_plot, 是否绘制ks曲线
def distin_ks_roc(X, targetname='y', is_plot=True):
    # 取得原始y值和预测的概率值
    df = X.loc[:, [targetname, targetname + '_proba']]
    # FPR, TPR, ks
    fpr, tpr, thresh = roc_curve(df[targetname], df[targetname + '_proba'])
    ks = max(abs(fpr - tpr))
    x_ks = np.argwhere(abs(fpr - tpr) == ks)[0][0]

    if is_plot:
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.plot(fpr, linewidth=2, label='Bad')
        ax.plot(tpr, linewidth=2, label='Good')
        ax.plot(abs(fpr - tpr), linewidth=2, label='K-S Curve')
        ax.plot([x_ks], [ks], 'o')
        ax.annotate('K-S Statistic(%.2f)' % ks, xy=(x_ks, ks), xytext=(x_ks, ks / 1.5),
                    fontsize=20, arrowprops=dict(arrowstyle='fancy', facecolor='red'))
        ax.grid(linewidth=0.5, linestyle='--')
        ax.legend(loc='left upper', fontsize='large')
        ax.set_title('K-S Curve', fontsize=15)
        ax.set_ylim((0, 1))
        ax.set_xlim((0, len(fpr)))
    return ks


# divergence是模型区分度,综合考虑两个集合的中心距离与半径, 越大区分度越高
# divergence=(u_bad - u_good)**2 / (0.5*(var_bad + var_good))
# 输入X 输入的含y, y_pred, y_proba的集合
# 输入targetname, 目标变量名
def distin_divergence(X, targetname='y'):
    # 取得原始y值和预测的概率值
    df = X.loc[:, [targetname, targetname + '_proba']]
    # 计算均值
    mean_good = df.loc[df[targetname] == 0, targetname + '_proba'].mean()
    mean_bad = df.loc[df[targetname] == 1, targetname + '_proba'].mean()
    # 计算方差
    var_good = df.loc[df[targetname] == 0, targetname + '_proba'].var()
    var_bad = df.loc[df[targetname] == 1, targetname + '_proba'].var()
    # 计算区分度
    divergence = (mean_bad - mean_good) ** 2 / (0.5 * (var_bad + var_good))
    return divergence


# 计算Gini系数
# 先基于score进行升序排序
# Gini系数图中, 横坐标是累计的总数或者总数占比
# 纵坐标是累计的正例数或者正例数占比(洛伦兹曲线)
# 45%直线代表无差异, 洛伦兹曲线越靠近该直线, 表示区分度越低
# gini不纯度，越小越纯，区分度越好
# 在好坏样本占比差异较大时, 坏样本很少, 此时洛伦兹曲线和ROC曲线横纵轴取值基本一致, 曲线基本一致,
# 此时可以使用近似计算gini=2*AUC - 1
# 输入X 输入的含y, y_pred, y_proba的集合
# 输入targetname, 目标变量名
# 输入is_plot, 是否绘制Gini曲线
def distin_gini(X, targetname='y', is_plot=True):
    # 取得原始y值和预测的概率值
    df = X.loc[:, [targetname, targetname + '_proba']]
    # 样本个数, 坏样本个数
    cnt_total = df.shape[0]
    cnt_bad = df[df[targetname] == 1].shape[0]
    # 基于score进行升序
    df = df.sort_values(by=targetname + '_proba', ascending=True)
    # 计算总样本(排序后的)下的样本数占比
    df['cum_pnt_total'] = [i / cnt_total for i in list(range(1, cnt_total + 1))]
    # 计算坏样本(排序后的)下的样本数占比
    df['cum_pnt_bad'] = (df[targetname] == 1).cumsum() / cnt_bad
    # 计算45°线之间的差值
    df['cum_pnt_diff'] = df['cum_pnt_total'] - df['cum_pnt_bad']
    # 计算A, B, G
    # 用梯形拟合计算B的面积
    area_b = np.trapz(df['cum_pnt_bad'], df['cum_pnt_total'])
    # 计算面积A
    area_a = 0.5 - area_b
    # 计算GINI
    gini = area_a / (area_a + area_b)

    if is_plot:
        fontsize = 15
        fontdict = {'fontsize': fontsize}
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.stackplot(df['cum_pnt_total'], df['cum_pnt_bad'], df['cum_pnt_diff'],
                     colors=['#7DB2FA', '#AFC4D6', 'black'])
        ax.plot(df['cum_pnt_total'], df['cum_pnt_bad'], color='black', linewidth=2,
                label='Lorenz Curve(Gini=%.2f)' % gini)
        ax.plot(df['cum_pnt_total'], df['cum_pnt_total'], color='black', linewidth=2)
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        ax.set_title('Lorenz Curve and Gini coefficient', fontsize=fontsize)
        ax.set_xlabel('Cumulative Pct. of Total', fontdict=fontdict)
        ax.set_ylabel('Cumulative Pct. of Bad', fontdict=fontdict)
        ax.text(0.4, 0.45, 'Line of Equality (45 Degree)', rotation=45, fontsize=fontsize, fontweight='bold')
        ax.text(0.4, 0.2, 'Lorenz Curve', rotation=45, fontsize=fontsize, fontweight='bold')
        ax.text(0.8, 0.7, 'A', fontsize=25, fontweight='bold')
        ax.text(0.9, 0.1, 'B', fontsize=25, fontweight='bold')
        ax.legend(fontsize='xx-large', loc='left upper')
    return gini


# 计算模型稳定性PSI
# 基于分数进行n_split等分, 再计算PSI
# PSI的公式为PSI=sum((actual_pnt - expected_pnt) *  ln(actual_pnt / expected_pnt))
# psi的经验值,psi < 0.1:稳定性高，无需更新模型;
# 0.1 <= psi <0.25:稳定性一般,需要进一步研究;psi >= 0.25:稳定性差,需要更新模型
# 输入X_train, 训练集, 含y, y_pred, y_proba, score
# 输入X_test, 测试集, 含y, y_pred, y_proba, score
# 输入n_split, 分组数
def model_psi(X_train, X_test, n_split=10):
    # score分组dict
    dict_score_interval = {}
    # copy
    df_train = X_train.loc[:, ['score']]
    df_test = X_test.loc[:, ['score']]
    # 计算各数据集的样本数
    cnt_train = df_train.shape[0]
    cnt_test = df_test.shape[0]
    # 从验证集开始
    list_scores = list(set(df_train['score']))
    # 分数最大值、最小值
    max_score = max(list_scores)
    min_score = min(list_scores)

    # 采集的分数分段
    interval_scores = (max_score - min_score) / n_split
    # 采集刻度值
    list_score_split = sorted([int(min_score + i * interval_scores) for i in range(1, n_split)])
    # 形成分组dict, 注意需要手工添加最后一组
    for i in range(len(list_score_split)):
        if i == 0:
            label_l = '[0,'
            label_r = str(list_score_split[0]) + ')'
            label = label_l + label_r
            dict_score_interval[label] = 0
        else:
            label_l = '[' + str(list_score_split[i - 1]) + ','
            label_r = str(list_score_split[i]) + ')'
            label = label_l + label_r
            dict_score_interval[label] = i
    dict_score_interval['[%s, +Inf)' % max(list_score_split)] = n_split - 1
    # 做个升序sorted
    dict_score_interval = {k: v for k, v in sorted(dict_score_interval.items(), key=lambda x: x[1])}
    # 映射训练集和测试机的分数区间
    df_train['score_interval'] = df_train['score'].apply(lambda x: fbf.value_to_intervals(x, dict_score_interval))
    df_test['score_interval'] = df_test['score'].apply(lambda x: fbf.value_to_intervals(x, dict_score_interval))
    # 基于分数区间映射groupno
    df_train['score_groupno'] = df_train['score_interval'].map(dict_score_interval)
    df_test['score_groupno'] = df_test['score_interval'].map(dict_score_interval)

    # groupno
    list_groupno = list(dict_score_interval.values())
    # score_interval
    list_interval = list(dict_score_interval.keys())
    # group_pnt
    list_actual_cnt = list(df_train['score_groupno'].value_counts().sort_index() / cnt_train)
    # 组成psi统计用的dataframe
    df_psi = pd.DataFrame({'score_groupno': list_groupno,
                           'score_interval': list_interval,
                           'actual_pnt': list_actual_cnt})
    # 计算测试集, 注意这里可能存在训练集有而测试集没有的组, 所以这里要单独处理
    dict_expected_cnt = (df_test['score_groupno'].value_counts() / cnt_test).to_dict()
    df_psi['expected_pnt'] = df_psi['score_groupno'].map(dict_expected_cnt)
    # 对没匹配的分组要补充0.000001(不能为0,因为后续计算需要用到这个为分母), 不然后面计算会错误
    df_psi['expected_pnt'] = df_psi['expected_pnt'].fillna(1 / cnt_test)
    # 计算其他指标
    df_psi['diff_ac_exp_pnt'] = df_psi['actual_pnt'] - df_psi['expected_pnt']
    df_psi['div_ac_exp_pnt'] = df_psi['actual_pnt'] / df_psi['expected_pnt']
    df_psi['ln_ac_exp_pnt'] = np.log(df_psi['div_ac_exp_pnt'])
    df_psi['index'] = df_psi['diff_ac_exp_pnt'] * df_psi['ln_ac_exp_pnt']
    # 计算psi
    psi = df_psi['index'].sum()
    return psi, df_psi


# 计算LIFT提升度, 比较用模型与不用模型的提升
# 计算模型在该P阈值下的精度Precision = TP/(TP+FP)
# 由模型定义可知, 大于p值的会被分类正类(1),小于的则会成为负类(0)
# 所以经过proba排序后, 逻辑类似TPR,FPR
# 此时模型预测的坏样本数其实就是样本数
# 解读lift:
# 在样本内,样本数为M,含有N个正类需要抓取, 此时抓取概率为N/M,随机抓取
# 若以p_thres作为阈值,在大于阈值的样本(sub_sample)同样是抓取N个正类,此时由于范围小了,自然抓取概率变大,变大为N/(FP+TP)
# 则lift为此时精度与实际比率的提升率
# 从公式来看,可以理解为在M中抓,和在(FP+TP)内抓取N个正类, LIFT其实也就是M/(FP+TP), 即抓取面积的比率
# depth可以理解为样本的百分比或者某个分数、某个概率阈值
# 如果阈值设的很大,即depth很小,这个时候lift很大, 为了展现, 一般以10% * N 作为阈值点
# 输入X 输入的含y, y_pred, y_proba的集合
# 输入targetname, 目标变量名
# 输入n_split, 分组数
def plot_model_lift(X, targetname='y', n_split=10):

    # copy
    df = X.loc[:, [targetname, targetname + '_proba', 'score']]
    # 基于proba进行降序排序
    df = df.sort_values(by=targetname + '_proba', ascending=False).reset_index(drop=True)
    # 总样本数
    cnt_total = df.shape[0]
    # 每个分组样本数
    cnt_perbin = np.ceil(cnt_total / n_split)
    # 先计算得到阈值的index
    list_index = []
    for i in range(1, n_split + 1):
        if i == n_split:
            list_index.append(cnt_total - 1)
        else:
            list_index.append(int(i * cnt_perbin))
    # 以防万一做个排序
    list_index = sorted(list_index)

    # 计算TP,FP,DEPTH,PV_PLUS, BADRATE_ACTUAL,FLIT
    df['true_positive'] = (df['y'] == 1).cumsum()
    df['false_positive'] = (df['y'] == 0).cumsum()
    df['pv_plus'] = df['true_positive'] / (df['true_positive'] + df['false_positive'])
    df['depth'] = [i / cnt_total for i in range(1, cnt_total + 1)]
    df['badrate_actual'] = (df['y'] == 1).cumsum() / cnt_total
    df['lift_cum'] = df['pv_plus'] / df['badrate_actual']

    # 产生score分组区间
    list_scores = df.loc[list_index, 'score'].to_list()
    dict_score_interv = {}
    for i in range(len(list_index)):
        if i == 0:
            label = '[0,%s]' % str(list_scores[i])
            dict_score_interv[label] = i
        else:
            label = '(%s,%s]' % (str(list_scores[i - 1]), str(list_scores[i]))
            dict_score_interv[label] = i
    df['score_bin'] = df['score'].apply(lambda x: fbf.value_to_intervals(x, dict_score_interv))
    df['score_bin_no'] = df['score_bin'].map(dict_score_interv)
    # 汇总
    regroup = df.groupby(['score_bin_no', 'score_bin'])['y'].agg(['count', 'sum'])
    regroup = regroup.rename(columns={'count': 'total_cnt', 'sum': 'bad_cnt'}).reset_index()
    # 计算模型累积值
    regroup['bad_pnt_md'] = regroup['bad_cnt'] / regroup['bad_cnt'].sum()
    regroup['bad_pnt'] = regroup['total_cnt'] / cnt_total
    regroup['bad_pnt_cum_md'] = regroup['bad_pnt_md'].cumsum()
    regroup['bad_pnt_cum'] = regroup['bad_pnt'].cumsum()
    regroup['lift'] = regroup['bad_pnt_cum_md'] / regroup['bad_pnt_cum']

    fig, ax = plt.subplots(2, 1, figsize=(15, 15))
    # 绘制累积LIFT图
    # 绘图
    ax[0].plot(df.loc[list_index, 'depth'], df.loc[list_index, 'lift_cum'], label='Lift(Model)',
               linewidth=3, marker='o')
    ax[0].plot(df.loc[list_index, 'depth'], [1] * n_split, label='Lift(Random)',
               linewidth=3, marker='o', color='r')
    ax[0].set_title('Cumulative Lift Chart', fontsize=15)
    ax[0].set_xlabel('Depth', fontsize=15)
    ax[0].set_ylabel('Lift', fontsize=15)
    ax[0].legend(loc='right upper', fontsize='x-large')
    ax[0].grid()
    ax[0].xaxis.set_major_locator(MultipleLocator(0.1))
    ax[0].yaxis.set_major_locator(MultipleLocator(1))
    # 绘制分数分组LIFT图
    width = 0.35
    ax[1].bar(regroup['score_bin_no'] - width / 2, regroup['bad_pnt_md'], width=width, color='#7DB2FA')
    ax[1].bar(regroup['score_bin_no'] + width / 2, regroup['bad_pnt'], width=width, color='#AFC4D6')
    ax_twinx_1 = ax[1].twinx()
    ax_twinx_1.plot(regroup['score_bin_no'], regroup['lift'], label='Lift', marker='o', color='r')
    ax[1].set_title('Lift Chart', fontsize=15)
    ax[1].set_xlabel('Score Group', fontsize=15)
    ax[1].set_ylabel('Pct. of Bad', fontsize=15)
    ax[1].set_xticks(range(n_split))
    ax[1].set_xticklabels(list(regroup['score_bin']), fontsize=12)
    ax_twinx_1.set_ylabel('Lift', fontsize=15)
    ax[1].grid()
    ax_twinx_1.legend(loc='best', fontsize='x-large')











