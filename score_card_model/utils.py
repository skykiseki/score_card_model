import pandas as pd


def bin_badrate(df, col_name, target):
    """
    计算dataframe某个特征的bad_rate，
    返回一个df

    Parameters:
    ----------
    df：输入的dataframe， featName：特征名称.

    targetName：目标变量名（y值 1 or 0）

    Returns:
    -------
    df
    """
    # 计算该特征的分布个数
    num_feat_vals = df.groupby(col_name)[target].count()
    # 计算该特征下等于1（bad）的个数
    num_feat_bad = df.groupby(col_name)[target].sum()
    # 创建一个dataframe
    redf = pd.DataFrame({'num_feat_vals': num_feat_vals, 'num_feat_bad': num_feat_bad})
    # 计算该特征的bad_rate
    redf['bad_rate'] = redf['num_feat_bad'] / redf['num_feat_vals']
    return redf