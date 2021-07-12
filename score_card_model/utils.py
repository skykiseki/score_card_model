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



def chi2(regroup):
    """
    计算并返回两个相邻组的卡方值

    当相邻两组组坏样本都为0（或1）时，over_badrate为0（或1），期望坏样本数为0（或1），实际与期望一样，卡方值为0；
    计算regroup的卡方值chi2

    ps:只适合用在二分类0,1问题
    ps:注意这里是排序的regroup

    Parameters:
    ----------
    regroup：输入的dataframe

    Returns:
    -------
    卡方值
    """
    # 先copy,后期会有加计算字段
    sub_rg = regroup.copy()
    # 计算总频数n
    num_total = sub_rg['num_feat_vals'].sum()

    # 计算bad在各箱的频数
    sub_rg['num_feat_good'] = sub_rg['num_feat_vals'] - sub_rg['num_feat_bad']
    # 计算bad的频数
    num_bad = sub_rg['num_feat_bad'].sum()
    # 计算good的频数
    num_good = sub_rg['num_feat_good'].sum()
    # 计算bad(class='1')的期望频率
    overall_badrate = num_bad / num_total
    # 计算good(class = '0')的期望频率
    overall_goodrate = num_good / num_total
    # 计算各个箱bad的期望频率
    sub_rg['num_bad_expected'] = sub_rg['num_feat_vals'] * overall_badrate
    # 计算各个箱good的期望频率
    sub_rg['num_good_expected'] = sub_rg['num_feat_vals'] * overall_goodrate
    # 计算bad下的卡方值chi2, 如果bad_rate为0, 卡方值也为0
    if overall_badrate == 0:
        chi2_bad = 0
    else:
        chi2_bad = ((sub_rg['num_feat_bad'] - sub_rg['num_bad_expected']) ** 2 / sub_rg['num_bad_expected']).sum()
    # 计算good下的卡方值chi2, 如果good_rate为0, 卡方值也为0
    if overall_goodrate == 0:
        chi2_good = 0
    else:
        chi2_good = ((sub_rg['num_feat_good'] - sub_rg['num_good_expected']) ** 2 / sub_rg['num_good_expected']).sum()
    # 返回总体的卡方值chi2_bad + chi2_good
    return chi2_bad + chi2_good



def chi2sum_id(regroup, field='all'):
    """
    找出需要待合并的分箱位置(id)

    Parameters:
    ----------
    regroup, 特征bad_rate的dataframe, *****但是这里的dataframe是经过排序的
    field:
    若field ='all', 则比较所有的相邻组, 可选的取值为[0, regroup.shape[0] - 1], 返回(id_min, id_min + 1)
    若field = 0, 则仅能与index=1的组进行合并, 返回1
    若field = regroup.shape[0] - 1,则仅能与index = regroup.shape[0] - 2的组进行合并
    若field = k, 其中k属于[1, regroup.shape[0] - 1, 若k与k+1的卡方值更小, 则返回k+1, 若k与k-1的卡方值更小, 则返回k

    Returns:
    -------
    id_min: 最小卡方值所在箱的索引 + 1
    """
    # 特征的取值数,即原始箱数
    vals_cnt = regroup.shape[0]
    # 判断field的输入是否正确
    if isinstance(field, str):
        if field != 'all':
            print("Incorrect field para is inputted.The parameter 'field' should equal to 'all'.")
    elif not isinstance(field, int):
        print("Incorrect field para is inputted.The type of field para should be either 'str' or 'int'.")
    else:
        if field < 0 or field >= vals_cnt:
            print("Incorrect field para is inputted.The range of fielf para should be from 0 to " + str(vals_cnt - 1))

    if field == 'all':
        # 遍历相邻两组的index,分别计算各自的卡方值
        list_chi2 = []
        for i in range(vals_cnt - 1):
            # 注意这个地方,iloc不取结尾部分的index, 所以是加2
            sub_regroup = regroup.iloc[i: i + 2]
            # 计算子集的卡方值
            chi2_subrg = chi2(sub_regroup)
            list_chi2.append(chi2_subrg)
        # 计算最小卡方值所在的索引, 返回+1值
        id_minchi2 = list_chi2.index(min(list_chi2))
        id_min = id_minchi2 + 1
    elif field == vals_cnt - 1:
        id_min = vals_cnt - 1
    elif field == 0:
        id_min = 1
    else:
        k = field
        # 计算k-1和k+1处的卡方值
        chi2_last = chi2(regroup.iloc[k - 1: k + 1])
        chi2_next = chi2(regroup.iloc[k: k + 2])
        if chi2_last <= chi2_next:
            id_min = k
        else:
            id_min = k + 1
    return id_min