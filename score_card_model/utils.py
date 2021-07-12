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


def order_regroup(regroup, feat_order=False, check_object='chi2sum_min'):
    """
    特征分组的排序

    变量分组合并需要遵循以下规则：
    1、离散无序less, 合并的相邻组怀样本率应相邻
    2、离散有序变量, 合并的相邻组不能破坏其有序性：
    3、连续型变量合并的相邻组取值大小相近

    具体单调性要求看：
    https://blog.csdn.net/shenxiaoming77/article/details/79548807

    Parameters:
    ----------
    regroup:dataframe, 特征的bad_rate的dataframe, func bin_badrate的返回值, 输入时未排序

    feat_order:boolean, 分箱的特征是否有序

    check_object:str,
    检测对象
    'badrarte' -> 针对badrate为0或1的情况,可选取值：
    'min_pnt' -> 针对某组样本总数占所有组样本总数的比例是否大于等于期望最小比例的情况；
    'chi2sum_min' -> 针对需要找到卡方值最小相邻组的情况；

    Returns:
    -------
    当检测对象为badrarte, 返回badrate为0或1的情况下需要合并的箱id
    当检测对象为min_pnt, 返回基于最小箱样本占比需要合并的箱id
    当检测对象为chi2sum_min, 返回基于卡方值需要合并的箱id
    """
    # copy一份dataframe,不能直接传参, 否则会变成引用
    rg = regroup.copy()
    # 如果特征有序, 则按其自然编码排序,即按其索引进行排序
    # 这里要求在输入的时候特征已经是排序编码好的, 如00,01,02,03......11,12
    # 如果特征无序, 则按其bad_rate进行排序
    if feat_order:
        rg = rg.sort_index()
    else:
        rg = rg.sort_values(by='bad_rate')

    if check_object == 'chi2sum_min':
        # 全局最小卡方值合并, 返回需要合并的id
        id_merge = chi2sum_id(rg)
        return rg, id_merge

    elif check_object == 'min_pnt':
        # 对指定的箱进行合并
        # 找出最小pnt所在的value label
        # ****pnt_feat_vals需要额外计算****
        label_minpnt = rg['pnt_feat_vals'].idxmin()
        # 找出对应的索引序列值
        id_minpnt = list(rg.index).index(label_minpnt)
        # 对对应的k值进行合并
        id_pnt = chi2sum_id(rg, field=id_minpnt)
        return rg, id_pnt

    elif check_object == 'bad_rate':
        # 对bad_rate为0或者1的箱进行合并?
        # 对bad_rate=0
        label_minbadrate = rg['bad_rate'].idxmin()
        id_badrate_min = list(rg.index).index(label_minbadrate)
        id_badrate_0 = chi2sum_id(rg, field=id_badrate_min)
        # 对bad_rate=1
        label_maxbadrate = rg['bad_rate'].idxmax()
        id_badrate_max = list(rg.index).index(label_maxbadrate)
        id_badrate_1 = chi2sum_id(rg, field=id_badrate_max)
        # 事实上, 当为无序时, 因为这个函数开始就做了排序, 所以下面的id_badrate_0永远是1, id_badrate_1永远是len - 1

        return rg, id_badrate_0, id_badrate_1

    else:
        print("Incorrect 'check_object' para is inputted.")


def refresh_vals_dict(dict_vals_ori, dict_vals_proc):
    """
    根据 dict规则对原始的取值dict进行更新

    Parameters:
    ----------
    dict_valsOri: 原始的取值字典,其中前者的值是后者的键
    dict_valsProc: 规则取值字典

    Returns:
    -------
    dict_res: 结果返回更新之后的字典

    E.g:
    ---
    如dict_valsOri = {'OWN':'0', 'NONE':'1'}, dict_valsProc = {'0':0, '1':0}
    返回的是{'OWN':0, 'NONE':0}

    """
    dict_res = {}
    for key, value in dict_vals_ori.items():
        if value in dict_vals_proc.keys():
            dict_res[key] = dict_vals_proc[value]
        else:
            dict_res[key] = dict_vals_ori[key]
    return dict_res