import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score

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



def merge_neighbour(regroup, id_target, feat_vals_dict):
    """
    合并分箱

    Parameters:
    ----------
    regroup, 输入的排序后的bad_rate结果,即order_regroup的返回值
    id_target, 所找到的最小卡方值箱子索引值, 这里的target_id为大于0,小于特征取值个数,即[1, num_vals - 1]
    feat_vals_dict, dict,原始的箱子取值遍历


    Returns:
    -------
    返回刷新组序列之后的取值dict
    """


    # 先初始化索引值,方便取值,也方便后续更新refresh_val_2bin以确保一一对应, 这里可以理解为GroupNO
    rg = regroup.reset_index()
    # 取得对应字典, 注意, 这里的key和value需要翻转, 用于refresh便捷
    reindex_dict = {v: k for k, v in rg[rg.columns[0]].to_dict().items()}
    # 将GroupNo更新进feat_vals_dict
    feat_vals_dict = refresh_vals_dict(feat_vals_dict, reindex_dict)
    # 取得所有的序值
    list_idxid = list(rg.index)
    # 取得序值的个数
    num_idxid = len(list_idxid)

    # 初始化合并dict, 包含的值为合并前的序值:合并后的序值,如{0:0, 1:0, 2:1},后面再利用这个dict_merge进行refresh
    dict_merge = {i: i for i in list_idxid}
    # 若遍历搜索到的所需合并的id为1(注意id返回的是合并组(k, k+1)中的k+1值), 则0组和1组合并, 合并进去0组(合并的箱子序值为k)
    # 其他组不变, 但是由于0,1合并,序值推前1, 所以其他组的序值需要减1
    # 若遍历搜索到的所需合并的id为最后一组, 即num_idxid - 1, 则合并(num_idxid - 2, num_idxid - 1), 合并进num_idxid - 2
    # 其他组不变, 由于是最后两组合并, 所以前面组的序值不变
    # 若遍历搜索到的所需合并的id在以上两种情况之外(即在中间), 则小于k值的均不变, 大于等于k值的前置1位,即减1
    for k, v in dict_merge.items():
        if id_target == 1:
            if k == 0:
                # 等价于dict_merge[0] = 0
                dict_merge[k] = v
            else:
                dict_merge[k] = v - 1
        elif id_target == num_idxid - 1:
            if k == num_idxid - 1:
                dict_merge[k] = v - 1
            else:
                dict_merge[k] = v
        else:
            if k < id_target:
                dict_merge[k] = v
            else:
                dict_merge[k] = v - 1
    # 返回刷新组序列之后的取值dict
    return refresh_vals_dict(feat_vals_dict, dict_merge)


def monotone_badrate(regroup, shape='mono', u=False):
    """
    判断组间坏样本率期望单调性（不严格，允许相邻组badrate相等）与实际单调性是否相符;不符时是否放宽至U形（严格，不允许相邻组badrate相等）

    U形时，极值索引不一定是索引中位数，可能存在偏移；符合期望单调或U型，则返回True，否则返回False

    Parameters:
    ----------
    regroup: badrate的返回值
    shape: 单调类型; 'mono' 单调增或者减, 'mono_up' 单调递增, 'mono_down' 单调递减
    u: 是否允许U型分布, 参数为False, 'u' 正U或者倒U, 'u_up' 正U, 'u_down' 倒u

    PS:注意输入的必须是有序的特征

    """
    # 以防万一, 做个预备的参数检查
    if shape not in ['mono', 'mono_up', 'mono_down'] or u not in [False, 'u', 'u_up', 'u_down']:
        raise Exception('错误的参数类型, 请重新检查.')

    # 先对regroup进行初始化排序, 基于索引排序
    rg = regroup.sort_index(ascending=True)
    # 取出对应的bad_rate
    list_badrate = list(rg['bad_rate'])
    # 取出对应的分组个数
    cnt_bins = len(list_badrate)

    # 先判断cnt_bins的个数,如果分组个数只有1个或者空,无所谓的单调不单调, 不做后续处理, 直接给默认True
    if cnt_bins <= 1:
        return True
    # 先检查是否满足严格的单调增或者单调减, 注意因为是相邻组的比较, 所以最大索引值要减1
    # 这个地方不能取等号, 原本的代码是有等号, 但是我们这里为了有区分度, 所以不能有等号
    is_mono_up = all([list_badrate[i] < list_badrate[i + 1] for i in range(cnt_bins - 1)])
    is_mono_down = all([list_badrate[i] > list_badrate[i + 1] for i in range(cnt_bins - 1)])
    # 根据期望的类型进行返回结果
    if shape == 'mono_up':
        ret_is_mono = is_mono_up
    elif shape == 'mono_down':
        ret_is_mono = is_mono_down
    else:
        ret_is_mono = any([is_mono_up, is_mono_down])

    # 如果以上的单调性结果为False,而期望的单调型还允许是U型, 则需再次检查
    if u is not False and ret_is_mono is False:
        # 先判断是否存在一定的等值, 如果存在, 则不是U型, 直接返回
        is_exist_equality = any([list_badrate[i] == list_badrate[i + 1] for i in range(cnt_bins - 1)])
        if is_exist_equality:
            return ret_is_mono

        # 先初始化bad_rate最小和最大值得组序
        id_min_bad_rate = list_badrate.index(min(list_badrate))
        id_max_bad_rate = list_badrate.index(max(list_badrate))

        # 判断是否为倒U型,极大值的索引序值不在头和尾, 且最大值(序值)左边单调增, 最大值(序值)右边单调减
        if id_max_bad_rate != 0 and id_max_bad_rate != cnt_bins - 1:
            is_left_up = all([list_badrate[i] < list_badrate[i + 1] for i in range(id_max_bad_rate)])
            is_right_down = all([list_badrate[i] > list_badrate[i + 1] for i in range(id_max_bad_rate, cnt_bins - 1)])
            is_u_down = all([is_left_up, is_right_down])
        else:
            is_u_down = False

        # 判断是否为正U型, 极小值的索引序值不在头和尾, 切最大值(序值)左边单调减, 最大值(序值)右边单调增
        if id_min_bad_rate != 0 and id_min_bad_rate != cnt_bins - 1:
            is_left_down = all([list_badrate[i] > list_badrate[i + 1] for i in range(id_min_bad_rate)])
            is_right_up = all([list_badrate[i] < list_badrate[i + 1] for i in range(id_min_bad_rate, cnt_bins - 1)])
            is_u_up = all([is_left_down, is_right_up])
        else:
            is_u_up = False

        # 更新返回结果
        if u == 'u_up':
            ret_is_mono = is_u_up
        elif u == 'u_down':
            ret_is_mono = is_u_down
        else:
            ret_is_mono = any([is_u_down, is_u_up])

    return ret_is_mono


def regroup_special_merge(regroup_nor, regroup_special):
    """
    在分箱的时候,可能存在特殊值不加入分箱,自成一箱,这个时候可能存在这些自成一箱的部分bad_rate为0或者1
    在bad_rate为0或者1的时候, 这时候分箱属于极限分布,由于log0下为无限,所以可以手工+-1来调整最后的badrate取值(由于单调, 这时候不会很大影响)

    Parameters:
    ----------
    regroup_nor: 正常部分的regourp
    regroup_special: 特殊值部分的regourp

    Returns:
    -------
    修正num_feat_bad,bad_rate后的regroup
    """
    # 如果特殊值的分组存在bad_rate为0的情况
    while regroup_special['bad_rate'].min() == 0:
        # 先找出bad_rate为0的索引值(标签)
        label_0 = regroup_special['bad_rate'].idxmin()
        # 将该组的bad数调整为1, 然后重新计算bad_rate
        regroup_special.loc[label_0, 'num_feat_bad'] = 1
        regroup_special['bad_rate'] = regroup_special['num_feat_bad'] / regroup_special['num_feat_vals']

    # 如果特殊值得分组存在bad_rate为1的情况
    while regroup_special['bad_rate'].max() == 1:
        # 找出bad_rate为1的索引值(标签)
        label_1 = regroup_special['bad_rate'].idxmax()
        # 将该组的bad数调整减1,然后重新计算bad_rate
        regroup_special.loc[label_1, 'num_feat_bad'] -= 1
        regroup_special['bad_rate'] = regroup_special['num_feat_bad'] / regroup_special['num_feat_vals']

    # 如果两个regroup或者其中之一存在'pnt_feat_vals'字段, 要先剔除
    if 'pnt_feat_vals' in regroup_nor.columns:
        regroup_nor = regroup_nor.drop('pnt_feat_vals', axis=1)
    if 'pnt_feat_vals' in regroup_special:
        regroup_special = regroup_special.drop('pnt_feat_vals', axis=1)

    # 返回合并的dataframe
    return pd.concat([regroup_nor, regroup_special], axis=0)


def check_if_min_pnt(regroup, row_num, min_pnt=0.05):
    """
    检查是否存在样本数占比小于阈值的分组

    Parameters:
    ----------
    regroup: badrate group
    row_num: df.shape[0]
    min_pnt: 单个屬性最小占比

    Returns:
    bool
    """

    rg = regroup.copy()
    # 如果不存在pnt_feat_vals, 则重新创建一个
    if 'pnt_feat_vals' not in rg.columns:
        rg['pnt_feat_vals'] = rg['num_feat_vals'] / row_num
    if rg.loc[rg['pnt_feat_vals'] < min_pnt].shape[0] > 0:
        return True
    else:
        return False


def cal_woe_iv(regroup_woe_iv):
    """
    计算regroup(regroup_woe_iv)的WOE和IV
    这个时候的regroup_woe_iv只有num_feat_vals、num_feat_bad、bad_rate三个字段

    Parameters:
    ----------
    regroup_woe_iv: badrate regroup

    Returns:
    -------
    regroup_woe_iv: 增加woe,iv等列的badrate regroup
    dict_woe: 特征与woe的映射
    iv: 特征iv值
    """

    # 计算总坏样本数
    num_total_bad = regroup_woe_iv['num_feat_bad'].sum()
    # 计算sub_bad_rate, 该组坏样本数/总坏样本数
    regroup_woe_iv['sub_bad_rate'] = regroup_woe_iv['num_feat_bad'] / num_total_bad

    # 计算num_feat_good, 好样本数
    regroup_woe_iv['num_feat_good'] = regroup_woe_iv['num_feat_vals'] - regroup_woe_iv['num_feat_bad']
    # 计算总好样本数
    num_total_good = regroup_woe_iv['num_feat_good'].sum()
    # 计算sub_good_rate, 该组好样本数/总好样本数
    regroup_woe_iv['sub_good_rate'] = regroup_woe_iv['num_feat_good'] / num_total_good
    # 计算woe值, woe = ln(bad_rate/good_rate)
    regroup_woe_iv['woe'] = np.log(regroup_woe_iv['sub_bad_rate'] / regroup_woe_iv['sub_good_rate'])
    # 计算iv, iv = (good_rate - bad_rate) * woe
    regroup_woe_iv['iv'] = (regroup_woe_iv['sub_bad_rate'] - regroup_woe_iv['sub_good_rate']) * regroup_woe_iv['woe']
    # 返回regourp_woe_iv, dict_woe, 特征的IV值
    dict_woe = regroup_woe_iv['woe'].to_dict()
    iv = regroup_woe_iv['iv'].sum()
    return regroup_woe_iv, dict_woe, iv


def init_split(df, featname, init_bins=100):
    """
    对df下的featname特征进行分割, 最后返回中间的分割点刻度
    为了保证所有值都有对应的区间, 取两个值之间的中值作为分割刻度
    注意这里的分割方式不是等频等常用的方法, 仅仅是简单地找出分割点再进行融合最终进行分割
    注意, 分出的箱刻度与是否闭区间无关, 这个点取决于用户,这个函数仅考虑分箱的个数
    同时, 分箱多余的部分会进入最后一个箱, 如101个分100箱, 则最后一个箱有两个样本

    Parameters:
    ----------
    df: dataframe,输入的df,
    featname:str, 特征名称
    init_bins:int, 需要分的箱个数

    Returns:
    -------
    返回分割的刻度列表(升序)，如[1,5,9,18]

    """
    # 初始化取值个数列表, 同时排序
    list_unique_vals_order = sorted(list(set(df[featname])))
    # 取得中间的刻度值, 注意是遍历到len - 1
    list_median_vals = []
    for i in range(len(list_unique_vals_order) - 1):
        list_median_vals.append((list_unique_vals_order[i] + list_unique_vals_order[i + 1]) / 2)

    # 初始化初始分箱的个数,
    cnt_unique_vals = len(list_median_vals)
    # 如果初始分箱个数小于init_bins了, 则直接返回
    # 如果初始分箱个数大于init_bins, 则从头开始抓取init_bins个值,所以剩余值会留在最后一组
    if cnt_unique_vals <= init_bins:
        return list_median_vals
    else:
        # 计算每个箱的个数, 注意这里要用求商
        cnt_perbin = cnt_unique_vals // init_bins
        # 取得中间的init_bins个值
        list_median_vals = [list_median_vals[i * cnt_perbin] for i in range(init_bins - 1)]
        return list_median_vals


def feat_bins_split(df, featname, init_bins=100):
    """
    对超过维度阈值(默认100)的特征进行分组处理, 再返回分组字典, 对不超过维度阈值的特征则直接返回分组字典
    起始的组总是以(-1, vals]开始, 以(vals, inf)结束

    Parameters:
    ----------
    df: dataframe, 输入的dataframe
    featname: str, 特征名
    init_bins: 初始化的分箱个数

    Returns:
    -------
    cnt_unique_vals+1个分组数(左右两边,加间隔), 返回值格式为{interval0:排序值0, interval1:排序值1......}

    PS: 返回的区间是正序的, 注意*******
    """
    dict_vals_to_bins = {}
    # 先取出分割的刻度
    list_unique_vals_order = init_split(df, featname=featname, init_bins=init_bins)
    # 计算刻度的个数, 最终箱子数为刻度数 + 1
    cnt_unique_vals = len(list_unique_vals_order)
    # 取得最大最小值
    min_value, max_value = min(list_unique_vals_order), max(list_unique_vals_order)

    # 先处理起始区间和中间区间, 再在最后添加一个末尾区间(因为i在遍历的时候最后一个元素要遍历两次, 为了好理解就分开处理)
    for i in range(cnt_unique_vals):
        if i == 0:
            interval_label = '(-1,%.4f]' % min_value
            dict_vals_to_bins[interval_label] = 0
        else:
            interval_val_l = list_unique_vals_order[i - 1]
            interval_val_r = list_unique_vals_order[i]
            interval_label = '({0:.4f},{1:.4f}]'.format(interval_val_l, interval_val_r)
            dict_vals_to_bins[interval_label] = i
    # 插入末尾区间
    interval_label = '(%.4f,+Inf)' % max_value
    dict_vals_to_bins[interval_label] = cnt_unique_vals

    return dict_vals_to_bins


def intervals_split_merge(list_lab_intervals):
    """
    对界限列表进行融合

    e.g.
    如['(2,5]', '(5,7]'], 融合后输出为 '(2,7]'

    Parameters:
    ----------
    list_lab_intervals: list, 界限区间字符串列表

    Returns:
    -------
    label_merge: 合并后的区间
    """
    list_labels = []
    # 遍历每个区间, 取得左值右值字符串组成列表
    for lab in list_lab_intervals:
        for s in lab.split(','):
            list_labels.append(s.replace('(', '').replace(')', '').replace(']', ''))
    list_lab_vals = [float(lab) for lab in list_labels]
    # 取得最大最小值的索引
    id_max_val = list_lab_vals.index(max(list_lab_vals))
    id_min_val = list_lab_vals.index(min(list_lab_vals))
    # 取得最大最小值的字符串
    lab_max_interval = list_labels[id_max_val]
    lab_min_interval = list_labels[id_min_val]
    # 如果右边界限的值为+Inf,则改为')', 其他为']'
    l_label = '('
    if lab_max_interval == '+Inf':
        r_label = ')'
    else:
        r_label = ']'
    label_merge = l_label + lab_min_interval + ',' + lab_max_interval + r_label
    return label_merge


def merge_intervals(dict_vals_to_bins):
    """
    对合并的连续值区间进行

    e.g.
    如{'(-1, 2]': 0, '(2, 5]': 1, '(5, 7]': 1, '(7, +Inf)': 2}
    返回{'(-1, 2]': 0, '(2, 7]': 1, '(7, +Inf)': 2}

    Parameters:
    ----------
    dict_vals_to_bins: dict, 输入的分组取值

    Returns:
    -------
    res_dict_vals_to_bins, 合并后的分组取值
    """

    # 拷贝
    dup_dict_vals_to_bins = dict_vals_to_bins.copy()
    # 返回结果
    res_dict_vals_to_bins = {}
    # 特殊值字典
    sp_dict_vals_to_bins = {}

    # 先保存特殊值, 并且从初始的字典中删除特殊值
    for key, val in dict_vals_to_bins.items():
        if not isinstance(key, str):
            sp_dict_vals_to_bins[key] = val
            del dup_dict_vals_to_bins[key]

    # 计算初始的字典元素序值与字典元素key
    list_keys = list(dup_dict_vals_to_bins.keys())
    list_values = list(dup_dict_vals_to_bins.values())
    # 对索引序值进行分组合并
    for id_val in sorted(set(list_values)):
        list_lab_intervals = []
        for i in range(len(list_values)):
            # 如果组序(去重)和列表值(重复)相等, 则取出对应列表序值对应的分组
            if id_val == list_values[i]:
                list_lab_intervals.append(list_keys[i])
        label_interval = intervals_split_merge(list_lab_intervals)
        res_dict_vals_to_bins[label_interval] = id_val
    # 合并特殊值
    res_dict_vals_to_bins.update(sp_dict_vals_to_bins)
    # 做个排序
    res_dict_vals_to_bins = {k: v for k,v in sorted(res_dict_vals_to_bins.items(), key=lambda x: x[1])}
    return res_dict_vals_to_bins


def value_to_intervals(value, dict_valstoinv):
    """

    根据字典对值返回对应的区间

    e.g.
    dict_valstoinv:{(-1, 100]: 0, (100, +Inf): 1}
    value: 50
    返回值(-1, 100]

    PS:如果存在有重复匹配的区间, 则取第一个匹配上的

    Parameters:
    ----------
    value: number, 数值
    dict_valstoinv: dict, 区间字典

    Returns:
    -------
    数值对应的区间

    """


    for key in list(dict_valstoinv.keys()):
        if isinstance(key, str):
            label_l = key.split(',')[0]
            label_r = key.split(',')[1]
            value_l = float(label_l[1:])
            value_r = float(label_r[:-1])
            # 两边都是开区间
            if label_l[0] == '(' and label_r[-1] == ')' and value_l < value < value_r:
                return key
            # 左开右闭
            elif label_l[0] == '(' and label_r[-1] == ']' and value_l < value <= value_r:
                return key
            # 左闭右开
            elif label_l[0] == '[' and label_r[-1] == ')' and value_l <= value < value_r:
                return key
            # 两边都是闭区间
            elif label_l[0] == '[' and label_r[-1] == ']' and value_l <= value <= value_r:
                return key
            # 如果没有遍历到, 则遍历下一个key
            else:
                continue
        else:
            if value == key:
                return key
    # 如果整体都没有遍历到, 则可能是字典或者数值有问题, 抛出异常
    str_exception = '没有对应的区间.请重新检查.'
    raise Exception(str_exception)



def chi2_cutting_discrete(df_data, feat_list, target,
                          special_feat_val={},
                          max_intervals=8, min_pnt=0.05,
                          discrete_order={}, mono_expect={}):
    """

    类别型特征(有序&无序少)卡方分箱

    Parameters:
    ----------
    df_data: 训练集,
    feat_list: 参与分箱的特征,
    target: y值特征名称
    special_feat_val: dict, 某个特征下不参与分箱的特殊值,具体格式如下:
    {特征名1: [特殊值1...特殊值r], 特征名2: [特殊值1...特殊值o, ......], 特征名k: [特殊值1...特殊值n]}
    max_intervals: 非0整数,最大分箱数, 默认8
    min_pnt: 最小分箱数目占比, 默认0.05, 取(0,1)之间
    discrete_order: dict, 表示离散有序特征, 具体格式如下:
    如{特征名1:{特征值1:序值1,...,特征值n:序值n} ,..., 特征名k:{特征值1:序值1,...,特征值n:序值n}};
    注意这个地方, 有序特征的有序性在这个特征中体现,函数中会根据这个更新分组序值
    mono_except: dict, 默认空表变量离散无序，无需检查单调性；非空dict表离散有序，需要检查badrate单调性,参数赋值
    形如{ 特征名1:{'shape':期望单调性,'u':不单调时，是否允许U形} ,..., 特征名k:{'shape':期望单调性,'u':不单调时，是否允许U形}}，
     'shape'可选择参数：'mono'期望单调增或减，'mono_up'期望单调增，'mono_down'期望单调减；
    'u'可选参数：'u'正U（开口向上）或倒U（开口向下），'u_up'正U，'u_down'倒U, False 不允许U型

    Returns:
    -------
    dict_discrete_feat_to_bins:  特征的分组取值, 形式为{featname:{val1: 序值1, val2: 序值2} }
    dict_discrete_iv: woe编码后的IV; 形式为:{featname: iv值}
    dict_discrete_woe: 分组后的woe值; {featname: {0: woe值, 1: woe值, ......, k: woe值} }

    """
    # 先判断参数是否正确
    # 判断df_train是否为空, 是否为正确的类型, 是否为0行
    if df_data is None:
        raise Exception('None dataframe is inputed.')
    elif not isinstance(df_data, pd.DataFrame):
        raise Exception('Input is not a dataframe.')
    elif df_data.shape[0] == 0:
        raise Exception('The dataframe has a row num of 0.')
    # 判断分箱特征列表是否为空
    elif (len(feat_list) == 0) or (feat_list is None):
        raise Exception('Empty list is inputed.')
    # 判断最大分箱数max_intervals是否为正整数
    elif (max_intervals < 0) or not isinstance(max_intervals, int):
        raise Exception('Max_intervals is incorrect.')
    # 判断最小分箱数占比min_pnt是否为(0,1)
    elif (min_pnt <= 0) or (min_pnt >= 1):
        raise Exception('Min_pnt is incorrect.')

    # 该函数不允许有序性存在特殊值
    for key in discrete_order.keys():
        if key in special_feat_val.keys():
            str_exception = "'%s' is orderly feature and exists in dictionary of speacial values." % key
            raise Exception(str_exception)

    # 判断特殊值是否真实存在
    if len(special_feat_val) > 0:
        for featName, valsList in special_feat_val.items():
            list_feat_vals = list(set(df_data[featName]))
            for val in valsList:
                if val not in list_feat_vals:
                    raise Exception("'{0}' does not exist in feature '{1}'.".format(val, featName))

    # 计算dataframe的行数
    row_num = df_data.shape[0]

    dict_discrete_feat_to_bins = {}
    dict_discrete_iv = {}
    dict_discrete_woe = {}

    # 开始遍历
    for feat in tqdm(feat_list, desc="Cutting discrete features"):
        # 参数初始化
        intervals = max_intervals
        # 初始化dataframe
        df = df_data.loc[:, [feat, target]]

        # 初始化regroup_ori, 以badrate进行升序排序
        regroup_ori = bin_badrate(df, col_name=feat, target=target)
        regroup_ori = regroup_ori.sort_values(by='bad_rate')
        # 初始化list_index
        list_index = list(regroup_ori.index)
        # 默认先按badrate进行排序,产生初始的取值字典
        feat_val_to_bins = {i: list_index.index(i) for i in list_index}

        # 先判断是否存在特殊值special_feat_val,如果存在,则将最大分箱数减去特殊值的个数,然后剔除特殊值
        if len(special_feat_val) > 0 and feat in special_feat_val.keys():
            intervals -= len(special_feat_val[feat])
            # 复制一份特殊值的df
            df_special = df.loc[df[feat].isin(special_feat_val[feat])]
            # 更新df(剔除特殊值)
            df = df.loc[~df[feat].isin(special_feat_val[feat])]
            # 更新feat_valToBins
            regroup_ori = bin_badrate(df, col_name=feat, target=target)
            regroup_ori = regroup_ori.sort_values(by='bad_rate')
            list_index = list(regroup_ori.index)
            feat_val_to_bins = {i: list_index.index(i) for i in list_index}

        # 创建初始regroup_init
        # 创建feat + '_init'表示分组号
        df[feat + '_init'] = df[feat].map(feat_val_to_bins)
        # regroup_init = bin_badrate(df, col_name=feat + '_init', target=target)

        # 若特征属于有序离散型特征
        if feat in discrete_order.keys():
            # 由于有序,需要对feat_valToBins的序值进行重新排列
            feat_val_to_bins.update(discrete_order[feat])
            feat_val_to_bins = {k: v for k, v in sorted(feat_val_to_bins.items(), key=lambda x: x[1])}
            # 更新df
            df[feat + '_init'] = df[feat].map(feat_val_to_bins)
            # 初始计算regroup_init
            regroup_init = bin_badrate(df, col_name=feat + '_init', target=target)

            # 对有序的离散特征进行最小卡方值合并策略
            while regroup_init.shape[0] > intervals:
                # 先找到需要合并的最小卡方值索引序列号id_merge
                # 注意返回的regroup_init是排序过的
                regroup_init, id_merge_init = order_regroup(regroup_init, feat_order=True)
                # 进行组合并, 同时更新取值字典
                feat_val_to_bins = merge_neighbour(regroup_init, id_merge_init, feat_val_to_bins)
                # 更新feat_init列
                df[feat + '_init'] = df[feat].map(feat_val_to_bins)
                # 计算合并后的bad_rate
                regroup_init = bin_badrate(df, col_name=feat + '_init', target=target)

        # 下面开始判断bad_rate=1 或者 bad_rate=0的情况
        # 先初始化feat_badrate组序号
        df[feat + '_badrate'] = df[feat + '_init']
        # 初始化regroup_badrate
        regroup_badrate = bin_badrate(df, col_name=feat + '_badrate', target=target)

        # 判断是否存在bad_rate =1 或者 bad_rate=0的情况
        while regroup_badrate['bad_rate'].min() == 0 or regroup_badrate['bad_rate'].max() == 1:
            # 如果是有序的特征,排序后找出badrate为0或者badrate为1的分组
            # 如果是无序的特征,同上,但注意这个时候是用badrate进行排序
            if feat in discrete_order.keys():
                regroup_badrate, id_badrate_0, id_badrate_1 = order_regroup(regroup_badrate, feat_order=True,
                                                                            check_object='bad_rate')
            else:
                regroup_badrate, id_badrate_0, id_badrate_1 = order_regroup(regroup_badrate, feat_order=False,
                                                                            check_object='bad_rate')
            # 排序后找出需要合并的id_merge
            if regroup_badrate['bad_rate'].min() == 0:
                id_merge_badrate = id_badrate_0
            else:
                id_merge_badrate = id_badrate_1
            # 开始合并, 并且更新feat_valToBins
            feat_val_to_bins = merge_neighbour(regroup_badrate,
                                               id_target=id_merge_badrate,
                                               feat_vals_dict=feat_val_to_bins)

            # 更新feat + '_badrate'的值
            df[feat + '_badrate'] = df[feat].map(feat_val_to_bins)
            # 更新合并后的regroup_badrate
            regroup_badrate = bin_badrate(df, col_name=feat + '_badrate', target=target)

        # 下面开始判断每个分箱的样本数占比是否小于阈值(默认5%)
        # 初始化df和regroup,用feat + '_min_pnt'表示
        df[feat + '_min_pnt'] = df[feat + '_badrate']

        regroup_min_pnt = bin_badrate(df, col_name=feat + '_min_pnt', target=target)
        # 增加pnt_feat_vals字段, 注意分母是总体样本值, 前面如果出现特殊值, 则需要加回来
        regroup_min_pnt['pnt_feat_vals'] = regroup_min_pnt['num_feat_vals'] / row_num

        # 开始处理分组样本数小于阈值的情况
        while regroup_min_pnt['pnt_feat_vals'].min() < min_pnt:
            # 注意区分有序和无序特征
            # 返回值为rg & 合并的id
            if feat in discrete_order.keys():
                regroup_min_pnt, id_merge_pnt = order_regroup(regroup_min_pnt,
                                                              feat_order=True,
                                                              check_object='min_pnt')
            else:
                regroup_min_pnt, id_merge_pnt = order_regroup(regroup_min_pnt,
                                                              feat_order=False,
                                                              check_object='min_pnt')
            # 开始进行合并, 更新分组取值
            feat_val_to_bins = merge_neighbour(regroup_min_pnt,
                                               id_target=id_merge_pnt,
                                               feat_vals_dict=feat_val_to_bins)
            # 对df进行更新
            df[feat + '_min_pnt'] = df[feat].map(feat_val_to_bins)
            # 更新regroup_min_pnt
            regroup_min_pnt = bin_badrate(df, col_name=feat + '_min_pnt', target=target)
            # 重新计算pnt_feat_vals
            regroup_min_pnt['pnt_feat_vals'] = regroup_min_pnt['num_feat_vals'] / row_num

        # 开始检查单调性
        # 先进行初始化df & regroup, 以feat + '_mono'统计最新分组,
        df[feat + '_mono'] = df[feat + '_min_pnt']
        regroup_mono = bin_badrate(df, col_name=feat + '_mono', target=target)

        # 仅对有序离散特征进行检查单调性
        if feat in mono_expect.keys():
            # 若存在特征列入单调检查, 但是不属于有序特征, 需要报错
            if feat not in discrete_order.keys():
                raise Exception("'%s'特征不属于有序变量, 不需要进行单调性检查, 请重新整理." % feat)
            # 检查feat是否单调
            is_mono = monotone_badrate(regroup_mono,
                                       shape=mono_expect[feat]['shape'],
                                       u=mono_expect[feat]['u'])

            while not is_mono:
                # 更新regroup_mono以及遍历需要更新的分组序值id
                regroup_mono, id_merge_mono = order_regroup(regroup_mono, feat_order=True)
                # 更新feat_valToBins
                feat_val_to_bins = merge_neighbour(regroup_mono,
                                                   id_target=id_merge_mono,
                                                   feat_vals_dict=feat_val_to_bins)
                # 更新df, feat + '_mono'
                df[feat + '_mono'] = df[feat].map(feat_val_to_bins)
                # 更新regroup_mono
                regroup_mono = bin_badrate(df, col_name=feat + '_mono', target=target)
                # 再次检查单调性, 更新is_mono
                is_mono = monotone_badrate(regroup_mono,
                                           shape=mono_expect[feat]['shape'],
                                           u=mono_expect[feat]['u'])

        # 初始化特殊值的regroup_woe_iv
        regroup_woe_iv = regroup_mono.copy()

        # 对特殊值进行处理
        if feat in special_feat_val.keys() and len(special_feat_val[feat]) > 0:
            # 先更新取值字典
            # 取值字典加回特殊值, 注意组序为原来组序最大值+1开始计
            maxidx_feat_val_to_bins = max(feat_val_to_bins.values())
            list_sp_vals = special_feat_val[feat]
            for i in range(len(list_sp_vals)):
                feat_val_to_bins[list_sp_vals[i]] = maxidx_feat_val_to_bins + 1 + i
            # 更新df_special, 新增一个分组序号的列
            df_special[feat + '_init'] = df_special[feat].map(feat_val_to_bins)
            # 计算df_special的bad_rate
            regroup_special = bin_badrate(df_special, col_name=feat + '_init', target=target)
            # 返回合并特殊值分组后的regroup_woe_iv
            regroup_woe_iv = regroup_special_merge(regroup_woe_iv, regroup_special=regroup_special)
            # 注意, 如果最后的regroup存在分组占比数小于min_pnt的, 需要提示, 但不做处理


        # 计算分组的woe和iv
        # 注意这里用的参数是regroup_woe_iv
        # 这个时候只有num_feat_vals、num_feat_bad、bad_rate三个字段
        regroup_woe_iv, dict_woe, iv = cal_woe_iv(regroup_woe_iv)

        # 对分组取值进行排序
        feat_val_to_bins = {k: v for k, v in sorted(feat_val_to_bins.items(), key=lambda x: x[1])}
        # 插入
        dict_discrete_feat_to_bins[feat] = feat_val_to_bins
        dict_discrete_iv[feat] = iv
        dict_discrete_woe[feat] = dict_woe

    return dict_discrete_feat_to_bins, dict_discrete_iv, dict_discrete_woe


def chi2_cutting_continuous(df_data, feat_list, target,
                            discrete_more_feats=[],
                            special_feat_val={},
                            max_intervals=5, min_pnt=0.05,
                            mono_expect={}):
    """
    连续性特征分箱
    其中无序离散多型的特征会做badrate编码,以badrate作为连续值进行处理
    连续值均有序

    Parameters:
    ----------
    df_train: dataframe, 训练集,
    feat_list: list, 参与分箱的特征, 其中feat_list需要包含discrete_more_feats
    target: str, y值特征名称
    discrete_more_feats: 大于阈值分箱数的特征名称列表[特证名1, 特征名2......]
    special_feat_val: dict, 某个特征下不参与分箱的特殊值,具体格式如下:
    {特征名1: [特殊值1...特殊值r], 特征名2: [特殊值1...特殊值o, ......], 特征名k: [特殊值1...特殊值n]}
    PS:实际运用场景中, 这里只会有-1一个特殊值

    max_intervals: 非0整数,最大分箱数, 默认5
    min_pnt: 最小分箱数目占比, 默认0.05, 取(0,1)之间
    discrete_order: dict, 表示离散有序特征, 具体格式如下:
    如{特征名1:{特征值1:序值1,...,特征值n:序值n} ,..., 特征名k:{特征值1:序值1,...,特征值n:序值n}};
    注意这个地方, 有序特征的有序性在这个特征中体现,函数中会根据这个更新分组序值
    mono_except: dict, 默认空表变量离散无序，无需检查单调性；非空dict表离散有序，需要检查badrate单调性,参数赋值
    形如{ 特征名1:{'shape':期望单调性,'u':不单调时，是否允许U形} ,..., 特征名k:{'shape':期望单调性,'u':不单调时，是否允许U形}}，
        'shape'可选择参数：'mono'期望单调增或减，'mono_up'期望单调增，'mono_down'期望单调减；
        'u'可选参数：'u'正U（开口向上）或倒U（开口向下），'u_up'正U，'u_down'倒U, False 不允许U型

    Returns:
    -------
    dict_contin_feat_to_bins:
    无序离散多型特征的分组取值, 形式为{featname:{val1: 序值1, val2: 序值2} }
    连续型特征的分组取值, 形式为{featname: { (-1, val1]: 序值1, (val1, val2]: 序值2} }

    dict_contin_iv: woe编码后的IV; 形式为:{featname: iv值}
    dict_contin_woe: 分组后的woe值; {featname: {0: woe值, 1: woe值, ......, k: woe值} }

    """
    # 先判断参数是否正确
    # 判断df_train是否为空, 是否为正确的类型, 是否为0行
    if df_data is None:
        raise Exception('None dataframe is inputed.')
    elif not isinstance(df_data, pd.DataFrame):
        raise Exception('Input is not a dataframe.')
    elif df_data.shape[0] == 0:
        raise Exception('The dataframe has a row num of 0.')
    # 判断分箱特征列表是否为空
    elif (len(feat_list) == 0) or (feat_list is None):
        raise Exception('Empty list is inputed.')
    # 判断最大分箱数max_intervals是否为正整数
    elif (max_intervals < 0) or not isinstance(max_intervals, int):
        raise Exception('Max_intervals is incorrect.')
    # 判断最小分箱数占比min_pnt是否为(0,1)
    elif (min_pnt <= 0) or (min_pnt >= 1):
        raise Exception('Min_pnt is incorrect.')

    # 判断特殊值是否真实存在
    if len(special_feat_val) > 0:
        for featName, valsList in special_feat_val.items():
            list_feat_vals = list(set(df_data[featName]))
            for val in valsList:
                if val not in list_feat_vals:
                    raise Exception("'{0}' does not exist in feature '{1}'.".format(val, featName))

    # 计算dataframe的行数
    row_num = df_data.shape[0]

    dict_contin_feat_to_bins = {}
    dict_contin_iv = {}
    dict_contin_woe = {}

    # 开始处理
    for feat in tqdm(feat_list, desc="Cutting continuous features"):
        # 参数初始化, 和离散型不同, 这里不需要feat_valToBins
        intervals = max_intervals
        df = df_data.loc[:, [feat, target]]

        # 处理顺序是df_transf, df_special, df_bins, df_init, df_merge, df_badrate, df_min_pnt, df_mono
        # 如果特征类型属于无序离散多, 则需要先进行badrate编码,转化为连续值后再进行处理
        # 如果特征类型本身就是连续值, 不需要转化处理
        if feat in discrete_more_feats:
            regroup_tf = bin_badrate(df, col_name=feat, target=target)
            dict_feat_to_badrate = regroup_tf['bad_rate'].to_dict()
            df[feat + '_transf'] = df[feat].map(dict_feat_to_badrate)

        else:
            df[feat + '_transf'] = df[feat]

        # 先做特殊值处理, 在连续型特征里一般以-1作为特殊值单独一组
        if len(special_feat_val) > 0 and feat in special_feat_val.keys():
            intervals -= len(special_feat_val[feat])
            # 复制一份特殊值的df
            df_special = df.loc[df[feat].isin(special_feat_val[feat])]
            # 同时也创建feat + '_bins'列
            df_special[feat + '_bins'] = df_special[feat + '_transf']
            # 更新df(剔除特殊值)
            df = df.loc[~df[feat].isin(special_feat_val[feat])]

        # 基于transf后的数据生成分组字典, 再初始化df_bins
        dict_vals_to_bins = feat_bins_split(df, featname=feat + '_transf', init_bins=100)
        df[feat + '_bins'] = df[feat + '_transf'].apply(lambda x: value_to_intervals(value=x,
                                                                                     dict_valstoinv=dict_vals_to_bins))
        # 初始化df_init
        df[feat + '_init'] = df[feat + '_bins'].map(dict_vals_to_bins)

        # 初始化df_merge, regroup_merge
        df[feat + '_merge'] = df[feat + '_init']
        regroup_merge = bin_badrate(df, col_name=feat + '_bins', target=target)

        # 对排序的的连续特征进行最小卡方值合并策略(直接参考离散型)
        # 特征是连续型,所以肯定是有序的
        while regroup_merge.shape[0] > intervals:
            # 先找到需要合并的最小卡方值索引序列号id_merge
            # 注意返回的regroup_init是排序过的
            regroup_merge, id_merge = order_regroup(regroup_merge, feat_order=True)
            # 进行组合并, 同时更新取值字典
            dict_vals_to_bins = merge_neighbour(regroup_merge, id_merge, dict_vals_to_bins)
            # 更新feat_init列
            df[feat + '_merge'] = df[feat + '_bins'].map(dict_vals_to_bins)
            # 计算合并后的bad_rate
            regroup_merge = bin_badrate(df, col_name=feat + '_merge', target=target)

        # 初始化df_badrate, regroup_badrate
        df[feat + '_badrate'] = df[feat + '_merge']
        regroup_badrate = bin_badrate(df, col_name=feat + '_badrate', target=target)

        # 开始检查badrate为0或者1的情况:
        while regroup_badrate['bad_rate'].max() == 1 or regroup_badrate['bad_rate'].min() == 0:
            # 遍历寻找需要合并的分组id
            regroup_badrate, id_badrate_0, id_badrate_1 = order_regroup(regroup_badrate,
                                                                        feat_order=True,
                                                                        check_object='bad_rate')
            if regroup_badrate['bad_rate'].max() == 1:
                id_merge_badrate = id_badrate_1
            else:
                id_merge_badrate = id_badrate_0

            # 基于卡方进行合并, 更新取值字典
            dict_vals_to_bins = merge_neighbour(regroup_badrate,
                                              id_target=id_merge_badrate,
                                              feat_vals_dict=dict_vals_to_bins)
            # 重新更新df和regroup_badrate
            df[feat + '_badrate'] = df[feat + '_bins'].map(dict_vals_to_bins)
            regroup_badrate = bin_badrate(df, col_name=feat + '_badrate', target=target)

        # 初始化df_min_pnt, regroup_min_pnt, pnt_feat_vals
        df[feat + '_min_pnt'] = df[feat + '_badrate']
        regroup_min_pnt = bin_badrate(df, col_name=feat + '_min_pnt', target=target)
        regroup_min_pnt['pnt_feat_vals'] = regroup_min_pnt['num_feat_vals'] / row_num

        # 开始检查分组样本数占比是否低于阈值(默认5%)
        while regroup_min_pnt['pnt_feat_vals'].min() < min_pnt:
            # 和无序离散少&有序离散有一点区别, 连续型的值可能会存在最终合并为1组的情况,所以要做个判断
            # 如果合并至仅剩余两组,则直接跳出
            if regroup_min_pnt.shape[0] == 2:
                break
            # 遍历找到需要合并的分组
            regroup_min_pnt, id_merge_pnt = order_regroup(regroup_min_pnt,
                                                          feat_order=True,
                                                          check_object='min_pnt')
            # 开始进行合并, 更新分组取值
            dict_vals_to_bins = merge_neighbour(regroup_min_pnt,
                                              id_target=id_merge_pnt,
                                              feat_vals_dict=dict_vals_to_bins)
            # 对df进行更新
            df[feat + '_min_pnt'] = df[feat + '_bins'].map(dict_vals_to_bins)
            # 更新regroup_min_pnt
            regroup_min_pnt = bin_badrate(df, col_name=feat + '_min_pnt', target=target)
            # 重新计算pnt_feat_vals
            regroup_min_pnt['pnt_feat_vals'] = regroup_min_pnt['num_feat_vals'] / row_num

        # 开始检查单调性
        # 先进行初始化df & regroup, 以feat + '_mono'统计最新分组,
        df[feat + '_mono'] = df[feat + '_min_pnt']
        regroup_mono = bin_badrate(df, col_name=feat + '_mono', target=target)

        # 对特定的特征检查单调性
        if feat in mono_expect.keys():
            # 检查feat是否单调
            is_mono = monotone_badrate(regroup_mono,
                                       shape=mono_expect[feat]['shape'],
                                       u=mono_expect[feat]['u'])
            while not is_mono:
                # 更新regroup_mono以及遍历需要更新的分组序值id
                regroup_mono, id_merge_mono = order_regroup(regroup_mono, feat_order=True)
                # 更新feat_valToBins
                dict_vals_to_bins = merge_neighbour(regroup_mono,
                                                  id_target=id_merge_mono,
                                                  feat_vals_dict=dict_vals_to_bins)
                # 更新df, feat + '_mono'
                df[feat + '_mono'] = df[feat + '_bins'].map(dict_vals_to_bins)
                # 更新regroup_mono
                regroup_mono = bin_badrate(df, col_name=feat + '_mono', target=target)
                # 再次检查单调性, 更新is_mono
                is_mono = monotone_badrate(regroup_mono,
                                           shape=mono_expect[feat]['shape'],
                                           u=mono_expect[feat]['u'])

        # 初始化特殊值的regroup_woe_iv
        regroup_woe_iv = regroup_mono.copy()
        # 对特殊值进行处理
        if feat in special_feat_val.keys() and len(special_feat_val[feat]) > 0:
            # 先更新取值字典,对特殊值进行赋值组序
            # 取值字典加回特殊值, 注意组序为组序0开始计, 之前的组序全部+1
            for sp_val in special_feat_val[feat]:
                for k, v in dict_vals_to_bins.items():
                    dict_vals_to_bins[k] = v + 1
                dict_vals_to_bins[sp_val] = 0
            # 先基于df创建regroup_nor
            df[feat + '_nor'] = df[feat + '_bins'].map(dict_vals_to_bins)
            regroup_nor = bin_badrate(df, col_name=feat + '_nor', target=target)
            # 再基于df_spcial创建regroup_sp
            df_special[feat + '_sp'] = df_special[feat + '_bins'].map(dict_vals_to_bins)
            regroup_sp = bin_badrate(df_special, col_name=feat + '_sp', target=target)
            # 合并,更新regroup_woe_iv
            regroup_woe_iv = regroup_special_merge(regroup_nor, regroup_sp)

        # 计算分组的woe和iv
        # 注意这里用的参数是regroup_woe_iv
        # 这个时候只有num_feat_vals、num_feat_bad、bad_rate三个字段
        regroup_woe_iv, dict_woe, iv = cal_woe_iv(regroup_woe_iv)

        # 插入最终的分组取值, dict_contin_feat_to_bins
        # 对于取值分组,需要分成两种
        # 如果特征属于连续值,则直接合并区间, 然后直接进行插入
        # 如果特征属于类别型(离散无序多), 则需要抓取的是特征取值:分组值
        if feat in discrete_more_feats:
            dict_vals_to_bins = merge_intervals(dict_vals_to_bins)
            regroup_tf[feat + '_interval'] = regroup_tf['bad_rate'].apply(lambda x: value_to_intervals(x,
                                                                                                       dict_vals_to_bins))
            regroup_tf[feat + '_groupno'] = regroup_tf[feat + '_interval'].map(dict_vals_to_bins)
            # 重新刷新&排序dict_valsToBins
            dict_vals_to_bins = regroup_tf[feat + '_groupno'].to_dict()
            dict_vals_to_bins = {k: v for k, v in sorted(dict_vals_to_bins.items(),
                                                       key=lambda x: x[1])}
            dict_contin_feat_to_bins[feat] = dict_vals_to_bins
        else:
            dict_vals_to_bins = merge_intervals(dict_vals_to_bins)
            dict_contin_feat_to_bins[feat] = dict_vals_to_bins

        # 插入iv 和 woe
        dict_contin_iv[feat] = iv
        dict_contin_woe[feat] = dict_woe

    return dict_contin_feat_to_bins, dict_contin_iv, dict_contin_woe

def model_roc_auc(y_true, y_proba, is_plot=False, dict_plot_params=None):
    """
    绘制roc曲线
    输入X_train_woe, 含预测结果和预测概率的训练集
    输入is_plot, 是否绘图

    Parameters:
    ----------
    y_true: list, 实际的y(默认1为正例)

    y_proba: list, 预测的y的probability(概率)

    is_plot: boolean, 是否要绘图

    dict_plot_params: dict, 对plot的图像的参数

    Returns:
    -------
    auc: float, auc值
    """
    # 转换类型
    y_true, y_proba = list(y_true), list(y_proba)

    # 处理参数
    if dict_plot_params is None:
        dict_plot_params = {'fontsize': 15,
                            'figsize': (15, 8),
                            'linewidth': 5}

    if 'fontsize' in dict_plot_params.keys():
        fontsize = dict_plot_params['fontsize']
    else:
        fontsize = 15

    if 'figsize' in dict_plot_params.keys():
        figsize = dict_plot_params['figsize']
    else:
        figsize = (15, 8)

    if 'linewidth' in dict_plot_params.keys():
        linewidth = dict_plot_params['linewidth']
    else:
        linewidth = 5

    # 计算auc
    auc = roc_auc_score(y_true=y_true, y_score=y_proba)

    # 是否绘图
    if is_plot:
        fontdict = {'fontsize': fontsize}

        # 绘制roc曲线
        fpr, tpr, thresholds = roc_curve(y_true=y_true,
                                         y_score=y_proba,
                                         pos_label=1)
        fig, ax = plt.subplots(figsize=figsize)

        # 绘制roc
        ax.plot(fpr, tpr, label='ROC Curve(AUC=%.2F)' % auc, linewidth=linewidth)

        # 绘制(0,0) (1,1)直线
        ax.plot([0, 1], [0, 1], linestyle='--', c='r', linewidth=linewidth)

        ax.set_title('Receiver Operating Characteristic', fontdict=fontdict)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel('False Positive Rate', fontdict=fontdict)
        ax.set_ylabel('True Positive Rate', fontdict=fontdict)
        ax.legend(loc='best', fontsize='x-large')
    # 返回结果
    return auc


def model_ks(y_true, y_pred, y_proba, is_plot=True, dict_plot_params=None):
    """
    计算模型的ks(Kolmogorov-Smirnov)

    ks: <0 模型错误, 0~0.2 模型较差, 0.2~0.3 模型可用, >0.3 模型预测性较好

    ks计算公式为 ks = max(cum(Bi)/ Bad(total) - cum(Gi) / Good(total))

    Parameters:
    ----------
    y_true: list, 真实的y

    y_pred: list,预测的y

    y_proba: list, 预测的y的概率

    dict_plot_params: dict, 绘图的参数

    Returns:
    -------
    ks: 模型的ks值


    """
    # 转换类型
    y_true, y_pred, y_proba = list(y_true), list(y_pred), list(y_proba)

    # 处理参数
    if dict_plot_params is None:
        dict_plot_params = {'fontsize': 15,
                            'figsize': (15, 8),
                            'linewidth': 2}

    if 'fontsize' in dict_plot_params.keys():
        fontsize = dict_plot_params['fontsize']
    else:
        fontsize = 15

    if 'figsize' in dict_plot_params.keys():
        figsize = dict_plot_params['figsize']
    else:
        figsize = (15, 8)

    if 'linewidth' in dict_plot_params.keys():
        linewidth = dict_plot_params['linewidth']
    else:
        linewidth = 2

    # 创建df, 计算总样本数、bad样本数、good样本数
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'y_proba': y_proba})
    cnt_total = df['y_true'].shape[0]
    cnt_bad = df.loc[df['y_true'] == 1].shape[0]
    cnt_good = cnt_total - cnt_bad

    # 对df基于proba进行降序排序
    df = df.sort_values(by='y_proba', ascending=False)

    # 计算总样本(排序后的)下的样本数占比
    df['cum_pnt_total'] = [i / cnt_total for i in list(range(1, cnt_total + 1))]

    # 计算好样本(排序后的)下的样本数占比
    df['cum_pnt_good'] = (df['y_true'] == 0).cumsum() / cnt_good

    # 计算坏样本(排序后的)下的样本数占比
    df['cum_pnt_bad'] = (df['y_true'] == 1).cumsum() / cnt_bad

    # 计算差值的绝对值
    df['cum_diff_value'] = abs(df['cum_pnt_bad'] - df['cum_pnt_good'])

    # 计算ks值
    ks = df['cum_diff_value'].max()

    # 绘制ks曲线
    if is_plot:
        ## 取得ks下的index, 计算在该idx下的cum_pnt_total
        idxmax = df['cum_diff_value'].idxmax()
        x_ks = df.loc[idxmax, 'cum_pnt_total']

        ## 绘图
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(df['cum_pnt_total'], df['cum_pnt_bad'], linewidth=linewidth, label='Bad')
        ax.plot(df['cum_pnt_total'], df['cum_pnt_good'], linewidth=linewidth, label='Good')
        ax.plot(df['cum_pnt_total'], df['cum_diff_value'], linewidth=linewidth, label='K-S Curve')
        ax.plot([x_ks], [ks], 'o')
        ax.annotate('K-S Statistic(%.2f)' % ks, xy=(x_ks, ks),
                    xytext=(x_ks, ks / 1.5),
                    fontsize=fontsize,
                    arrowprops=dict(arrowstyle='fancy', facecolor='red'))

        ax.grid(linewidth=0.4, linestyle='--')
        ax.legend(loc='best', fontsize='large')
        ax.set_title('K-S Curve', fontsize=fontsize)

        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))

    return ks