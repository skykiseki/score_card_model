import pandas as pd
import numpy as np

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