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
pip install score_card_model
```

使用方法
====


初始化:
----

代码示例:

```python
## 加载
import pandas as pd
from score_card_model.ScoreCardModel import ScoreCardModel

# 读取数据
df_data = pd.read_excel("./test.xlsx")

# 创建和初始化类, 前提数据已经预处理完毕, 没有缺失值
scm_obj = ScoreCardModel(df=df_data, target='loan_status')

# 

```
