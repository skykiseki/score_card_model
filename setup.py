# -*- coding: utf-8 -*-
from setuptools import setup

LONGDOC = """
我家还蛮大的, 欢迎你们来我家van.

https://github.com/skykiseki

score_card_model
====

"风险评分卡模型开发" 
基于最早的FICO风险评分卡逻辑进行优化


完整文档见 ``README.md``

GitHub: https://github.com/skykiseki/score_card_model
"""

setup(name='score_card_model',
      version='1.0.0',
      description='Risk Score Card Model',
      long_description=LONGDOC,
      long_description_content_type="text/markdown",
      author='Wei, Zhihui',
      author_email='evelinesdd@qq.com',
      url='https://github.com/skykiseki/score_card_model',
      license="MIT",
      classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Natural Language :: Chinese (Simplified)',
        'Natural Language :: Chinese (Traditional)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Text Processing',
        'Topic :: Text Processing :: Indexing',
        'Topic :: Text Processing :: Linguistic',
      ],
      python_requires='>=3.6',
      install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'statsmodels',
        'matplotlib',
        'seaborn'
      ],
      keywords='Risk Score Card',
      packages=['score_card_model'],
      package_dir={'score_card_model':'score_card_model'}
)