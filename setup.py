# -*- coding: utf-8 -*-
from setuptools import setup

LONGDOC = """
我家还蛮大的, 欢迎你们来我家van.

https://github.com/skykiseki

new-words-detection
====

"新词发现", 
目前仅有基于苏神（苏剑林）文章写的左右熵以及互信息方法进行词库建设，其他的算法后面有空会继续更新


具体链接参考
[《新词发现的信息熵方法与实现》](https://spaces.ac.cn/archives/3491)



完整文档见 ``README.md``

GitHub: https://github.com/skykiseki/NewWordDectection
"""

setup(name='new-words-detection',
      version='1.0.3',
      description='Chinese Words Segmentation Utilities',
      long_description=LONGDOC,
      long_description_content_type="text/markdown",
      author='Wei, Zhihui',
      author_email='evelinesdd@qq.com',
      url='https://github.com/skykiseki/NewWordDectection',
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
        'numpy'
      ],
      keywords='NLP,Chinese word detection,Chinese word segementation',
      packages=['new_words_detection'],
      package_dir={'NewWordDetection':'new_words_detection'}
)