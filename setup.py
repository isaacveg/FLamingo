# set up FLamingo
from setuptools import setup, find_packages

setup(
    name='FLamingo',
    version='0.1.0',
    author='Isaacveg',
    author_email='isaaczhu@mail.ustc.edu.cn',
    description='A brief description of the package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/isaacveg/FLamingo',
    packages=find_packages(),
    install_requires=[
        # 依赖列表
        'mpi4py==3.1.4',
        'numpy==1.23.5',
        'Pillow==9.3.0',
        'PyYAML==6.0',
        'scikit_learn==1.3.0',
        'spacy==3.6.1',
        'torch==2.0.1',
        'torchvision==0.15.2',
        'tqdm==4.65.0'
    ],
    classifiers=[
        # 开发状态
        'Development Status :: 3 - Alpha',
        # 目标用户
        'Intended Audience :: Developers',
        # 许可证（与LICENSE文件匹配）
        'License :: OSI Approved :: MIT License',
        # 兼容的Python版本
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.8',
)
