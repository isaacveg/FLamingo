# set up FLamingo
from setuptools import setup, find_packages

setup(
    name='FLamingo',
    version='0.0.1-beta',
    author='Isaacveg',
    author_email='isaaczhu@mail.ustc.edu.cn',
    description='A brief description of the package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/isaacveg/FLamingo',
    packages=find_packages(include=['FLamingo','FLamingo.*']),
    package_data={'FLamingo': ['templates/*.txt']},
    install_requires=[
        # 依赖列表
        'mpi4py>=3.1',
        'numpy>=1',
        'Pillow>=9',
        'PyYAML>=6.0',
        'scikit_learn>=1.3.0',
        'spacy>=3.6',
        'torch>=2.0',
        'torchvision>=0.15',
        'tqdm>=4'
    ],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ],
    python_requires='>=3.8',
)
