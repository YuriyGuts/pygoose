from setuptools import setup, find_packages


setup(
    name='pygoose',
    version='0.1.1',
    description='Utility tool belt for Kaggle competitions and other Data Science experiments',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Utilities',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
    ],
    url='http://github.com/YuriyGuts/pygoose',
    author='Yuriy Guts',
    author_email='yuriy.guts@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'joblib',
        'tqdm',
        'numpy',
        'pandas',
        'matplotlib',
        'keras',
        'seaborn',
    ],
    include_package_data=True,
    zip_safe=False,
    setup_requires=[
        'pytest-runner'
    ],

    tests_require=[
        'pytest'
    ],
)
