from setuptools import find_packages, setup

with open('README.md', 'r', encoding='utf-8') as file:
    long_description = file.read()

setup(
    name='EWarningSignals',
    packages=find_packages(include=['earlywarningsignals*']),
    include_package_data=True,
    package_data={
        '': ['*.txt', '*.csv'],
        'earlywarningsignals/data': ['*.txt', '*.csv']
    },
    version='0.1.0',
    description='Generation of early warning signals to detect the tipping point before a pandemic outbreak.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Angel Fragua',
    license='MIT',
    install_requires=[
        'geopandas~=0.10.2;platform_system=="Linux"',
        'pandas~=1.4.2',
        'numpy~=1.22.3',
        'scipy~=1.8.0',
        'tqdm~=4.64.0',
        'networkx~=2.8',
        'GraphRicciCurvature~=0.5.3',
        'matplotlib~=3.5.1'
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)
