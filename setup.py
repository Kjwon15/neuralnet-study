from setuptools import setup, find_packages

setup(
    name='nn',
    packages=find_packages(exclude=['tests']),
    test_require=[
        'pytest ~= 3.3.0',
    ]
)
