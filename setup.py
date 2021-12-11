from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='sortinghatinf',
    url='https://github.com/bobotran/SortingHatLib',
    author='Vraj Shah',
    author_email='pvn251@gmail.com',
    description='A library that executes SortingHat feature type inference on Pandas dataframes',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['sortinghatinf'],
    package_data={"": ["resources/*"]},
    install_requires=['pandas','numpy', 'nltk', 'joblib', 'scikit-learn==0.22.2.post1'],
    python_requires=">=3.6",
    version='0.0.6',
    license='MIT',
)
