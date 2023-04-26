#Copyright 2021 Ryan Tran, Victor Zhu, Arun Kumar
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

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
    install_requires=['pandas','numpy<1.24', 'nltk', 'joblib', 'scikit-learn==1.2.2'],
    python_requires=">=3.6",
    version='0.0.7',
    license='MIT',
)
