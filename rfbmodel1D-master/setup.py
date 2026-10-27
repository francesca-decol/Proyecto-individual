from pathlib import Path
from setuptools import setup, find_packages
long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")
setup(
    name='rfbmodel1D',
    version='0.1',
    packages=find_packages(include=['rfbmodel1D', 'rfbmodel1D.*']),
    description='1-D transient DAE model of an H2-V regenerative redox flow battery cell',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Joaquín Barros',
    author_email='jbarrosvan@uc.cl',
    url='TODO: add the GitHub repo URL',
    license='TODO: match the LICENSE file',
    python_requires='>=3.10',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'pandas',
        'scikit-learn',
        'solve_dae>=0.2.3',
    ],
)