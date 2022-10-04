import setuptools

from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / 'README.md').read_text()

setuptools.setup(
    name='slater_dl_wrapper',
    version='0.0.4',
    author='Ryan Slater',
    author_email='ryan.j.slater.2@gmail.com',
    description='A wrapper for PyTorch to make it easier to train and evaluate models',
    packages=['slater_dl_wrapper'],
    url='https://github.com/CodingPenguin1/slater-dl-wrapper',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=['matplotlib>=3.6.0', 'numpy>=1.23.3', 'torch>=1.12.1']
)
