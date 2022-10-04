import setuptools

setuptools.setup(
    name='slater_dl_wrapper',
    version='0.0.3',
    author='Ryan Slater',
    author_email='ryan.j.slater.2@gmail.com',
    description='A wrapper for PyTorch to make it easier to train and evaluate models',
    packages=['slater_dl_wrapper'],
    url='https://github.com/CodingPenguin1/slater-dl-wrapper',
    install_requires=['matplotlib>=3.6.0', 'numpy>=1.23.3', 'torch>=1.12.1']
)
