from setuptools import setup, find_packages

setup(
    name='hihobot-synthesis',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/Hiroshiba/hihobot-synthesis',
    author='Kazuyuki Hiroshiba',
    author_email='hihokaruta@gmail.com',
    install_requires=[
        'nnmnkwii',
        'pyworld',
        'librosa',
        'torch',
    ],
    package_data={
        '': ['questions_jp.hed'],
    }
)
