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
        'librosa<0.7.0',
        'torch<1.1.0',
    ],
    package_data={
        '': ['questions_jp.hed'],
    }
)
