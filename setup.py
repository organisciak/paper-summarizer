from distutils.util import convert_path
from setuptools import setup, find_packages

main_ns = {}
ver_path = convert_path('mymodule/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

setup(
    name='paper_summarizer',
    version=main_ns['__version__'],
    url='https://github.com/organisciak/paper_summarizer',
    author=main_ns['__author__'],
    author_email=main_ns['__email__'],
    description='A library to summarize research papers using OpenAI language models',
    packages=find_packages(),
    install_requires=[
        'openai',
        'tiktoken',
        'PyPDF2',
        'tqdm',
        'yaml_sync',
        'IPython'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        # TODO decide on licence. maybe.. 'License :: OSI Approved :: MIT License', 
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
