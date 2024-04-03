from setuptools import setup, find_packages

setup(
    name='textraer',
    version='0.1.0',
    packages=find_packages(),
    description='Text processor with classification using DataBERT.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'torch',
        'transformers',
        'numpy',
        'scikit-learn',
        'huggingface_hub',
        'nltk',
        'PyPDF2',
        'requests'
    ],
    python_requires='>=3.6',
    author='James Liounis',
    author_email='liounisjames@gmail.com',
    url='https://github.com/avsolatorio/data-use/tree/main/scripts/DataBERT',
    license='MIT',
)

