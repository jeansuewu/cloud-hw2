from setuptools import setup, find_packages

setup(
    name='cloud-classifier',
    version='0.1',
    author='Suzie Wu',
    author_email='suzie221@u.northwestern.edu',
    description='A simple Cloud Classifer',
    long_description='A simple Cloud classifier',
    long_description_content_type='text/markdown',
    # url='https://github.com/pujari/iris-classifier',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
    install_requires=[
        'numpy',
        'scikit-learn',
        'tensorflow',
        'keras',
    ],
)
