import setuptools

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
     name='bnn',  
     version='0.1',
     author="Marco Fronzi",
     author_email="marco.fronzi@gmail.com",  
     description="A Python package for 2-dimensional heterostracture properties prediction using Bayesian Neural Networks",  
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/fronzi/projBNN", 
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
     install_requires=[
         'numpy==1.17.2',
         'matplotlib==3.1.1',
         'scikit-learn==0.21.3',
         'scikit-image==0.15.0',
         'pandas==0.25.1',
         'seaborn==0.9.0',
         'Keras-Applications==1.0.8',
         'Keras-Preprocessing==1.1.0',
         'tensorflow==1.14.0',
     ],
     python_requires='=3.7', 
)
