from setuptools import setup, find_packages

setup(name='slime',
      version='0.1',
      description='sLIME',
      url='https://github.com/pleask/slime',
      author='Patrick Leask',
      author_email='patrickaaleask@gmail.com',
      license='MIT',
      packages=find_packages(),
      python_requires='>=3.5',
      install_requires=[
          'numpy',
          'lime',
          'tqdm',
          'sklearn'
      ],
      include_package_data=True,
      zip_safe=False)