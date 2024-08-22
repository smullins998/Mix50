from setuptools import setup
import sys, os

# with open('README.md', encoding='utf8') as f:
#     long_description = f.read()


setup(name='Mix50',
      version='0.1',
      description='Fast and simple DJ and audio effects in Python with Librosa',
     # long_description=long_description,
      long_description_content_type="text/markdown",
      classifiers=[
          'Development Status :: 1 - Beta',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.11',
          'Topic :: Multimedia :: Sound/Audio :: Analysis'
      ],
      keywords='Fast and simple DJ and audio effects in Python with Librosa',
      url='https://github.com/smullins998/Mix50',
      author='Sean Mullins',
      author_email='smullins998@gmail.com',
      license='MIT',
      packages=['mix50'],
      install_requires=[
    'librosa==0.8.1',
    'numpy==1.24.0',
    'pandas==1.5.3',
    'SoundFile==0.10.2',
    'pyrubberband', 
    'scipy==1.9.3',
    'pydub==0.25.1',
    'scikit-learn',  
    'ipython==8.26.0',
    'essentia', 
    'matplotlib==3.6.3',
    'sounddevice', 
      ],
      include_package_data=True,
      zip_safe=False)