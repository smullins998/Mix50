from setuptools import setup

# with open('README.md', encoding='utf8') as f:
#     long_description = f.read()

setup(
    name='Mix50',
    version='0.4',
    description='Fast and simple DJ and audio effects in Python with Librosa',
    # long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        'Development Status :: 4 - Beta',  # Updated classifier
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.11',
        'Topic :: Multimedia :: Sound/Audio :: Analysis'
    ],
    url='https://github.com/smullins998/Mix50',
    author='Sean Mullins',
    author_email='smullins998@gmail.com',
    license='MIT',
    packages=['Mix50'],
    install_requires=[
        'librosa>=0.10.2.post1',
        'numpy<2',
        'pandas',
        'SoundFile',
        'pyrubberband',
        'scipy',
        'pydub',
        'scikit-learn',
        'ipython',
        'essentia',
        'matplotlib',
        'sounddevice',
    ],
    keywords=['python', 'audio', 'DSP', 'DJ', 'audio analysis'],
    include_package_data=True,
    zip_safe=False
)
