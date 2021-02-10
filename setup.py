import setuptools

setuptools.setup(
    name='AITAToolbox',
    version='2.0.1',
    description='Tools for AITA',
      
    url='https://github.com/ThomasChauve/aita',
    author='Thomas Chauve',
    author_email='thomas.chauve@univ-grenoble-alpes.fr',
    license='GPL-3.0',
    packages=setuptools.find_packages(exclude=['docs','Exemple']),

    
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3 :: Only'
    ],
      
    project_urls={
        'Source': 'https://github.com/ThomasChauve/aita',
        'Tracker': 'https://github.com/ThomasChauve/aita/issues',
        'Documentation': 'https://thomaschauve.github.io/aita/build/html/index.html',
    },
      
    python_requires='>=3.9',
    
    install_requires=[
        'Shapely',
        'pygmsh',
        'numpy',
        'matplotlib',
        'scikit-image',
        'scikit-learn',
        'tqdm',
        'scipy',
        'vtk',
        'Pillow',
        'gmsh'
    ],
     
)
