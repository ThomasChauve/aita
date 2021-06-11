.. AITAToolbox documentation master file, created by
   sphinx-quickstart on Tue Nov 20 10:23:06 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. warning:: This tools is not maintened anymore. For a similar tools please go to AITA-book : https://thomaschauve.github.io/AITA-book/docs/intro.html

Welcome to AITA Toolbox's documentation!
========================================

AITA Toolbox is a python tools created to work on orientation map such as the one obtained using optical measurement (birefringence).

It is an open source code under GPL-v3. 

There is no guarantee if you are using it. 

It has been tested with python 3.8


.. toctree::
   :maxdepth: 1
   :caption: Documentation:


Installation
============

From pip
********

python 3.8 is require

.. code:: bash

    pip install AITAToolbox


From repository
***************

You can clone the repository :
  
.. code:: bash

    git clone https://github.com/ThomasChauve/aita
    cd aita/


Create a new environement envAITA by default
--------------------------------------------

You cand directly install the new conda environement

.. code:: bash

    conda env create --name envAITA -f environment.yml


Using AITAToolbox
=================

You need to run python in the good environement using :

.. code:: bash
    
    conda activate envAITA

Then you will find all the package in python using

.. code:: python

    import AITAToolbox
    
Devellopement
=============

If you want to add new function in the Toolbox you run in aita folder

.. code:: bash
    
    pip install -e .


Uninstall
=========

.. code:: bash
    
    pip uninstall AITAToolbox


.. toctree::
    :maxdepth: 1
    :numbered:
    :caption: Documentation

    Documentation/Documentation
    AITA/function

.. toctree::
    :maxdepth: 1
    :numbered:
    :caption: CLASS

    CLASS/image2d
    CLASS/setvector3d


Contact
=======
:Author: Thomas Chauve
:Contact: thomas.chauve@univ-grenoble-alpes.fr

:organization: IGE
:status: This is a "work in progress"
:version: 2.1



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
