Fisrt step
==========

You can also have look to the jupyter notebook Exemple/ExempleAITA.ipynb

Loading data
************

Loading data from G50 analyser after you convert binary file into ASCII file with 5 columuns. The 'micro_test.bmp' file should be a black and white image with the grains boundaries in white.

.. code:: python
    
    import AITATool.loadData_aita as lda
    # The inputs come from G50 analyseur
    adr_data='orientation_test.dat'
    adr_micro='micro_test.bmp'
    

    data=lda.aita5col(adr_data,adr_micro)


Basic treatment
***************

You can crop the data.

.. code:: python
    
    data.crop()

You can filter the bad value for indexed orientation. Usualy 75 is a good value.

.. code:: python
    
    data.filter(75)


Plot orientation map
********************

You can plot Euler angle map.

.. code:: python
    
    import matplotlib.pyplot as plt
    plt.figure()
    data.phi1.plot()
    plt.title('phi1 Euler angle')

    plt.figure()
    data.phi1.plot()
    plt.title('phi Euler angle')
    plt.plot()

You can plot grains if you have include a microstructure when you load the data.

.. code:: python
    
    plt.figure()
    data.grains.plot()
    plt.title('Grains')
    plt.plot()

You can plot a colormap associated with a colorwheel

.. code:: python
    
    plt.figure()
    data.plot()
    plt.plot()

The colorwheel

.. code:: python
    
    import IGETools.aita as aita
    lut2d=aita.lut()
    plt.imshow(lut)
    plt.plot()

Plot pole figure
****************

Various options can be used to plot pole figure. Projection plane, add circle at different angle. Plot eigenvalue on the pole figure. Here we just show the basic figure, which is ready for puplication.


.. code:: python
	
    data.plotpdf(contourf=True,angle=0)
    plt.plot()

Others
******

There is more function. To have a look into it look at the function overview.
