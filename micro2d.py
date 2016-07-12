# -*- coding: utf-8 -*-
'''
Created on 3 juil. 2015

.. py:module:: micro class

@author: Thomas Chauve
@contact: thomas.chauve@lgge.obs.ujf-grenoble.fr
@license: CC-BY-CC
'''

import image2d as im2d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from skimage import morphology
import scipy
from IPython.utils.py3compat import xrange

class micro2d(im2d.image2d):
    '''
    micro2d is a class which herit from image2d but it is restricted to microstructure object (background 0, boundary 1)
    '''
    pass

    def __init__(self,field, resolution):
        '''
        Constructor : field is a matrix and resolution is the step size in mm
        
        :param field: tabular of scalar data
        :type field: array
        :param resolution: step size resolution (millimeters) 
        :type resolution: float
        '''
        
        if np.size(np.where((field!=0) & (field!=1)))==0:
            self.field=field
            self.res=resolution
        else:
            print('error not only 0 and 1 in field')         
        return
    
    
    def grain_label(self):
        '''
        Label area in a black and white picture
        
        .. note:: black color for the background and white color for the boundary
        '''
        # function which label a microstructure skeleton in one number per grain
        new_img=self.field
        res=self.res
        new_grain = morphology.label(new_img, neighbors=4, background=1)
        grains=im2d.image2d(new_grain,res)
        return grains
    
    def plotBoundary(self,dilatation=0):
        '''
        Add boundary to the figure
        
        :param dilatation: number of iteration for the dilation of 1 value - used this to make larger the boundaries (default 2)
        :type dilatation: int
        :Exemple:
            >>> data.phi1.plot()
            >>> data.micro.plotBoundary(dilatation=10)
            >>> plt.show()
        
        .. note:: plot only the value of the pixel equal at 1
        '''
        # extract microstructure matrix
        micro=self.field
        # make the dilation the number of time wanted
        if dilatation>0:
            micro=scipy.ndimage.binary_dilation(micro, iterations=dilatation)
        # create a mask with the 0 value
        micro = np.ma.masked_where(micro == 0, micro)
        # size of the image2d
        ss=np.shape(self.field)    
        # plot the boundaries
        plt.imshow(micro, extent=(0,ss[1]*self.res,0,ss[0]*self.res), interpolation='none',cmap=cm.gist_gray)
        
        return
    
    def area(self):
        '''
        Compute the grain area for each grain
        
        :return: g_arean array of scalar of grain area in mm^2
        :rtype: g_area np.array
        '''
        
        g_map=self.grain_label()
        
        g_area=np.zeros(np.max(g_map.field))
        
        for i in list(xrange(np.size(g_area))):
            g_area[i]=np.size(np.where(g_map.field==i))*g_map.res**2.
            
        return g_area