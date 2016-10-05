# -*- coding: utf-8 -*-
'''
Created on 22 juil. 2015

.. py:module:: mask2d class

@author: Thomas Chauve
@contact: thomas.chauve@lgge.obs.ujf-grenoble.fr
@license: CC-BY-CC
'''

import image2d as im2d
import numpy as np
from skimage import io

class mask2d(im2d.image2d):
    '''
    mask2d is a class which herit from image2d but it is restricted to microstructure object (background NaN, selected area 1)
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
        
        if np.size(np.where((~np.isnan(field)) & (field!=1)))==0:
            self.field=field
            self.res=resolution
        else:
            print('error not only NaN and 1 in field')         
        return
    
#######################################################
################# Function ############################
#######################################################

def load_mask(adr_bmp,res=1):
    '''
    Create a mask from a black and white bmp image where whith correspond to the selected area
    
    :param adr_bmp: path of the mask bmp file
    :type adr_bmp: str
    :param res: resolution of the picture mm (default 1)
    :type res: float
    :return mask:
    :rtype mask: mask2d 
    '''
    # Load bmp file
    image_bmp = io.imread(adr_bmp)
    
    tmp=image_bmp[:,:,0]
    # create nan matrix
    ss=np.shape(tmp)
    mask=np.zeros(ss)
    mask[:]=np.NaN
    # replace by one the white area
    id=np.where(tmp!=0)
    mask[id]=1
    
    return mask2d(mask,res)

def complementary(self):
    '''
    Return complementary mask
    '''
    m=np.ones(np.shape(self.field))
    m[np.where(self.field==1)]=np.NaN

    return mask2d(m,self.res)
