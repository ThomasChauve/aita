# -*- coding: utf-8 -*-
'''
Created on 3 juil. 2015

.. py:module:: image2d class

image2d is a class used to manipulate image under matrix shape and to do the analyses on the picture

.. note:: It has been build to manipulate both aita data and dic data
.. warning:: As all function are applicable to aita and dic data, please be careful of the meaning of what you are doing depending of the input data used !  

@author: Thomas Chauve
@contact: thomas.chauve@lgge.obs.ujf-grenoble.fr
@license: CC-BY-CC
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage import morphology
import scipy
import symetricTensorMap
import TensorMap
import pylab

class image2d(object):
    '''
    image2d is a class for map of scalar data
    '''
    pass

    def __init__(self, field, resolution):
        '''
        Constructor : field is a matrix and resolution is the step size in mm
        
        :param field: tabular of scalar data
        :type field: array
        :param resolution: step size resolution (millimeters) 
        :type resolution: float
        '''
        
        self.field=field
        self.res=resolution
        
    def plot(self,vmin=np.NaN,vmax=np.NaN,colorbarcenter=False,colorbar=cm.jet):
        '''
        plot the image2d
        
        :param vmin: minimum value for the colorbar
        :type vmin: float
        :param vmax: maximun value for the colorbar
        :type vmax: float
        :param colorbarcenter: do you want center the colorbar around 0
        :type colorbarcenter: bool
        :param colorbar: colorbar from matplotlib.cm
        
        .. note:: colorbar : cm.jet for eqstrain-stress
        '''
        if np.isnan(vmin):
            vmin=np.nanmin(self.field)
            
        if np.isnan(vmax):
            vmax=np.nanmax(self.field)
        
        # size of the image2d
        ss=np.shape(self.field)
        # create image
        img=plt.imshow(self.field,aspect='equal',extent=(0,ss[1]*self.res,0,ss[0]*self.res),cmap=colorbar,vmin=vmin,vmax=vmax)
        
        if colorbarcenter:
            zcolor=np.max(np.max(np.abs(self.field)))
            plt.clim(-zcolor, zcolor)
        
        # set up colorbar
        plt.colorbar(img,orientation='vertical',aspect=4)
        
    def extract_data(self,pos=[]):
        '''
        Extract the value at the position 'pos' or where you clic
        
        :param pos: array [x,y] position of the data, if pos==[], clic to select the pixel
        :type pos: array
        '''
        
        if pos==[]:
            plt.imshow(self.field,aspect='equal')
            plt.waitforbuttonpress()
            print('select the pixel :')
            #grain wanted for the plot
            id=np.int32(np.array(pylab.ginput(1)))
        else:
            id=pos

        plt.close()    
        return self.field[id[0,1],id[0,0]],id
    
    def triple_junction(self):
        '''
        Localized the triple junction
        '''
        ss=np.shape(self.field)
        triple=[]
        pos=[]
        for i in list(xrange(ss[0]-2)):
            for j in list(xrange(ss[1]-2)):
                sub=self.field[i:i+2,j:j+2]
                id=np.where(sub[:]==sub[0,0])
                i1=len((id[0]))
                if (i1<(3)):
                    id=np.where(sub[:]==sub[0,1])
                    i2=len((id[0]))
                    if (i2<(3)):
                        id=np.where(sub[:]==sub[1,1])
                        i3=len((id[0]))
                        if ((i3==1 or i2==1 or i1==1) and i3<3):
                            triple.append(sub)
                            pos.append([i,j])
                            
        c=np.array(pos)
        z=np.arange(len(c[:,0]))                    
        plt.imshow(self.field)
        plt.plot(c[:,1],c[:,0],'+')
        
        for label, x, y in zip(z, c[:, 1], c[:, 0]):
            plt.annotate(
            label, 
            xy = (x, y), xytext = (-20, 20),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

        return triple,c
        
    def grain_label(self):
        '''
        Label area in a black and white picture
        
        .. note:: black color for the background and white color for the boundary
        '''
        # function which label a microstructure skeleton in one number per grain
        new_img=self.field
        res=self.res
        new_grain = morphology.label(new_img, neighbors=4, background=1)
        grains=image2d(new_grain,res)
        return grains
    
    def imresize(self,res):
        '''
        Resize the image with nearest interpolation to have a pixel of the length given in res
        
        :param res: new resolution map wanted (millimeters)
        :type res: float
        '''
        # Fraction of current size
        zoom=float(self.res/res)
        self.res=res
        self.field=scipy.ndimage.interpolation.zoom(self.field,zoom,order=0,mode='nearest')
        
    def diff(self,axis):
        '''
        Derive the image along the axis define
        
        :param axis: 'x' to do du/dx or 'y' to do du/dy
        :type axis: str
        :return: differential of the map in respect to the 'axis' wanted
        :rtype: image2d
        
        .. warning:: 'x' axis is left to right and 'y' is bottom to top direction        
        '''
        if (axis=='x'):
            dfield=np.diff(self.field,axis=1)/self.res
            nfield=dfield[1:,:] # remove one line along y direction to have same size for both diff
        elif (axis=='y'):
            dfield=-np.diff(self.field,axis=0)/self.res # the - is from the convention y axis from bottom to top
            nfield=dfield[:,1:]
        else:
            print('axis not good')
            
        dmap=image2d(nfield,self.res)
        
        return dmap
    
    def __add__(self, other):
        '''
        Sum of 2 maps
        '''
        if (type(other) is image2d):
            return image2d(self.field+other.field,self.res)
        elif (type(other) is float):
            return image2d(self.field+other,self.res)
            
    def __sub__(self, other):
        '''
        Subtract of 2 maps
        '''
        if (type(other) is image2d):
            return image2d(self.field-other.field,self.res)
        elif (type(other) is float):
            return image2d(self.field-other,self.res)
    
    def __mul__(self,other):
        '''
        Multiply case by case
        '''
        if (type(other) is image2d):
            return image2d(self.field*other.field,self.res)
        if (type(other) is symetricTensorMap.symetricTensorMap):
            return other*self
        if (type(other) is TensorMap.TensorMap):
            return other*self
        if (type(other) is float):
            return image2d(self.field*other,self.res)
        
    
    def __div__(self,other):
        'Divide self by other case by case'
        if (type(other) is image2d):
            return image2d(self.field*1/other.field,self.res)
        elif (type(other) is float):
            return self*1/other
    
    def pow(self, nb):
        '''
        map power nb
        
        :param nb:
        :type nb: float    
        '''
        
        return image2d(np.power(self.field,nb),self.res)
    
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
        
    def mask_build(self,r=0,grainId=[],pos_center=0):
        '''
        Create a mask map with NaN value where you don't want data and one were you want
        The default mask is a circle of center you choose and radius you define. 
        
        :param r: radius of the circle, If r=0 you clic of each of grain you want
        :type r: float
        :param grainId: You select the grainId you want in an array
        :type: array
        :return: mask
        :rtype: image2d
        
        .. note:: if you want applied a mask one your data just do data*mask where data is an image2d object
        '''
        # size of the figure
        ss=np.shape(self.field)
        mask_map=np.empty(ss, float)
        mask_map.fill(np.nan)
        if (r==0 and len(grainId)==0) or r==0:
            if len(grainId)==0:
                plt.imshow(self.field,aspect='equal')
                plt.waitforbuttonpress()
                print('Select grains :')
                print('midle mouse clic when you are finish')
                xp=np.int32(np.array(plt.ginput(0)))
                plt.close('all')
            
                gId=self.field[xp[:,1],xp[:,0]]
            else:
                gId=grainId
                xp=0
                
            for i in range(len(gId)):
                idx=np.where(self.field==gId[i])
                mask_map[idx]=1
        else:
            if np.size(pos_center)==1:
                self.plot()
                plt.waitforbuttonpress()
                print('clic to the center of the circle')
                xp=np.int32(np.array(plt.ginput(1))/self.res)
            else:
                xp=pos_center
            idx=[]
            for i in np.int32(np.arange(2*r/self.res+1)+xp[0][0]-r/self.res):
                for j in np.int32(np.arange(2*r/self.res+1)+xp[0][1]-r/self.res):
                    if (((i-xp[0][0])**2+(j-xp[0][1])**2)**(0.5)<r/self.res):
                        idx.append([i,j])
            idx2=np.array(idx)
            y=ss[0]-idx2[:,1]
            x=idx2[:,0]
            v=(y>=0)*(y<ss[0])*(x>=0)*(x<ss[1])
            mask_map[[y[v],x[v]]]=1
            plt.close('all')
            
            pc=float(sum(v))/float(len(y))
            if pc<1:
                print('WARNING : area is close to the border only '+str(pc)+'% of the initial area as been selected')
        
        return image2d(mask_map,self.res),xp
    
    
    def skeleton(self):
        '''
            Skeletonized a label map build by grain_label
        '''
        
        # derived the image 
        a=self.diff('x')
        b=self.diff('y')
        # Normelized to one
        a=a/a
        b=b/b
        # Replace NaN by 0
        a.field[np.isnan(a.field)]=0
        b.field[np.isnan(b.field)]=0
        # Build the skeleton
        skel=a+b
        id=np.where(skel.field>0)
        skel.field[id]=1
        
        return skel
        
