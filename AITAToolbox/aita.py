# -*- coding: utf-8 -*-
'''
.. py:module:: AITA G50

Created on 3 juil. 2015
Toolbox for data obtained using G50 Automatique Ice Texture Analyser (AITA) provide by :
Russell-Head, D.S., Wilson, C., 2001. Automated fabric analyser system for quartz and ice. J. Glaciol. 24, 117–130

@author: Thomas Chauve
@contact: thomas.chauve@univ-grenoble-alpes.fr
@license: CC-BY-CC
'''

import pygmsh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import pylab
from skimage import io
import skimage.morphology
import skimage.measure
from tqdm import tqdm
import datetime
import random
import scipy
import colorsys

import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

import AITAToolbox.image2d as im2d
import AITAToolbox.setvector3d as vec3d

class aita(object):
    '''
	.. py:class:: aita
	
	"aita" is a python class to analyse output from G50-AITA analyser.
	It provide an environnement to plot data, to create inpur for CraFT code,...    
    '''
    pass
    
    def __init__(self,phi1_field,phi_field,qua_field,micro_field,resolution=1):
        '''        
        :param phi1_field: Euler angle phi1 map
        :param phi_field: Euler angle phi map
        :param qua_field: quality facteur map
        :param resolution: spatial step size (mm); default = 1 mm
        :param micro_field: microstructure (0 background, 1 grain boundary)
        
        :type phi1_field np.array
        :type phi_field np.array
        :type qua_field: np.array
        :type resolution: float
        :type micro_adress: np.array
        
        :return: aita object output
        :rtype: aita
                             
        .. note:: Bunge Euler Angle convention is used (phi1,phi,phi2) ,phi2 is not compute as during optical measurement phi2 is not know.
        '''
        
        # create image object from data
        self.phi1=im2d.image2d(phi1_field,resolution)
        self.phi=im2d.image2d(phi_field,resolution)
        self.qua=im2d.image2d(qua_field,resolution)
        
        # create microstructure
        self.micro=im2d.micro2d(micro_field,resolution)
        self.grains=self.micro.grain_label()
        
        # replace grains boundary with NaN number
        self.grains.field=np.array(self.grains.field,float)
        idx=np.where(self.micro.field==1)
        self.grains.field[idx]=np.nan

        print("Sucessfull aita build !")  
        
    def crop(self,xmin=0,xmax=0,ymin=0,ymax=0):
        '''
        Crop function to select the area of interest
        
        :return: crop aita object
        :rtype: aita
        :Exemple: >>> data.crop()
        
        .. note:: clic on the top left corner and bottom right corner to select the area
        '''
        
        
        if (xmin+xmax+ymin+ymax)==0:
            
            print('Warning : if you are using jupyter notebook with %matplotlib inline option, you should add %matplotlib qt to have a pop up figure before this function. You can add %matplotlib inline after if you want to come back to the initial configuration')
            
            # plot the data
            h=self.phi.plot()
            # select top left and bottom right corner for crop
            print('Select top left and bottom right corner for crop :')
            x=np.array(pylab.ginput(2))/self.phi.res
            plt.close("all")
            # create x and Y coordinate

            xx=[x[0][0],x[1][0]]
            yy=[x[0][1],x[1][1]]
            # size of the initial map
            ss=np.shape(self.phi.field)
            # find xmin xmax ymin and ymax
            xmin=int(np.ceil(np.min(xx)))
            xmax=int(np.floor(np.max(xx)))
            ymin=int(ss[0]-np.ceil(np.max(yy)))
            ymax=int(ss[0]-np.floor(np.min(yy)))
            
        
        # crop the map
        self.phi.field=self.phi.field[ymin:ymax, xmin:xmax]
        self.phi1.field=self.phi1.field[ymin:ymax, xmin:xmax]
        self.qua.field=self.qua.field[ymin:ymax, xmin:xmax]
        self.micro.field=self.micro.field[ymin:ymax, xmin:xmax]
        self.grains=self.micro.grain_label()
        
        # replace grains boundary with NaN number
        self.grains.field=np.array(self.grains.field,float)
        idx=np.where(self.micro.field==1)
        self.grains.field[idx]=np.nan
        
        return np.array([xmin,xmax,ymin,ymax])
        
    def fliplr(self):
        '''
        Applied an horizontal miror to the data
        
        :return:  aita object with an horizontal miror
        :rtype: aita
        :Exemple: >>> data.fliplr()
        '''
        
        # horizontal miror (fliplr) on all the data in self
        self.phi.field=np.fliplr(self.phi.field)
        self.phi1.field=np.mod(math.pi-np.fliplr(self.phi1.field),2*math.pi) # change phi1 angle by pi-phi1 modulo 2*pi
        self.qua.field=np.fliplr(self.qua.field)
        self.micro.field=np.fliplr(self.micro.field)
        self.grains.field=np.fliplr(self.grains.field)
        
    def rot180(self):
        '''
        Rotate the data of 180 degree
        
        :return: crop aita object
        :rtype: aita
        :Exemple: >>> data.rot180()
        '''
        
        # rotate the position of the data if 180 degre
        self.phi.field=np.flipud(np.fliplr(self.phi.field))
        self.phi1.field=np.mod(math.pi+np.flipud(np.fliplr(self.phi1.field)),2*math.pi) # rotate the c-axis : phi1 = pi + phi1 mod(2*pi)
        self.qua.field=np.flipud(np.fliplr(self.qua.field))
        self.micro.field=np.flipud(np.fliplr(self.micro.field))
        self.grains.field=np.flipud(np.fliplr(self.grains.field))
        
    def filter(self,value):
        ''' 
        Remove data of bad quality
        
        :param value: limit quality value between 0 to 100
        :type value: int
        :return: data object with no orientation with quality value under threshold
        :rtype: aita
        :Exemple: >>> data.filter(75)
        '''
        # find where quality<value
        x=np.where(self.qua.field < value)
        self.phi.field[x]=np.NAN
        self.phi1.field[x]=np.NAN
        
    def mean_grain(self):
        '''
        Compute the mean orientation inside the grain
        
        :return: data with only one orientation per grains, the mean orientation
        :rtype: aita
        :Exemple: >>> data.mean_orientation()
        '''
        # number of grain
        nb_grain=int(np.nanmax(self.grains.field))
        # loop on all the grain
        for i in list(range(nb_grain+1)):
            # find the pixel inside the grain i
            idx=np.where(self.grains.field==i)
            # compute the mean value of phi1 and phi and replace the value in the map
            self.phi.field[idx]=np.nanmean(self.phi.field[idx])
            self.phi1.field[idx]=np.nanmean(self.phi1.field[idx])
        
    def imresize(self,res):
        '''
        Resize the data
        
        :param res: the new resolution wanted in millimeter (mm)
        :type res: float
        :return: data with the resolution wanted
        :rtype: aita
        :Exemple: >>> data.imresize(0.25)
        '''
        self.phi.imresize(res)
        self.phi1.imresize(res)
        self.qua.imresize(res)
        self.grains.imresize(res)
        
        # make larger the boundaries to keep them
        self.micro.field=scipy.ndimage.binary_dilation(self.micro.field, iterations=np.int32(res/(2*self.micro.res)))
        # resize
        self.micro.imresize(res)
        
    def craft(self,nameId):
        '''
        Create the inputs for craft
        
        :param nameId: name of the prefixe used for craft files
        :type nameId: str
        :return: create file : nameId_micro.vtk, nameId.phase, nameId.in, nameId.load, nameId.output
        :Exemple: >>> data.craft('manip01')
        
        .. note:: nameId.load, nameId.output need to be rewrite to get a correct loading and the output wanted
           
        .. note:: nameId.in need to be adapt depending of your folder structure used for craft
            
        .. note:: NaN orientation value are removed by the closest orientation
        '''
        ##############################################
        # remove the grain boundary (remove NaN value)
        ##############################################
        # find where are the NaN Value corresponding to the grain boundary
        idx=np.where(np.isnan(self.grains.field))
        # while NaN value are in the microstructure we replace by an other value ...
        while np.size(idx)>0:
            # for all the pixel NaN
            for i in list(range(np.shape(idx)[1])):
                # if the pixel is at the bottom line of the sample, we choose the pixel one line upper ...
                if idx[0][i]==0:
                    k=idx[0][i]+1
                #... else we choose the pixel one line higher.
                else:
                    k=idx[0][i]-1
                # if the pixel is at the left side of the sample, we choose the pixel at its right  ...
                if idx[1][i]==0:
                    kk=idx[1][i]+1
                # else we choose the pixel at its left.
                else:
                    kk=idx[1][i]-1
                # Replace the value by the value of the neighbor select just before
                self.phi.field[idx[0][i], idx[1][i]]= self.phi.field[k, kk]
                self.phi1.field[idx[0][i], idx[1][i]]= self.phi1.field[k, kk]
                self.grains.field[idx[0][i], idx[1][i]]= self.grains.field[k, kk]
                # re-evaluate if there is sill NaN value inside the microstructure
            idx=np.where(np.isnan(self.grains.field))# re-evaluate the NaN
            
        # find the value of the orientation for each phase
        phi1=[]
        phi=[]
        phi2=[]
        for i in list(range(np.max(np.int32(self.grains.field)+1))):
            idx=np.where(np.int32(self.grains.field)==i)
            if np.size(idx)!=0:
                phi1.append(self.phi1.field[idx[0][0]][idx[1][0]])
                phi.append(self.phi.field[idx[0][0]][idx[1][0]])
                phi2.append(random.random()*2*math.pi)
            else:
                phi1.append(np.nan)
                phi.append(np.nan)
                phi2.append(np.nan)
        ################################   
        # Write the microstructure input
        ################################
        # size of the map
        ss=np.shape(self.grains.field)
        # open micro.vtk file
        micro_out=open(nameId+'_micro.vtk','w')
        # write the header of the file
        micro_out.write('# vtk DataFile Version 3.0 ' + str(datetime.date.today()) + '\n')
        micro_out.write('craft output \n')
        micro_out.write('ASCII \n')
        micro_out.write('DATASET STRUCTURED_POINTS \n')
        micro_out.write('DIMENSIONS ' + str(ss[1]) + ' ' + str(ss[0]) +  ' 1\n')
        micro_out.write('ORIGIN 0.000000 0.000000 0.000000 \n')
        micro_out.write('SPACING ' + str(self.grains.res) + ' ' + str(self.grains.res) + ' 1.000000 \n')
        micro_out.write('POINT_DATA ' + str(ss[0]*ss[1]) + '\n')
        micro_out.write('SCALARS scalars float \n')
        micro_out.write('LOOKUP_TABLE default \n')
        for i in list(range(ss[0]))[::-1]:
            for j in list(range(ss[1])):
                micro_out.write(str(int(self.grains.field[i][j]))+' ')
            micro_out.write('\n')        
        micro_out.close()
        ################################   
        ##### Write the phase input ####
        ################################
        phase_out=open(nameId+'.phase','w')
        phase_out.write('#------------------------------------------------------------\n')
        phase_out.write('# Date ' + str(datetime.date.today()) + '      Manip: ' + nameId + '\n')
        phase_out.write('#------------------------------------------------------------\n')
        phase_out.write('# This file give for each phase \n# *the matetial \n# *its orientation (3 euler angles)\n')
        phase_out.write('#\n#------------------------------------------------------------\n')
        phase_out.write('# phase    material       phi1    Phi   phi2\n')
        phase_out.write('#------------------------------------------------------------\n')
        for i in list(range(np.size(phi))):
            if 1-np.isnan(phi[i]):
                phase_out.write(str(i) + '          0              ' + str(phi1[i]) + ' ' + str(phi[i]) + ' ' + str(phi2[i]) + '\n');  
        phase_out.close()
        ################################
        # Write an exemple of load file##
        ################################
        out_load=open(nameId + '.load','w');
        out_load.write('#------------------------------------------------------------\n')
        out_load.write('# Date ' + str(datetime.date.today()) + '      Manip: ' + nameId + '\n')
        out_load.write('#------------------------------------------------------------\n')
        out_load.write('# choix du type de chargement \n')
        out_load.write('# direction contrainte imposée: S \n')
        out_load.write('# contrainte imposée:          C \n')
        out_load.write('# déformation imposée:         D \n')
        out_load.write('C\n')
        out_load.write('#------------------------------------------------------------\n')
        out_load.write('# nb de pas    temps        direction            facteur\n')
        out_load.write('#                            11 22 33 12 13 23\n')
        out_load.write('                5.            0  1  0  0  0  0    -0.5\n')
        out_load.write('5.            100.          0  1  0  0  0  0    -0.5\n')
        out_load.write('#\n')
        out_load.write('#------------------------------------------------------------\n')
        out_load.close()
        ###################################
        # Write an exemple of output file #
        ###################################    
        out_output=open(nameId + '.output','w')
        out_output.write('#------------------------------------------------------------\n')
        out_output.write('# Date ' + str(datetime.date.today()) + '      Manip: ' + nameId + '\n')
        out_output.write('#------------------------------------------------------------\n')
        out_output.write('equivalent stress image = yes 10,60,100\n')
        out_output.write('equivalent strain image = yes 10,60,100\n')
        out_output.write('#\n')
        out_output.write('stress image = yes 10,60,100\n')
        out_output.write('strain image = yes 10,60,100\n')
        out_output.write('#\n')
        out_output.write('backstress image = yes 10,60,100\n')
        out_output.write('#\n')
        out_output.write('strain moment = yes 5:100\n')
        out_output.write('stress moment = yes 5:100\n')
        out_output.write('im_format=vtk\n')
        out_output.close()  
        #####################################
        ## Write the input file for craft####
        #####################################
        out_in=open(nameId + '.in','w');
        out_in.write('#------------------------------------------------------------\n')
        out_in.write('# Date ' + str(datetime.date.today()) + '      Manip: ' + nameId + '\n')
        out_in.write('#------------------------------------------------------------\n')
        out_in.write('#\n')
        out_in.write('#\n')
        out_in.write('#------------------------------------------------------------\n')
        out_in.write('# name of the file of the image of the microstructure\n')
        out_in.write('microstructure=../'+ nameId+'_micro.vtk\n')
        out_in.write('#\n')
        out_in.write('#------------------------------------------------------------\n')
        out_in.write('# name of the file of the description of phases\n')
        out_in.write('phases=../'+nameId+'.phase\n')
        out_in.write('#\n')
        out_in.write('#------------------------------------------------------------\n')
        out_in.write('# name of the file describing the materials the phases are made of:\n')
        out_in.write('materials=../../../../Ice_Constitutive_Law/glace3_oc2_5mai2011.mat\n')
        out_in.write('#\n')
        out_in.write('#------------------------------------------------------------\n')
        out_in.write('# file of the loading conditions:\n')
        out_in.write('loading=../'+nameId + '.load\n')
        out_in.write('#\n')
        out_in.write('#------------------------------------------------------------\n')
        out_in.write('# file telling the outputs one wants to obtain:\n')
        out_in.write('output=../' +nameId + '.output\n')
        out_in.write('#\n')
        out_in.write('#------------------------------------------------------------\n')
        out_in.write('# The parameter C0 has to be set by craft:\n')
        out_in.write('C0=auto\n')
        out_in.write('#\n')
        out_in.write('#------------------------------------------------------------\n')
        out_in.write('# # required precision for equilibrium and for loading conditions:\n')
        out_in.write('precision=1.e-4, 1.e-4\n')
        out_in.write('#------------------------------------------------------------\n')
        out_in.close()
    
    def plotpdf(self,peigen=True,select_grain=False,grainlist=[],nbp=10000,contourf=False,cm2=cm.viridis,bw=0.1,projz=1,angle=np.array([30.,60.]),cline=15,n_jobs=-1):
        '''
        Plot pole figure for c-axis (0001)
        
        :param peigen: Plot the eigenvalues and eigenvectors on the pole figure (default = False)
        :type peigen: bool
        :param select_grain: select the grains use for the pole figure
        :type select_grain: bool
        :param grainlist: give the list of the grainId you want to plot
        :type grainlist: list
        :param nbp: number of pixel plotted
        :type nbp: int
        :param contourf: Do you want to add contouring to your pole figure ? (Default : False)
        :type contourf: bool
        :param cm2: colorbar (default : cm.viridis)
        :type cm2: cm
        :param bw: bandwidth to compute kernel density (default : 0.1) bw=0 mean find the best fit between 0.01 and 1
        :type bw: float
        :param projz: 0 or 1. It choose the type of projection. 0 (1) means projection in the plane z=0 (1).
        :type projz: int
        :param angle: plot circle for this angle value (default : np.array([30.,60.])) 0 if you don't want inner circle.
        :type angle: np.array
        :param cline: Number of line in contourf (default 15) Used only when contourf=True.
        :type cline: int
        :param n_jobs: number of job in parellel (CPU). Only use when bw=0 (best fit) (default : -1 mean all processor)
        :type n_jobs: int
        :return: pole figure image
        :rtype: matplotlib figure
        :return: eigenvalue
        :rtype: float
        :Exemple:
            >>> eigenvalue = data.plotpdf(peigen=True)
        '''
        
        if select_grain:
            if grainlist==[]:
                plt.imshow(self.grains.field,aspect='equal')
                plt.waitforbuttonpress()
                print('midle mouse clic when you are finish')
                #grain wanted for the plot
                id=np.int32(np.array(pylab.ginput(0)))
                plt.close('all')
                # find the label of grain
                label=self.grains.field[id[:,1],id[:,0]]
            else:
                label=grainlist
            tazi=[]
            tcol=[]
            for i in list(range(len(label))):
                idx=np.where(self.grains.field==label[i])
                tazi.append(list(np.mod(self.phi1.field[idx[0],idx[1]]-math.pi/2,2*math.pi)))
                tcol.append(list(self.phi.field[idx[0],idx[1]]))
                
            azi=np.transpose(np.concatenate(np.array(tazi)))
            col=np.transpose(np.concatenate(np.array(tcol)))
        else:
            # compute azimuth and colatitude
            azi=np.mod(self.phi1.field.reshape((-1,1))-math.pi/2,2*math.pi)
            col=self.phi.field.reshape((-1,1))


        # remove nan value
        idnan=np.isnan(azi)
        idlist=np.where(idnan==True)
        
        azi=np.delete(azi,idlist,0)
        col=np.delete(col,idlist,0)
        
        # compute [xc,yc,zc] the coordinate of the c-axis
        xc = np.multiply(np.cos(azi),np.sin(col))
        yc = np.multiply(np.sin(azi),np.sin(col))
        zc = np.cos(col)  
        
        v=vec3d.setvector3d(np.transpose(np.array([xc[:,0],yc[:,0],zc[:,0]])))
        v.stereoplot(nbpoints=nbp,contourf=contourf,bw=bw,cm=cm2,angle=angle,plotOT=peigen,projz=projz,cline=cline,n_jobs=n_jobs)

        plt.text(-1.4, 1.4, r'[0001]')
        
        eigvalue,eigvector=v.OrientationTensor2nd()
        
        return eigvalue
    
    def grain_ori(self):
        '''
        Give the grain orientation output
        '''
        plt.imshow(self.grains.field,aspect='equal')
        plt.waitforbuttonpress()
        print('midle mouse clic when you are finish')
        #grain wanted for the plot
        id=np.int32(np.array(pylab.ginput(0)))
        plt.close('all')
        
        phi=self.phi.field[id[:,1],id[:,0]]
        phi1=self.phi1.field[id[:,1],id[:,0]]
        
        return [phi1,phi]
        
    def plot(self,nlut=512):
        '''
        Plot the data using a 2d lut
        
        :param nlut: number of pixel tou want for the 2d LUT (default 512)
        :type nlut: int
        :return: figure of orientation mapping
        :rtype: matplotlib figure
        :Exemple: 
            >>> data.plot()
            >>> plt.show()
            >>> # print the associated color wheel
            >>> lut=lut()
            >>> plt.show()
            
        .. note:: It takes time to build the colormap
        '''
        # size of the map
        nx=np.shape(self.phi.field)
        # create image for color map
        img=np.ones([nx[0],nx[1],3])
        # load the colorwheel
        rlut=lut(nx=nlut,circle=False)
        nnlut=np.shape(rlut)
        nnnlut=nnlut[0]
        # fill the color map
        XX=(nnnlut-1)/2*np.multiply(np.sin(self.phi.field),np.cos(self.phi1.field))+(nnnlut-1)/2
        YY=(nnnlut-1)/2*np.multiply(np.sin(self.phi.field),np.sin(self.phi1.field))+(nnnlut-1)/2
    
        for i in list(range(nx[0])):
            for j in list(range(nx[1])):
                if ~np.isnan(self.phi.field[i,j]):
                    img[i,j,0]=rlut[np.int32(XX[i,j]),np.int32(YY[i,j]),0]
                    img[i,j,1]=rlut[np.int32(XX[i,j]),np.int32(YY[i,j]),1]
                    img[i,j,2]=rlut[np.int32(XX[i,j]),np.int32(YY[i,j]),2]
                
        h=plt.imshow(img,extent=(0,nx[1]*self.phi.res,0,nx[0]*self.phi.res))               
        
        return h,img
    

    def misorientation_profile(self, plot='all',orientation=False,pos=0):       
        '''
        Compute the misorientation profile along a line
        
        :param plot: option for to misorientation profile plot, 'all' (default), 'mis2o', 'mis2p'
        :type plot: str
        :param orientation: option for the color code used for the map, False (default) use phi1 and True use colorwheel (take time)
        :type orientation: bool
        :param pos: coordinate of the profile line - 0 (default) click on the map to select the 2 points
        :type pos: array
        :return: x - coordinate along the line
        :rtype: array, float
        :return: mis2o,mis2p - misorientation angle to the origin, and misorientation angle to the previous pixel
        :rtype: array, float
        :return: h - matplotlib image with line draw on the orientation map, subplot with mis2o and/or mis2p profile
        :return: pos - coordinate of the profile line
        :Exemple: 
            >>> [x,mis2o,mis2p,h,pos]=data.misorientation_profile()
            >>> rpos = pos[::-1]
            >>> [x,mis2o,mis2p,hr,pos]=data.misorientation_profile(pos=rpos)
            >>> plt.show()
        '''
        
        # size of the map
        ss=np.shape(self.phi1.field)
        # plot the data with phi1 value
        if np.size(pos)==1:
            h=plt.figure()
            self.phi1.plot()
            # select initial and final points for the line
            print('Select initial and final points for the line :')
            pos=np.array(pylab.ginput(2))
            plt.close(h)
        
        yy=np.float32([pos[0][0],pos[1][0]])/self.phi.res
        xx=np.float32([pos[0][1],pos[1][1]])/self.phi.res
        
        # numbers of pixel along the line
        nb_pixel=np.int32(np.sqrt((xx[1]-xx[0])**2+(yy[1]-yy[0])**2))
        
        # calcul for each pixel
        phi=[]
        phi1=[]
        x=[]
        xi=[]
        yi=[]
        mis2p=[]
        mis2o=[]
        ori=[]
        for i in list(range(nb_pixel)):
            # find the coordinate x an y along the line
            xi.append(ss[0]-np.int32(np.round(i*(xx[1]-xx[0])/nb_pixel+xx[0])))
            yi.append(np.int32(np.round(i*(yy[1]-yy[0])/nb_pixel+yy[0])))
            # extract phi and phi1
            phi.append(self.phi.field[xi[i],yi[i]])
            phi1.append(self.phi1.field[xi[i],yi[i]])
            
            # ori0 and orii are the c axis vector
            ori.append(np.mat([np.cos(np.mod(phi1[i]-math.pi/2,2*math.pi))*np.sin(phi[i]) , np.sin(np.mod(phi1[i]-math.pi/2,2*math.pi))*np.sin(phi[i]) ,np.cos(phi[i])]))   
            # mis2o is the misorientation between pixel i and the origin
            mis2o.append(np.float(np.arccos(np.abs(ori[0]*np.transpose(ori[i])))*180/math.pi))
            if i>0:
            # mis2p is the misorientation to the previous pixel    
                mis2p.append(np.float(np.arccos(np.abs(ori[i]*np.transpose(ori[i-1])))*180/math.pi))
            # x is the position along the line
                x.append(np.sqrt((xi[i]-xi[0])**2+(yi[i]-yi[0])**2))
            else:
                mis2p.append(0.0)
                x.append(0.0)


        #hh=plt.figure()
        plt.subplot(211)
        if orientation:
            self.plot()
        else:
            self.phi1.plot()
        #plt.hold('on')
        plt.plot(yy*self.phi.res,xx*self.phi.res)
        # plot misorientation profile
        plt.subplot(212)
        if plot=='all' or plot=='mis2o':
            plt.plot(x,mis2o,'-b')
        if plot=='all' or plot=='mis2p':
            plt.plot(x,mis2p,'-k')    
        plt.grid(True)
            
        return x, mis2o, mis2p, pos
    
    
    def addgrain(self,ori=0):
        '''
        add a grain inside the microstructure
        
        :param ori: orienation of the new grain [phi1 phi] (default random value)
        :type ori: array, float
        :return: new_micro, object with the new grain include
        :rtype: aita
        :Exemple: 
            >>> data.addgrain()      
        '''
        
        # select the contour of the grains
        h=self.grains.plot()
        # click on the submit of the new grain
        plt.waitforbuttonpress()
        print('click on the submit of the new grain :')
        x=np.array(pylab.ginput(3))/self.grains.res
        plt.close('all')
        
        # select a subarea contening the triangle
        minx=np.int(np.fix(np.min(x[:,0])))
        maxx=np.int(np.ceil(np.max(x[:,0])))
        miny=np.int(np.fix(np.min(x[:,1])))
        maxy=np.int(np.ceil(np.max(x[:,1])))
        
        # write all point inside this area
        gpoint=[]
        for i in list(range(minx,maxx)):
            for j in list(range(miny,maxy)):
                gpoint.append([i,j])
        
    
        # test if the point is inside the triangle    
        gIn=[]
        for i in list(range(len(gpoint))):
            gIn.append(isInsideTriangle(gpoint[i],x[0,:],x[1,:],x[2,:]))

        gpointIn=np.array(gpoint)[np.array(gIn)]
        
        #transform in xIn and yIn, the coordinate of the map
        xIn=np.shape(self.grains.field)[0]-gpointIn[:,1]
        yIn=gpointIn[:,0]
               
        # add one grains
        self.grains.field[xIn,yIn]=np.nanmax(self.grains.field)+1
        # add the orientation of the grains
        if ori==0:
            self.phi1.field[xIn,yIn]=random.random()*2*math.pi
            self.phi.field[xIn,yIn]=random.random()*math.pi/2
        else:
            self.phi1.field[xIn,yIn]=ori[0]
            self.phi.field[xIn,yIn]=ori[1]
            
        returnphi=self.phi1*mask
    
    def new_ori_TJ(self,mask,mean=True):
        '''
        Extract orientation to compare with CraFT simulation
        '''
        ng=(self.grains*mask).field        
        res=[]
        con=True
        
        while con:
            gID=self.grains.mask_build()
            print('triple junction label')
            x=input()
            ng=(self.grains*gID).field
            ngmax=np.nanmax(ng)
            for i in list(range(np.int32(ngmax))):
                id=np.where(self.grains.field==i)
                if len(id[0])>0:
                    if mean:
                        pp=np.array([[id[0][0],id[1][0]]])
                        phi1,pos=self.phi1.extract_data(pos=pp)
                        phi,pos=self.phi.extract_data(pos=pp)
                        if ~np.isnan(phi1):
                            res.append([i,phi1,phi,float(x)])
                    else:
                        for j in list(range(len(id[0]))):
                            pp=np.array([[id[0][j],id[1][j]]])
                            phi1,pos=self.phi1.extract_data(pos=pp)
                            phi,pos=self.phi.extract_data(pos=pp)
                            if ~np.isnan(phi1):
                                res.append([i,phi1,phi,float(x)])
                            
            print('continue ? 0 no, 1 yes')
            con=input()
        
        return res
    
    def mask(self,mask):
        '''
        Applied mask on aita data
        
        :param mask:
        :type mask: im2d.mask2d
        
        :return: aita object with the mask applied 
        :rtype: aita
        '''
        
        if (type(mask) is im2d.mask2d):
            phi1=self.phi1*mask
            phi=self.phi*mask
            qua=self.qua*mask
            micro=self.micro
            res=self.micro.res
            # reduce the size of the aita data : remouve band of NaN
            x,y=np.where(mask.field==1)
            minx=np.min(x)
            maxx=np.max(x)
            miny=np.min(y)
            maxy=np.max(y)
            
           
            
            ma=aita(phi1.field[minx:maxx,miny:maxy],phi.field[minx:maxx,miny:maxy],qua.field[minx:maxx,miny:maxy],micro.field[minx:maxx,miny:maxy],res)
            
        else:
            print('mask is not and mask2d object')
            ma=False
           
        return ma
    
    def grelon(self):
        '''
        Compute the angle between the directions defined by the "center" and the pixel with the c-axis direction
        
        :return: angle (degree)
        :rtype: im2d.image2d
        '''
        
        # Find the center
        self.phi1.plot()
        print('Click on the center of the hailstone')
        posc=plt.ginput(1)
        plt.close('all')
        
        ss=np.shape(self.phi1.field)
        
        # vecteur C
        xc=np.cos(self.phi1.field-math.pi/2)*np.sin(self.phi.field)
        yc=np.sin(self.phi1.field-math.pi/2)*np.sin(self.phi.field)
        
        nn=(xc**2+yc**2.)**.5
        xc=xc/nn
        yc=yc/nn
        
        # build x y
        xi=np.zeros(ss)
        yi=np.transpose(np.zeros(ss))
        xl=np.arange(ss[0])
        yl=np.arange(ss[1])
        xi[:,:]=yl
        yi[:,:]=xl
        yi=np.transpose(yi)
        # center and norm
        xcen=np.int32(posc[0][0]/self.phi1.res)
        ycen=(ss[0]-np.int32(posc[0][1]/self.phi1.res))
        xv=xi-xcen
        yv=yi-ycen
        
        nn=(xv**2.+yv**2.)**0.5
        xv=xv/nn
        yv=yv/nn
        #
        plt.figure()
        plt.imshow(nn)
        plt.figure()
        plt.quiver(xi[xcen-50:xcen-50],yi[ycen-50:ycen-50],xv[xcen-50:xcen-50],yv[ycen-50:ycen-50],scale=1000)
        
        #
        acos=xv*xc+yv*yc
        
        angle=np.arccos(acos)*180./math.pi
        
        id=np.where(angle>90)
        angle[id]=180-angle[id]
                
        return im2d.image2d(angle,self.phi1.res),xi,yi,xv,yv
        
        

    def gmsh_geo(self,name,resGB=1,resInG=20,DistMin=4.5,DistMax=5):
        '''
        Create geo file for GMSH input
        resInG             _______
                          /
                         /
                        / |
                       /  
        resGB ________/   |
                 DistMin  DistMax
                   
        :param name: output file name without extension
        :type name: str
        :param resGB: resolution on the Grains Boundaries (in pixel)
        :type resGB: float
        :param resInG: resolution within the Grains (in pixel)
        :type resInG: float
        :param LcMin: starting distance for the transition between resGB and resInG
        :type Lcmin: float
        :param LcMax: ending distance for the transition between resGB and resInG
        '''
        
        res=self.grains.res
        #Extract grainId map
        grainId=self.grains.field
        #remove the 0 value in the grainId numpy. To do so it is dilating each grain once.
        #print('Building grainId map')
        for i in tqdm(range(np.int(np.nanmax(grainId)))):
            mask=grainId==i+1
            mask=skimage.morphology.dilation(mask)
            grainId[mask]=i+1
        
        # Extract contours of each grains
        contour=[]
        for i in list(range(np.int(np.nanmax(grainId)))):
            gi=grainId==i+1
            if np.sum(gi)!=0:
                contour.append(skimage.measure.find_contours(gi,level=0.5,fully_connected='high')[0])
        
        # Open the geo file to write in it
        geo_out=open(name+'.geo','w')
        geo_out.write('Mesh.Algorithm=5; \n')
        
        # Extract the contour of the microstructure
        ss=grainId.shape
        xmin=0
        ymin=0
        xmax=ss[1]-1
        ymax=ss[0]

        # Variable with all the point exported in the .geo file
        # I already add the corner points 
        allPoints=[np.array([xmin,ymin]),np.array([xmin,ymax]),np.array([xmax,ymax]),np.array([xmax,ymin])]

        # write the corner point in the geo file
        geo_out.write('Point('+str(1)+')={'+str(xmin*res)+','+str(ymin*res)+',0.0,'+str(resInG*res)+'}; \n')
        geo_out.write('Point('+str(2)+')={'+str(xmin*res)+','+str(ymax*res)+',0.0,'+str(resInG*res)+'}; \n')
        geo_out.write('Point('+str(3)+')={'+str(xmax*res)+','+str(ymax*res)+',0.0,'+str(resInG*res)+'}; \n')
        geo_out.write('Point('+str(4)+')={'+str(xmax*res)+','+str(ymin*res)+',0.0,'+str(resInG*res)+'}; \n')
        
        # Build line for the sample contour in .geo file
        geo_out.write('\n')
        geo_out.write('Line(1)={1,2};\n')
        geo_out.write('Line(2)={2,3};\n')
        geo_out.write('Line(3)={3,4};\n')
        geo_out.write('Line(4)={4,1};\n')

        # Build the line loop to define the surface
        geo_out.write('Line Loop(1) = {1,2,3,4};\n')
        # Define the Plane Surface
        geo_out.write('Plane Surface(2) = {1};\n')
        geo_out.write('\n')
        # Define Physical Surface and Limit where limite condition will be applied
        geo_out.write('Physical Line("Left face") = {1};\n')
        geo_out.write('Physical Line("Top face") = {2};\n')
        geo_out.write('Physical Line("Right face") = {3};\n')
        geo_out.write('Physical Line("Bottom face") = {4};\n')
        geo_out.write('Physical Surface("Ice") = {2};\n')
        geo_out.write('\n')

        
        # Write boudaries points in .geo file
        #print('Write boudaries points in .geo file')
        k_point=5
        for i in tqdm(range(len(contour))):
            for j in list(range(len(contour[i]))):
                x=contour[i][j][1]
                y=ss[0]-contour[i][j][0]
                pos=np.array([x,y]) # Position of the point in pixel
                if np.sum(np.sum(pos==allPoints,axis=1)==2)==0: # Test if the point is already save in allPoints and therefore exported in the .geo file
                    allPoints.append(pos) # Save position of point
                    geo_out.write('Point('+str(k_point)+')={'+str(x*res)+','+str(y*res)+',0.0,'+str(resInG*res)+'}; \n') # Export Point in .geo
                    k_point+=1 # Increment point label in .geo file

        nb_point=k_point-1 # save the number of point exported    
        geo_out.write('\n')
        
        # Write Field option in .geo file to have finner mesh close to grains boundaries
        geo_out.write('Field[1]=Distance;\n')
        # Export all the Points in NodesList variable in .geo file
        geo_out.write('Field[1].NodesList={')
        k_point=5
        for i in list(range(nb_point)):
            geo_out.write(str(k_point)+',')
            k_point+=1
            
        geo_out.write(str(k_point))
        geo_out.write('};\n')

        geo_out.write('Field[2] = Threshold;\n')
        geo_out.write('Field[2].IField = 1;\n')
        geo_out.write('Field[2].LcMin = '+str(resGB*res)+';\n')
        geo_out.write('Field[2].LcMax = '+str(resInG*res)+';\n')
        geo_out.write('Field[2].DistMin = '+str(DistMin*res)+';\n')
        geo_out.write('Field[2].DistMax = '+str(DistMax*res)+';\n')
        geo_out.write('Background Field = 2;\n')
        geo_out.close()
        
        print('Export .geo done')
        
    def mesh(self,name,resGB=1,resInG=20,DistMin=2.8,DistMax=3,opt=0):
        '''
        Create mesh file
        '''
        res=self.grains.res
        #Extract grainId map
        grainId=self.grains.field
        #remove the 0 value in the grainId numpy. To do so it is dilating each grain once.
        #print('Building grainId map')
        for i in tqdm(range(np.int(np.nanmax(grainId)))):
            mask=grainId==i+1
            mask=skimage.morphology.dilation(mask)
            grainId[mask]=i+1
        
        # Extract contours of each grains
        contour=[]
        for i in list(range(np.int(np.nanmax(grainId)))):
            gi=grainId==i+1
            if np.sum(gi)!=0:
                contour.append(skimage.measure.find_contours(gi,level=0.5,fully_connected='high')[0])
                
        ss=grainId.shape
        xmin=0
        ymin=0
        xmax=ss[1]-1
        ymax=ss[0]
        if opt:
            allPoints=[]
            for i in tqdm(range(len(contour))):
                for j in list(range(len(contour[i]))):
                    x=contour[i][j][1]
                    y=ss[0]-contour[i][j][0]
                    pos=np.array([x,y]) # Position of the point in pixel
                    if len(allPoints)==0 or np.sum(np.sum(pos==allPoints,axis=1)==2)==0: # Test if the point is already save in allPoints and therefore exported in the .geo file
                        allPoints.append(pos) # Save position of point

            with pygmsh.geo.Geometry() as geom:
                poly = [
                    geom.add_polygon([[xmin*res, ymin*res],[xmin*res, ymax*res],[xmax*res, ymax*res],[xmax*res, xmin*res],],mesh_size=resInG*res)
                ]

                for i in list(range(len(allPoints))):
                    poly.append(geom.add_point([allPoints[i][0]*res,allPoints[i][1]*res],mesh_size=resInG*res))

                #print(GB)

                field0 = geom.add_boundary_layer(
                    nodes_list=poly[1::],
                    lcmin=resGB*res,
                    lcmax=resInG*res,
                    distmin=DistMin*res,
                    distmax=DistMax*res,
                )
                geom.set_background_mesh([field0], operator="Min")

                mesh = geom.generate_mesh()
        else:
            allPoints=[]
            GB=[]
            for i in tqdm(range(len(contour))):
                gi=[]
                for j in list(range(len(contour[i]))):
                    x=contour[i][j][1]
                    y=ss[0]-contour[i][j][0]
                    pos=np.array([x,y]) # Position of the point in pixel
                    if len(allPoints)==0 or np.sum(np.sum(pos==allPoints,axis=1)==2)==0:
                        allPoints.append(pos) # Save position of point
                        id=np.where(np.sum(pos==allPoints,axis=1)==2)[0][0]
                        gi.append(id+1)
                    else:
                        id=np.where(np.sum(pos==allPoints,axis=1)==2)[0][0]
                        gi.append(id+1)
                    
                GB.append(gi)
                        
            with pygmsh.geo.Geometry() as geom:
                poly = [
                    geom.add_polygon([[xmin*res, ymin*res],[xmin*res, ymax*res],[xmax*res, ymax*res],[xmax*res, xmin*res],],mesh_size=resInG*res)]

                # add points to geom
                for i in list(range(len(allPoints))):
                    poly.append(geom.add_point([allPoints[i][0]*res,allPoints[i][1]*res],mesh_size=resInG*res))

                        
                # add line to geom
                for i in list(range(len(GB))):
                    for j in list(range(len(GB[i])-1)):
                        evaltxt='poly.append(geom.add_line('
                        evaltxt=evaltxt+'poly['+str(GB[i][j])+'],poly['+str(GB[i][j+1])+']))'
                        eval(evaltxt)
                            
                list_lines=poly[len(allPoints)+1::]
                list_lines.append(poly[0].lines[0])
                list_lines.append(poly[0].lines[1])
                list_lines.append(poly[0].lines[2])
                list_lines.append(poly[0].lines[3])
                        
            
                field0 = geom.add_boundary_layer(
                    edges_list=list_lines,
                    lcmin=resGB*res,
                    lcmax=resInG*res,
                    distmin=DistMin*res,
                    distmax=DistMax*res
                    )
                geom.set_background_mesh([field0], operator="Min")

                mesh = geom.generate_mesh()
            
        mesh.write(name+'.vtk')
        
        # Use vtk to add grainId value
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(name+'.vtk')
        reader.Update()
        polydata = reader.GetOutput()
        # compute grainId
        mesh_grains=vtk.vtkIntArray()
        mesh_grains.SetNumberOfComponents(0)
        mesh_grains.SetName("GrainsId")

        ss=np.shape(self.grains.field)

        for i in list(range(polydata.GetNumberOfCells())):
            if polydata.GetCellType(i)==5:
                tri=polydata.GetCell(i)
                center=np.zeros(3)
                tri.TriangleCenter(tri.GetPoints().GetPoint(0),tri.GetPoints().GetPoint(1),tri.GetPoints().GetPoint(2),center)
                mesh_grains.InsertNextValue(np.int(grainId[np.int(ss[0]-center[1]/res),np.int(center[0]/res)]))
            else:
                mesh_grains.InsertNextValue(0)
                
        polydata.GetCellData().AddArray(mesh_grains)
        
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(name+'.vtu')
        writer.SetInputData(polydata)
        writer.Write()
        
        return polydata
            
    
            
            
                
      
##########################################################################
###################### Function need for aita class  #####################
##########################################################################        
        
def cart2pol(x, y):
    '''
    Convert cartesien coordinate x,y into polar coordinate rho, theta
    
    :param x: x cartesian coordinate
    :param y: y cartesian coordinate
    :type x: float
    :type y: float
    :return: rho (radius), theta (angle)
    :rtype: float
    :Exemple: >>> rho,theta=cart2pol(x,y)
    '''
    # transform cartesien to polar coordinate
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def lut(nx=512,circle=True):
    '''
    Create a 2D colorwheel
    
    :param nx: number of pixel for the colorwheel
    :param circle: do you want create a black circle around
    :param semi: do you want a semi LUT
    :type nx: int
    :type circle: bool
    :return: lut
    :rtype: array of size [nx,nx,3]
    :Exemple:
        >>> lut2d=lut()
        >>> plt.imshow(lut)
        >>> plt.show()
    '''
    
    
    x=np.linspace(-math.pi/2, math.pi/2, nx)
    y=np.linspace(-math.pi/2, math.pi/2, nx)
    xv, yv = np.meshgrid(x, y)
    rho,phi=cart2pol(xv, yv)
    h = (phi-np.min(phi))/(np.max(phi)-np.min(phi))
    v = rho/np.max(rho)

    luthsv = np.ones((nx, nx,3))
    luthsv[:,:,0]=h
    luthsv[:,:,2]=v
    # colorwheel rgb
    lutrgb = np.ones((nx, nx,3))
    for i in list(range(nx)):
        for j in list(range(nx)):
            lutrgb[i,j,0],lutrgb[i,j,1],lutrgb[i,j,2]=colorsys.hsv_to_rgb(luthsv[i,j,0],luthsv[i,j,1],luthsv[i,j,2])

        
    # build a circle color bar        
    if circle:
        for i in list(range(nx)):
            for j in list(range(nx)):
                if ((i-nx/2)**2+(j-nx/2)**2)**0.5>(nx/2):
                    lutrgb[i,j,0]=0 
                    lutrgb[i,j,1]=0
                    lutrgb[i,j,2]=0
                    


    return lutrgb
    
    
def isInsideTriangle(P,p1,p2,p3): #is P inside triangle made by p1,p2,p3?
    '''
    test if P is inside the triangle define by p1 p2 p3
    
    :param P: point you want test
    :param p1: one submit of the triangle
    :param p2: one submit of the triangle
    :param p3: one submit of the triangle
    :type P: array
    :type p1: array
    :type p2: array
    :type p3: array
    :return: isIn
    :rtype: bool
    :Exemple:
        >>> isInsideTriangle([0,0],[-1,0],[0,1],[1,0])
        >>> isInsideTriangle([0,-0.1],[-1,0],[0,1],[1,0])
    '''
    x,x1,x2,x3 = P[0],p1[0],p2[0],p3[0]
    y,y1,y2,y3 = P[1],p1[1],p2[1],p3[1]
    full = abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    first = abs (x1 * (y2 - y) + x2 * (y - y1) + x * (y1 - y2))
    second = abs (x1 * (y - y3) + x * (y3 - y1) + x3 * (y1 - y))
    third = abs (x * (y2 - y3) + x2 * (y3 - y) + x3 * (y - y2))
    return abs(first + second + third - full) < .0000001

def euler2azi(phi1,phi):
    '''
    Convert Euler angle to azimuth and colatitude
    :param phi1:
    :type phi1: array
    :param phi:
    :type phi: array
    :return: azi
    :rtype: array
    :return: col
    :rtype: array
    '''
    col=phi
    azi=np.mod((phi1-math.pi/2.),2.*math.pi)
    
    return azi,col
