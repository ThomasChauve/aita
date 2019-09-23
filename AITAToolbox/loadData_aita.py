import AITAToolbox.aita as aita
import numpy as np
import math
from skimage import io
from PIL import Image

def aita5col(data_adress,micro_adress=0):
    '''
    Function to open data from AITA analyser using 'cistodat' which give output file with 5 column (x,y,azi,col,qua)
    
    :param data_adress: orientation.dat file
    :type data_adress: str
    :param micro_adress: .bmp (24 bit) with black background and grains boundary in white
    :type micro_adress: str
    :return: aita object
    :rtype: aita
    '''
    # load data from G50 output
    file=open(data_adress,'r')
    azi,col,qua = np.loadtxt(file, skiprows=19,usecols=(2,3,5),dtype='f,f,f',comments='[eof]',unpack=True)    
    file.close()
    # read head of file
    file=open(data_adress,'r')
    a=[]
    [a.append(file.readline()) for i in list(range(16))]
    file.close()
    # resolution mu m
    res=int(a[5][10:12])
    # transforme the resolution in mm
    resolution=res/1000. 
    # number of pixel along x
    nx=int(a[14][9:14])
    # number of pixel along y
    ny=int(a[15][9:13])
    
    # reashape the vector to a matrix
    # use Bunge Euler angle convention
    phi1_field=np.mod((azi.reshape((ny,nx))+90)*math.pi/180,2*math.pi)
    phi_field=col.reshape((ny,nx))*math.pi/180
    qua_field=qua.reshape((ny,nx))
    
    #open micro.bmp if necessary
    if micro_adress==0:
        micro_field=np.zeros((ny,nx))
    else:
        micro_bmp = io.imread(micro_adress)
        mm=np.max(micro_bmp)
        if len(micro_bmp.shape)==3:
            micro_field=micro_bmp[:,:,0]/mm
        elif len(micro_bmp.shape)==2:
            micro_field=micro_bmp[:,:]/mm
        
    return aita.aita(phi1_field,phi_field,qua_field,micro_field,resolution=resolution)

def aita3col(data_adress, im_adress, micro_adress=0):
    '''
    Function to open data from AITA analyser using 'cistodat' which give output file with 3 column azi,col,qua)
    Require to load an image file to get the image dimension nx, ny
    
    :param data_adress: orientation.dat file
    :type data_adress: str
    :param im_adress: any .bmp image file from AITA G50
    :type im_adress: str
    :param micro_adress: .bmp (24 bit) with black background and grains boundary in white
    :type micro_adress: str
    :return: aita object
    :rtype: aita
    '''
    # load data from G50 output
    with open(data_adress,'r') as file:    
        azi, col, qua = np.loadtxt(file, skiprows=16,comments='[eof]',unpack=True)
    
    with open(data_adress,'r') as file:
        a = [file.readline() for i in range(16)]
    # resolution mu m
    res=int(a[5][10:12])
    # transforme the resolution in mm
    resolution=res/1000. 

    im = Image.open(im_adress)
    [nx,ny] = im.size
    
    # reashape the vector to a matrix
    # use Bunge Euler angle convention

    phi1_field = np.mod((azi.reshape((ny,nx))+90)*math.pi/180,2*math.pi)
    phi_field = col.reshape((ny,nx))*math.pi/180
    qua_field = qua.reshape((ny,nx))
    
    #open micro.bmp if necessary
    if micro_adress==0:
        micro_field=np.zeros((ny,nx))
    else:
        micro_bmp = io.imread(micro_adress)
        mm=np.max(micro_bmp)
        if len(micro_bmp.shape)==3:
            micro_field=micro_bmp[:,:,0]/mm
        elif len(micro_bmp.shape)==2:
            micro_field=micro_bmp[:,:]/mm
    
    return aita.aita(phi1_field,phi_field,qua_field,micro_field,resolution=resolution)
