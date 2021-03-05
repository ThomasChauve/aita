import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.tri as tri
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import random
import sys
import AITAToolbox.uniform_dist as uni_dist
import os

class setvector3d(object):
    '''
    Object to work on a set of 3d unit vector
    '''
    
    pass

    def __init__(self,data):
        '''
        :param data: data is an array of dimention 3
        :type data: np.array
        '''
        
        if np.shape(data)[1] !=3:
            print('Your array should have 3 columns X,Y,Z')
            
        
        norm_data=np.linalg.norm(data,axis=1)
        id=np.where(norm_data!=1)
        data = np.float64(data)
        for i in list(range(len(id[0]))):
            if i==0:
                #print('Normalising vector length to 1')
                data[id[0][i],:]=data[id[0][i],:]/norm_data[id[0][i]]
        
        
        self.vector=data
        
    def inv(self):
        '''
        Create a new set of vector -v
        '''
        return setvector3d(-self.vector)
        
        
    def OrientationTensor2nd(self):
        '''
        Compute the normelized second order orientation tensor

        :return eigvalue: eigen value w[i]
        :rtype eigvalue: np.array
        :return eigvector: eigen vector v[:,i]
        :rtype eigvector: np.array
        :note: eigen value w[i] is associate to eigen vector v[:,i] 
        '''
        a11 = np.float32(np.nanmean(np.float128(np.multiply(self.vector[:,0],self.vector[:,0]))))
        a22 = np.float32(np.nanmean(np.float128(np.multiply(self.vector[:,1],self.vector[:,1]))))
        a33 = np.float32(np.nanmean(np.float128(np.multiply(self.vector[:,2],self.vector[:,2]))))
        a12 = np.float32(np.nanmean(np.float128(np.multiply(self.vector[:,0],self.vector[:,1]))))
        a13 = np.float32(np.nanmean(np.float128(np.multiply(self.vector[:,0],self.vector[:,2]))))
        a23 = np.float32(np.nanmean(np.float128(np.multiply(self.vector[:,1],self.vector[:,2]))))
         
        Tensor=np.array([[a11, a12, a13],[a12, a22, a23],[a13, a23, a33]])
        eigvalue,eigvector=np.linalg.eig(Tensor)
        
        idx = eigvalue.argsort()[::-1]
           
        return eigvalue[idx],eigvector[:,idx]
        
        
    def stereoplot(self,contourf=False,bw=0.03,plotOT=True,nbpoints=0,projz=1,angle=np.array([30.,60.]),cm=cm.viridis,cline=15,n_jobs=-1):
        '''
        Plot a stereographic projection of the vector

        :param contourf: filled contour plot (default False)
        :type contourf: bool
        :param bw: bandwidth for Kernel density function (default 0.03). bw=0 mean find the best fit between 0.01 and 1
        :type bw: float
        :param plotOT: plot the eigen vector on the pole figure (default True)
        :type plotOT: bool
        :param nbpoints: number of random vector selected to compute the pole figure (default 0, mean every pixel are used) Be careful with to many point the computation can be very slow. Up to 10000 it is still reasonable.
        :type nbpoints: int
        :param projz: 0 or 1. It choose the type of projection. 0 (1) means projection in the plane z=0 (1).
        :type projz: int
        :param angle: angle in degree for inner circle between 0 and 90 (0 mean no inner circle, default np.array([30.,60.]))
        :type angle: np.array
        :param cm: colorbar
        :type cm: cm
        :param cline: Number of line in contourf (default 15) Used only when contourf=True.
        :type cline: int
        :param n_jobs: number of job in parellel (CPU). Only use when bw=0 (best fit) (default : -1 mean all processor)
        :type n_jobs: int
        '''
       
        if nbpoints==0:
            subset=self
        else:
            subset=self.subset(nbpoints)
        
        v1=subset.concatenate(subset.inv())
        #############################
        ## compute the PDF sklearn ##
        #############################
        # http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html
        phi,theta=v1.cart2spher()
        theta=theta-np.pi
        phi=phi-np.pi/2

        if bw==0: #it means automatically compute bw value. See https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
            grid = GridSearchCV(KernelDensity(metric='haversine',kernel='gaussian', algorithm='ball_tree'),{'bandwidth': np.linspace(0.01, 1., 10)},cv=5,n_jobs=n_jobs) # 20-fold cross-validation
            grid.fit(np.transpose(np.array([phi,theta])))
            print(grid.best_params_)
            kde2 = grid.best_estimator_
        else:
            kde2 = KernelDensity(bandwidth=bw, metric='haversine',kernel='gaussian', algorithm='ball_tree')
            kde2.fit(np.transpose(np.array([phi,theta])))
        
        # Evaluate the KDE in a given set of direction
        if contourf:
            val=uni_dist.unidist
            dim=int(np.size(val)/3)
            xx=val.reshape([dim,3])
            id=np.where(xx[:,2]>0)
            xxuse=xx[id[0],:]
            viso=setvector3d(xxuse)
            # add point on the disc for contourf
            tot=1000
            omega = np.linspace(0, 2*np.pi, tot)
            zcir = np.zeros(tot)
            xcir = np.cos(omega)
            ycir = np.sin(omega)
            
            vcir=setvector3d(np.transpose(np.array([xcir,ycir,zcir])))
            
            vtot=viso.concatenate(vcir)
            
            phie,thetae=vtot.cart2spher()
            thetae=thetae-np.pi
            phie=phie-np.pi/2
            weights=kde2.score_samples(np.transpose(np.array([phie,thetae])))
            X=vtot.vector[:,0]
            Y=vtot.vector[:,1]
            Z=vtot.vector[:,2]
        else:
            phie,thetae=subset.cart2spher()
            thetae=thetae-np.pi
            phie=phie-np.pi/2
            weights=kde2.score_samples(np.transpose(np.array([phie,thetae])))
            X=subset.vector[:,0]
            Y=subset.vector[:,1]
            Z=subset.vector[:,2]
                
        # put all the direction in the upper hemisphere as we consider v ans -v equivalent
        id=np.where(Z<0)
        X[id]=-X[id]
        Y[id]=-Y[id]
        Z[id]=-Z[id]
        
        
        # Choose the type of projection
        if projz==0:
            LpL=1./(1.+Z)
            xx=LpL*X
            yy=LpL*Y
            rci=np.multiply(1./(1.+np.sin((90-angle)*np.pi/180.)),np.cos((90-angle)*np.pi/180.))
            rco=1.
        else:
            vz1=setvector3d(np.transpose(np.array([X,Y,Z])))
            phip,thetap=vz1.cart2spher()
            xx = np.multiply(2*np.sin(phip/2),np.cos(thetap))
            yy = np.multiply(2*np.sin(phip/2),np.sin(thetap))
            rci=2.*np.sin(angle/2.*np.pi/180.)
            rco=2.**0.5
        
        # Prepare the contour plot
        #plt.figure(figsize=(10,10),dpi=160)
        
        if contourf:
            triang = tri.Triangulation(xx, yy)
            plt.tricontour(xx, yy, np.exp(weights), cline, linewidths=0.5, colors='k')
            plt.tricontourf(xx, yy, np.exp(weights), cline,cmap=cm)
        else:
            plt.scatter(xx, yy, c=np.exp(weights), s=20,cmap=cm)
        
        plt.colorbar(orientation='vertical',aspect=4,shrink=0.5)
        # Compute the outer circle
        omega = np.linspace(0, 2*np.pi, 1000)
        x_circle = rco*np.cos(omega)
        y_circle = rco*np.sin(omega)
        plt.plot(x_circle, y_circle,'k', linewidth=3)
        # compute a 3 circle
        if np.size(angle)>1:
            for i in list(range(len(rci))):
                x_circle = rci[i]*np.cos(omega)
                y_circle = rci[i]*np.cos(i*np.pi/180.)*np.sin(omega)
                
                plt.plot(x_circle, y_circle,'k', linewidth=1.5)
                plt.text(x_circle[200], y_circle[300]+0.04,'$\phi$='+str(angle[i])+'°')
            # plot Theta line
            plt.plot([0,0],[-1*rco,1*rco],'k', linewidth=1.5)
            plt.text(rco-0.2, 0+0.06,'$\Theta$=0°')
            plt.text(-rco+0.1, 0-0.06,'$\Theta$=180°')
            plt.plot([-rco,rco],[0,0],'k', linewidth=1.5)
            plt.text(-0.25, rco-0.25,'$\Theta$=90°')
            plt.text(0.01, -rco+0.15,'$\Theta$=270°')
            plt.plot([-0.7071*rco,0.7071*rco],[-0.7071*rco,0.7071*rco],'k', linewidth=1.5)
            plt.plot([-0.7071*rco,0.7071*rco],[0.7071*rco,-0.7071*rco],'k', linewidth=1.5)
          
            
        # draw a cross for x and y direction
        plt.plot([1*rco, 0],[0, 1*rco],'+k',markersize=12)
        # write axis
        plt.text(1.05*rco, 0, r'X')
        plt.text(0, 1.05*rco, r'Y')
        plt.axis('equal')
        plt.axis('off')
                   
        
        if plotOT:
            eigvalue,eigvector=self.OrientationTensor2nd()
            for i in list(range(3)): # Loop on the 3 eigenvalue
                if (eigvector[2,i]<0):
                    v=-eigvector[:,i]
                else:
                    v=eigvector[:,i]
                    
                    
                if projz==0:    
                    LpLv=1./(1.+v[2])
                    xxv=LpLv*v[0]
                    yyv=LpLv*v[1]
                else:
                    phiee=np.arccos(v[2])
                    thetaee=np.arctan2(v[1],v[0])
                    xxv = np.multiply(2*np.sin(phiee/2),np.cos(thetaee))
                    yyv = np.multiply(2*np.sin(phiee/2),np.sin(thetaee))
                    
                plt.plot(xxv,yyv,'sk',markersize=8)
                plt.text(xxv+0.04, yyv+0.04,str(round(eigvalue[i],2)))
        
        return
    
    def cart2spher(self):
        '''
        Return vector on spherical coordinate phi theta

        :return phi: Angle from the z axis
        :rtype phi: np.array
        :return theta: Angle within the xOy plane
        :rtype theta: np.array
        '''
        
        phi=np.arccos(self.vector[:,2])
        theta=np.arctan2(self.vector[:,1],self.vector[:,0])
        
        return phi,theta
    
    def concatenate(self,v1):
        '''
        Concatenate 2 set of vector3d
        '''
        
        return setvector3d(np.concatenate((self.vector,v1.vector)))
    
    def subset(self,nbpoints):
        '''
        Select a random subset of the set of vector 3d

        :param nbpoints: number of point
        :type nbpoints: int
        :return v: 
        :rtype v: setvector3d
        '''
        
        id=[random.randint(0, np.shape(self.vector)[0]-1) for p in range(0, nbpoints)]
        
        return setvector3d(self.vector[id,:])
