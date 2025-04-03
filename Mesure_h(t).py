#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 18:01:11 2023

@author: krishanbumma
"""

import matplotlib.pyplot as plt
import numpy as np
import skimage as skimage
import skimage.filters as filters
from scipy.signal import find_peaks
from skimage import io
import pandas as pd
import  scipy.interpolate as interpolate




"""

Mesure de la hauteiur du FDS

Il faut que l'image initiale ait une hauteur de 1cm (scale automatique ensuite)
c'est la hauteur entre le bas de la cellule et le bas de l'electrode

"""
Data_exp = pd.read_csv('/Users/krishanbumma/Desktop/Mesure Phi(z)/RecapMesurePhi(z)_csv.csv')

Path = '/Users/krishanbumma/Desktop/Manip09_10/'
fname='/Users/krishanbumma/Desktop/Manip09_10/09_10_manip1_2fps_cropht.tif'
Stack = io.imread(fname)

# Scale = float(Data_exp.loc[Data_exp['Manip']==fname[len(Path):]]['Scale (pix/cm)'])

Scale =  233.96226415094338


Stack_red=Stack[:,:,:,0]
Stack_green=Stack[:,:,:,1]
Stack_blue=Stack[:,:,:,2]

Stack_green = Stack_green.astype(float)
Stack_blue = Stack_blue.astype(float)

Stack_gmb=-Stack_green#-Stack_blue


Flatten=np.sum(Stack_gmb, axis=2)/Stack_gmb.shape[2]


Flatten = filters.gaussian(Flatten, sigma = 1)
Flatten=np.flip(Flatten, axis=1)
Der_flatten=Flatten[:,1:]-Flatten[:,:-1]

plt.imshow(Stack_gmb[100])

plt.show()

# plt.plot(Flatten[400])

# plt.plot(Der_flatten[400])



# plt.show()


#%%
H_list = []
Hm = []
Hp = []
t_list = []

h_pos = 0 
for t in range(len(Der_flatten)): 
    try : 
        k=3
        New_h_pos = np.argmin(Der_flatten[t][k:])+k#max(np.argmin(Der_flatten[t]), h_pos)#[max(0,h_pos-100):h_pos+100]))     max(0,h_pos-200):min(h_pos+200, len(Der_flatten[t]))]))
        #hm = 
        hm=None
        i = 0
        while hm==None  : 
            if -(Der_flatten[t][New_h_pos-i]) > -(Der_flatten[t][New_h_pos]/1.5) : 
                i+=1
            elif New_h_pos-i==0 : 
                hm=New_h_pos-i
                Hm.append(hm)
                #
                print('here')
                
                break
            else : 
                hm=New_h_pos-i
                Hm.append(hm)
                print('here2')
                break
                
        
        hp=None
        i = 0
        while hp==None  : 
            if  -(Der_flatten[t][New_h_pos+i]) > -(Der_flatten[t][New_h_pos]/1.5) : 
                i+=1
            elif New_h_pos+i==len(Der_flatten[t])-1: 
                hp=New_h_pos+i
                Hp.append(hp)
                #print('here3')
                break
            
            else : 
                hp=New_h_pos+i
                Hp.append(hp)
                #print('here4')
                break

    
        # print('len hp = ', len(Hp))
        # print('len hm = ', len(Hm))
        t_list.append(t/2)
        H_list.append(New_h_pos)#)+max(0,h_pos)+100)
        
        h_pos=New_h_pos#+h_pos+1-100

            
    except : 
        pass
    
    if t/10==int(t/10):
    
        fig, ax = plt.subplots()
        ax.plot([i for i in range(len(Der_flatten[t]))],Der_flatten[t])
        ax.plot([New_h_pos,New_h_pos], [np.min(Der_flatten[t]), np.max(Der_flatten[t])])
        ax.plot([hm,hm], [np.min(Der_flatten[t]), np.max(Der_flatten[t])])
        ax.plot([hp,hp], [min(Der_flatten[t]), max(Der_flatten[t])])
        ax.set_title(t)
        plt.show()
        
    #     # #%%  
    #     # fig, ax=plt.subplots() 
    #     # ax.plot(t_list,H_list)
    #     # ax.fill_between(t_list, Hm, Hp, color='blue', alpha=0.3)
    #     # ax.set_xscale('log')
    #     # ax.set_yscale('log')
    #     # plt.show()
            


fig, ax=plt.subplots() 
ax.plot(t_list[:],H_list[:])
ax.fill_between(t_list[:], Hm[:], Hp[:], color='blue', alpha=0.3)
ax.set_xscale('log')
ax.set_yscale('log')
plt.show()
    





DF=pd.DataFrame()

DF['time(s)'] = t_list[0:]
DF['h(cm)'] = np.array(H_list[0:])/Scale
DF['h_moins(cm)'] = np.array(Hm[0:])/Scale
DF['h_plus(cm)'] = np.array(Hp[0:])/Scale






DF.to_csv('/Users/krishanbumma/Desktop/Manip09_10/09_10_manip1_2fps_cropht_mesureh.csv')
    