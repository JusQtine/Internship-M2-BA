#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 16:03:37 2024

@author: krishanbumma
"""





import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as scpo 
from scipy.signal import savgol_filter, butter
from scipy import signal
from scipy.ndimage import  gaussian_filter
from matplotlib import colormaps as cm
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# from Calcul_Deff import Deff_foam
# import PlotModelsConductivite as PMC
import os
from math import exp, erf,sqrt, pi,trunc
import scipy as scp
import random as rd

#%%

#os.environ['PATH'] = '/Users/krishanbumma/anaconda3/bin:/Users/krishanbumma/anaconda3/condabin:/opt/local/bin:/opt/local/sbin:/usr/local/bin:/System/Cryptexes/App/usr/bin:/usr/bin:/bin:/usr/sbin:/sbin:/Library/TeX/texbin:/opt/X11/bin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/local/bin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/bin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/appleinternal/bin'
#plt.style.use('/Users/krishanbumma/Desktop/Redaction/utile/Figure_largeur_moyenne.mplstyle')

#save_path = '/Users/krishanbumma/Desktop/Mesure Phi(z)/Codes_traitement_manips/Plot Redaction/'


#%%

Alpha_10 = 0.028411203324717194 
EAlpha10 = 0.00064440501313297
sigma_0 =  0






def affine(x,a,b): 
    return a*x+b
#Cellule 1




def fitting_function(t, D,B,C,tau1):
    U=np.sqrt(D*t)
    V=B*t+C
    f = 1/(1 + np.exp(-(t/tau1)))
    return f*U + (1-f)*V


def fitting_function_Hmax(t, D,B,C):
    
    return D/(t+B)+C



def fitting_function_phil(t,A,B,C,D,E) : 
    return A+t*B+t**2*C+t**3*D+t**4*E







def Phi(lam, T):
    B=52.25980723655028
   
    Cond = lam/(sigma_0*(1+Alpha_10*(T-10)))
    Phi = 3*Cond*(3+B*Cond)/(3 + (9+2*B)*Cond + (-3+B)*Cond**2)
    
    return Phi

    
def Phip(lam, T):
    B=32.08014816371646
    Cond = lam/(sigma_0*(1+Alpha_10*(T-10)))
    Phi = 3*Cond*(3+B*Cond)/(3 + (9+2*B)*Cond + (-3+B)*Cond**2)
    
    return Phi

def Phim(lam, T):
    B=94.64352393039862
    Cond = lam/(sigma_0*(1+Alpha_10*(T-10)))
    Phi = 3*Cond*(3+B*Cond)/(3 + (9+2*B)*Cond + (-3+B)*Cond**2)
    
    return Phi



print("check")

Data_cond = pd.read_csv('D:/Justine/11_10/11_10_etalonnage/11_10_etalonnage.csv')
Data_T  =pd.read_csv('D:/Justine/11_10/11_10_etalonnage/11_10_etalonnage_temperature.csv')

#Data_cond = pd.read_csv('/Users/krishanbumma/Desktop/Manip09_10/09_10_manip0_etalonnage.csv')
#Data_T  =pd.read_csv('/Users/krishanbumma/Desktop/Manip09_10/09_10_manip0_temperaturesetalonnage.csv')

print("check")

Cond = Data_cond['Cond_cell(Ohm^-1)']
eCond = Data_cond['errCond_cell(Ohm^-1)']
Cond_time=Data_cond['temps(s)']
T_t = Data_T["Temp."]
t_T = np.array([i for i in range(len(T_t))])

print("check")

sigma_ref = np.average(Cond)
T_ref = np.average(T_t)


sigma_0 = sigma_ref/(1+Alpha_10*(T_ref-10))




#%%


#%%

#Traitement de la manip de solidification


nom_manip = '11_10_manip2'

Data_cond = pd.read_csv('D:/Justine/11_10/11_10_manip2/11_10_manip2.csv')
Data_cond = Data_cond.iloc[:1000] #Je ne garde que les 800s premières seconde (on a 2fps)

Data_T  =pd.read_csv('D:/Justine/11_10/11_10_manip2/11_10_manip2_temperature.csv')
Data_T = Data_T.iloc[:1000]

Data_h = pd.read_csv('D:/Justine/11_10/11_10_manip2/11_10_manip2_2fps_tif_stack_crop_bas_mesureh.csv')
Data_h = Data_h.iloc[:1000]

Data_Hmax = pd.read_csv('D:/Justine/11_10/11_10_manip2/11_10_manip2_haut_mousse.csv')

hauteur_cell_pix = 578

#Data_cond = pd.read_csv('/Users/krishanbumma/Desktop/Manip09_10/09_10_manip2.csv')
#Data_T  =pd.read_csv('/Users/krishanbumma/Desktop/Manip09_10/09_10_manip2_temperature.csv')
#Data_h = pd.read_csv('/Users/krishanbumma/Desktop/Manip09_10/09_10_manip2_2fps_cropht_mesureh.csv')
#Data_Hmax = pd.read_csv('/Users/krishanbumma/Desktop/Manip09_10/09_10_manip2_2fps_MesureHmax.csv')

# Data_Manips = pd.read_csv('/Users/krishanbumma/Desktop/Mesure Phi(z)/RecapMesurePhi(z)_csv.csv')
# Data_Manips=Data_Manips.loc[Data_Manips['Manip']=='Manip7_solidification_2fps_cropped.tif']

# Phi0=Data_Manips['Phi_0']
# Ts=Data_Manips['T(t=0s)']
# scale=float(Data_Manips['Scale (pix/cm)'])

Phi0 = 0.1
Ts = -25
scale = hauteur_cell_pix/3 #pix/cm






Cond = np.array(Data_cond['Cond_cell(Ohm^-1)'])
eCond = np.array(Data_cond['errCond_cell(Ohm^-1)'])
Cond_time=np.array(Data_cond['temps(s)'])
T_t = Data_T["Temp."]
t_T = np.array([i for i in range(len(T_t))])

t_h = Data_h['time(s)']
h_t = Data_h['h(cm)']
hm_t = Data_h['h_moins(cm)']
hp_t = Data_h['h_plus(cm)']

Cond_time =  Cond_time[np.where(Cond_time<np.max(t_h))]
eCond = eCond[np.where(Cond_time<np.max(t_h))]
Cond=Cond[np.where(Cond_time<np.max(t_h))]


t_Hmax=Data_Hmax['Slice']
Hmax_t= ((np.ones(len(Data_Hmax))*hauteur_cell_pix) - Data_Hmax['Y'])/scale

#t_Hmax = np.array([1,242,752,1021,1739,1757,2417,2555,3249,3331,3598])
#Hmax_t = (np.array([588-10.500,588-7.000,588-52.500,588-50.500,588-65.500,588-67.500,588-70.500,588-74.500,588-73.500,588-75.500,588-75.500])*3)/588



Hmax_t_interp=np.interp(Cond_time, t_Hmax, Hmax_t)




T_t_interp = np.interp(Cond_time, t_T, T_t)
h_t_interp = np.interp(Cond_time, t_h, h_t)
hp_t_interp = np.interp(Cond_time, t_h, hp_t)
hm_t_interp = np.interp(Cond_time, t_h, hm_t)
# h_t_theorique = np.sqrt(Deff_foam(float(Phi0), float(Ts), 20)*Cond_time)


Phi_t=[Phi(Cond[i], T_t_interp[i]) for i in range(len(Cond))]
Phip_t=[Phip(Cond[i], T_t_interp[i]) for i in range(len(Cond))]           
Phim_t=[Phim(Cond[i], T_t_interp[i]) for i in range(len(Cond))]          



# Phi_t_lisse = savgol_filter(Phi_t, 25, 1)
# h_t_interp = savgol_filter(h_t_interp, 25, 1)


Phi_t_lisse = gaussian_filter(Phi_t, 1)
# h_t_interp = gaussian_filter(h_t_interp, 10)
ptphil, covphil = scpo.curve_fit(fitting_function_phil, Cond_time, Phi_t)
Phi_t_lisse = [fitting_function_phil(Cond_time[i],*ptphil) for i in range(len(Cond_time))]



t_phi_ice = []
Phi_ice = []
Phi_ice_up = []
Phi_ice_down = []





pt,cov = scpo.curve_fit(fitting_function, t_h, h_t, p0=[0.0001, 1,1,100])
h_t_interp = [fitting_function(t,*pt) for t in Cond_time]
pt,cov = scpo.curve_fit(fitting_function, t_h, hp_t, p0=[0.0001, 1,1,100])
hp_t_interp = [fitting_function(t,*pt) for t in Cond_time]
pt,cov = scpo.curve_fit(fitting_function, t_h, hm_t, p0=[0.0001, 1,1,100])
hm_t_interp = [fitting_function(t,*pt) for t in Cond_time]



#%%

###Partie modifiée pour ne plus avoir le problème "out of range"

density_ratio = 0.91
h_t_phi_ice = np.interp(t_phi_ice, Cond_time, h_t_interp)
liste_dh = []

i = 0
while i < len(Cond_time) - 2:
    dh = h_t_interp[i] - h_t_interp[i]
    j = 0
    while dh <= 0.0 and (i + j) < len(Cond_time) - 1:
        j += 1
        if (i + j) < len(Cond_time):
            dt = Cond_time[i + j] - Cond_time[i]
            dh = h_t_interp[i + j] - h_t_interp[i]
        else:
            break  # Sortie si on dépasse les limites

    if j < len(Cond_time) - i:
        t_phi_ice.append((Cond_time[i + j] + Cond_time[i]) / 2)
        Phi_Flux = (Phi_t[i] - (Phi_t_lisse[i + j] - Phi_t_lisse[i]) * (Hmax_t_interp[i] - h_t_interp[i]) / dh) / density_ratio
        Phi_ice.append(0)
        liste_dh.append(dh)

        k = 1
        Phi_Flux = Phi_Flux * dh
        Phi_k_ap = min(1, Phi_Flux / dh)
        Delta_Phi_k_av_x_dh = Phi_k_ap * dh
        Phi_Flux = Phi_Flux - Delta_Phi_k_av_x_dh
        Phi_ice[-k] = Phi_k_ap

        while Phi_Flux > 0.00 and k < len(Phi_ice):
            k += 1
            Phi_k_av = Phi_ice[-k]
            Phi_k_ap = min(1, Phi_k_av + Phi_Flux / liste_dh[-k])
            Delta_Phi_k_av_x_dh = Phi_k_ap * liste_dh[-k] - Phi_k_av * liste_dh[-k]
            Phi_ice[-k] = Phi_k_ap
            Phi_Flux = Phi_Flux - Delta_Phi_k_av_x_dh
            
    i += j

# Deuxième boucle
i = 0
while i < len(Cond_time) - 1:
    dh = hm_t_interp[i] - hm_t_interp[i]
    j = 0
    while dh <= 0.0 and (i + j) < len(Cond_time) - 1:
        j += 1
        if (i + j) < len(Cond_time):
            dt = Cond_time[i + j] - Cond_time[i]
            dh = hm_t_interp[i + j] - hm_t_interp[i]
        else:
            break  # Sortie si on dépasse les limites

    if j < len(Cond_time) - i:
        Phi_Flux = (Phip_t[i] - (Phi_t_lisse[i + j] - Phi_t_lisse[i]) * (Hmax_t_interp[i] - hm_t_interp[i]) / dh) / density_ratio
        Phi_ice_up.append(0)
        liste_dh.append(dh)

        k = 1
        Phi_Flux = Phi_Flux * dh
        Phi_k_ap = min(1, Phi_Flux / dh)
        Delta_Phi_k_av_x_dh = Phi_k_ap * dh
        Phi_Flux = Phi_Flux - Delta_Phi_k_av_x_dh
        Phi_ice_up[-k] = Phi_k_ap

        while Phi_Flux > 0.00 and k < len(Phi_ice_up):
            k += 1
            Phi_k_av = Phi_ice_up[-k]
            Phi_k_ap = min(1, Phi_k_av + Phi_Flux / liste_dh[-k])
            Delta_Phi_k_av_x_dh = Phi_k_ap * liste_dh[-k] - Phi_k_av * liste_dh[-k]
            Phi_ice_up[-k] = Phi_k_ap
            Phi_Flux = Phi_Flux - Delta_Phi_k_av_x_dh
            
    i += j

# Troisième boucle
i = 0
while i < len(Cond_time) - 1:
    dh = hp_t_interp[i] - hp_t_interp[i]
    j = 0
    while dh <= 0.0 and (i + j) < len(Cond_time) - 1:
        j += 1
        if (i + j) < len(Cond_time):
            dt = Cond_time[i + j] - Cond_time[i]
            dh = hp_t_interp[i + j] - hp_t_interp[i]
        else:
            break  # Sortie si on dépasse les limites

    if j < len(Cond_time) - i:
        Phi_Flux = (Phim_t[i] - (Phi_t_lisse[i + j] - Phi_t_lisse[i]) * (Hmax_t_interp[i] - hp_t_interp[i]) / dh) / density_ratio
        Phi_ice_down.append(0)
        liste_dh.append(dh)

        k = 1
        Phi_Flux = Phi_Flux * dh
        Phi_k_ap = min(1, Phi_Flux / dh)
        Delta_Phi_k_av_x_dh = Phi_k_ap * dh
        Phi_Flux = Phi_Flux - Delta_Phi_k_av_x_dh
        Phi_ice_down[-k] = Phi_k_ap

        while Phi_Flux > 0.00 and k < len(Phi_ice_down):
            k += 1
            Phi_k_av = Phi_ice_down[-k]
            Phi_k_ap = min(1, Phi_k_av + Phi_Flux / liste_dh[-k])
            Delta_Phi_k_av_x_dh = Phi_k_ap * liste_dh[-k] - Phi_k_av * liste_dh[-k]
            Phi_ice_down[-k] = Phi_k_ap
            Phi_Flux = Phi_Flux - Delta_Phi_k_av_x_dh
            
    i += j

h_t_phi_ice = np.interp(t_phi_ice, Cond_time, h_t_interp)

h_t_phi_ice = h_t_phi_ice[:-1]
Phi_ice = Phi_ice[:-1]
Phi_ice_down = Phi_ice_down[:-1]
Phi_ice_up = Phi_ice_up[:-1]

Delta_h = np.array(h_t_phi_ice[1:] - h_t_phi_ice[:-1])
integrale1_Phi = np.sum(Delta_h * (np.array(Phi_ice[:-1]) + np.array(Phi_ice[1:])) / 2)
print("integrale_1 = " + str(integrale1_Phi))
print("Phi_ice_moyen = " + str(integrale1_Phi / np.sum(Delta_h)))


#%%

"""
Ici on ne refait pas de fit sur Phip_t et Phi_m_t parce qu'ils sont quasiment parallèle

"""
"""
density_ratio=0.91
h_t_phi_ice = np.interp(t_phi_ice,Cond_time, h_t_interp)
liste_dh = []
i=0
while i<len(Cond_time)-2 : 
    
            dt = Cond_time[i] - Cond_time[i]
            dh = h_t_interp[i]-h_t_interp[i]
            j=0
            while dh<=0.0:     
                j+=1
                print(j)
                dt = Cond_time[i+j] - Cond_time[i]
                dh = h_t_interp[i+j]-h_t_interp[i]
            t_phi_ice.append((Cond_time[i+j]+Cond_time[i])/2)
            Phi_Flux = (Phi_t[i]-(Phi_t_lisse[i+j]-Phi_t_lisse[i])*(Hmax_t_interp[i]-h_t_interp[i])/dh)/density_ratio
            Phi_ice.append(0)
            liste_dh.append(dh)
            k=1
            Phi_Flux=Phi_Flux*dh
            Phi_k_ap = min(1, Phi_Flux/dh)
            Delta_Phi_k_av_x_dh = Phi_k_ap*dh
            Phi_Flux = Phi_Flux-Delta_Phi_k_av_x_dh
            Phi_ice[-k] =  Phi_k_ap 
           
            while Phi_Flux>0.00 and k<len(Phi_ice):  
                k+=1
                Phi_k_av = Phi_ice[-k]
                Phi_k_ap = min(1, Phi_k_av+Phi_Flux/liste_dh[-k])
                Delta_Phi_k_av_x_dh =Phi_k_ap*liste_dh[-k]-Phi_k_av*liste_dh[-k]
                Phi_ice[-k] = Phi_k_ap
                Phi_Flux = Phi_Flux - Delta_Phi_k_av_x_dh
            i+=j

i=0
while i<len(Cond_time)-1 : 
    
            dt = Cond_time[i] - Cond_time[i]
            dh = hm_t_interp[i]-hm_t_interp[i]
            j=0
            while dh<=0.0:      
                j+=1
                dt = Cond_time[i+j] - Cond_time[i]
                dh = hm_t_interp[i+j]-hm_t_interp[i]
            Phi_Flux = (Phip_t[i]-(Phi_t_lisse[i+j]-Phi_t_lisse[i])*(Hmax_t_interp[i]-hm_t_interp[i])/dh)/density_ratio

        
            Phi_ice_up.append(0)
            liste_dh.append(dh)
            
            k=1
            Phi_Flux=Phi_Flux*dh
            Phi_k_ap = min(1, Phi_Flux/dh)
            # print('here')
            # print(Phi_k_ap )
            Delta_Phi_k_av_x_dh = Phi_k_ap*dh
            Phi_Flux = Phi_Flux-Delta_Phi_k_av_x_dh
            Phi_ice_up[-k] =  Phi_k_ap 
           
            while Phi_Flux>0.00 and k<len(Phi_ice_up):  
                k+=1
                # print("boucle entrée")
                # print(k)
                Phi_k_av = Phi_ice_up[-k]
                Phi_k_ap = min(1, Phi_k_av+Phi_Flux/liste_dh[-k])
                Delta_Phi_k_av_x_dh =Phi_k_ap*liste_dh[-k]-Phi_k_av*liste_dh[-k]
                Phi_ice_up[-k] = Phi_k_ap
                Phi_Flux = Phi_Flux - Delta_Phi_k_av_x_dh
            i+=j
     
        
i=0
while i<len(Cond_time)-1 : 
    
            dt = Cond_time[i] - Cond_time[i]
            dh = hp_t_interp[i]-hp_t_interp[i]
            j=0
            while dh<=0.0: 
                
                j+=1
                dt = Cond_time[i+j] - Cond_time[i]
                dh = hp_t_interp[i+j]-hp_t_interp[i]


            Phi_Flux = (Phim_t[i]-(Phi_t_lisse[i+j]-Phi_t_lisse[i])*(Hmax_t_interp[i]-hp_t_interp[i])/dh)/density_ratio
        
            Phi_ice_down.append(0)
            liste_dh.append(dh)
            
            k=1
            Phi_Flux=Phi_Flux*dh
            Phi_k_ap = min(1, Phi_Flux/dh)
            # print('here')
            # print(Phi_k_ap )
            Delta_Phi_k_av_x_dh = Phi_k_ap*dh
            Phi_Flux = Phi_Flux-Delta_Phi_k_av_x_dh
            Phi_ice_down[-k] =  Phi_k_ap 
           
            while Phi_Flux>0.00 and k<len(Phi_ice_down):  
                k+=1
                # print("boucle entrée")
                # print(k)
                Phi_k_av = Phi_ice_down[-k]
                Phi_k_ap = min(1, Phi_k_av+Phi_Flux/liste_dh[-k])
                Delta_Phi_k_av_x_dh =Phi_k_ap*liste_dh[-k]-Phi_k_av*liste_dh[-k]
                Phi_ice_down[-k] = Phi_k_ap
                Phi_Flux = Phi_Flux - Delta_Phi_k_av_x_dh
            i+=j
     

h_t_phi_ice = np.interp(t_phi_ice,Cond_time, h_t_interp)

h_t_phi_ice=h_t_phi_ice[:-1]
Phi_ice=Phi_ice[:-1]
Phi_ice_down=Phi_ice_down[:-1]
Phi_ice_up=Phi_ice_up[:-1]




Delta_h=np.array(h_t_phi_ice[1:]-h_t_phi_ice[:-1])
integrale1_Phi=np.sum(Delta_h*(np.array(Phi_ice[:-1])+np.array(Phi_ice[1:]))/2)
print("integrale_1 =   " + str(integrale1_Phi))
print("Phi_ice_moyen =   " + str(integrale1_Phi/np.sum(Delta_h)))
"""

#%%





#Regularisation de Phiice(z)

"""
Ici on va prendre h_t_phi_ice
On le passe en m
On prolonge Phi_ice jusq-a zero
"""

h_t_phi_ice_m_regul = [0]
Phi_ice_regul = [Phi_ice[0]]
Phi_ice_regul_croissant = [Phi_ice[0]]
temp = np.array(h_t_phi_ice)/100
delta_z = 0.0001 
while h_t_phi_ice_m_regul[-1]<temp[0] - delta_z : 
    h_t_phi_ice_m_regul.append( h_t_phi_ice_m_regul[-1]+delta_z)
    Phi_ice_regul.append(Phi_ice[0])
    Phi_ice_regul_croissant.append(Phi_ice[0])
    
for i in range(len(h_t_phi_ice)): 
    h_t_phi_ice_m_regul.append( temp[i])
    Phi_ice_regul.append(Phi_ice[i])
    if i==0 : 
        Phi_ice_regul_croissant.append(Phi_ice[0])
    else : 
        Phi_ice_regul_croissant.append(np.max(Phi_ice[:i])) 
    
# plt.figure()
# plt.plot(Phi_ice_regul, h_t_phi_ice_m_regul)
# plt.show()






"""

Ici calcul d'incertitude de Phi_ice

avec la méthode donnée par alexandre

"""


Delta_phi_ice = 0
h_i=0
H_i =0
dh =0
dH =0
dPhi_l = 0 
delta_Phil = 0
A=0
B=0
count=0
for i in range(len(Cond_time)): 
    if  h_t_interp[i]-h_t_interp[i-1]>0 : 
        count+=1
        h_i+= h_t_interp[i]
        H_i += Hmax_t_interp[i]
        dh += h_t_interp[i]-h_t_interp[i-1]
        dH += Hmax_t_interp[i]-Hmax_t_interp[i-1]
        dPhi_l += Phi_t[i]-Phi_t[i-1]
        delta_Phil  += 0.008
        A += np.abs(1.1*(dh-dH)/dh)
        B += np.abs(1.1*(dPhi_l)/dh)
        

h_i= h_i/count
H_i = H_i/count
dh = dh/count
dH =dH/count
dPhi_l =dPhi_l/count
delta_Phil =delta_Phil/count
A=A/count
B=B/count

Delta_phi_ice =2*( A*delta_Phil + B*(0.3) )



# fig = plt.figure()
# gs = gridspec.GridSpec(1, 2)

# ax1 = plt.subplot(gs[0, 0])
# ax2 = plt.subplot(gs[0, 1])

# cmap = cm['gnuplot']
# sigma_scatter = ax1.scatter(Cond_time, Cond, c=T_t_interp,vmin=5, vmax=20,s=1, cmap=cmap, label='$\sigma_{cell}')
# ax1.set_xlabel('temps(s)')
# ax1.set_ylabel('$\sigma$ (Ohm$^{-1}$)')
# ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

# ax2.plot([0, max(Cond_time)],[Phi0, Phi0], label = '$\phi_{l,0}$', color = 'orange')
# ax2.plot(Cond_time, Phi_t, color='green', label='$\phi_l(t)$')
# ax2.fill_between(Cond_time,Phip_t,Phim_t, color='green',alpha=0.3)
# ax2.set_xlabel('temps (s)')
# ax2.set_ylabel('$\phi$')
# # ax2.plot(t_phi_ice, Phi_ice, label='$\phi_{ice}(t)$', c='blue')
# # ax2.fill_between(t_phi_ice,Phi_ice_up,Phi_ice_down, color='blue',alpha=0.3)
# plt.legend()
# fig.tight_layout()
# cbaxes = fig.add_axes([0.35, 0.8, 0.1, 0.02]) 
# plt.colorbar(sigma_scatter, cax=cbaxes,cmap=cmap,orientation='horizontal', label='T $(^\circ C)$')
# plt.legend()
# # plt.savefig('/Users/krishanbumma/Desktop/Mesure Phi(z)/Manips/Plots/2023-12-22_Phi5_Manip8_solidification_liquide_fraction.pdf')
# plt.show()



fig = plt.figure()
gs = gridspec.GridSpec(2, 2, height_ratios=[0.2,0.8], width_ratios=[0.6, 0.4])
ax = plt.subplot(gs[1, 0])
ax2 = plt.subplot(gs[1, 1])
ax1_top =  plt.subplot(gs[0, 0])
ax2_top =  plt.subplot(gs[0, 0])

ax1_top.spines.bottom.set_visible(False)
ax.spines.top.set_visible(False)
ax1_top.xaxis.tick_top()
ax1_top.tick_params(labeltop=False)  # don't put tick labels at the top
ax.xaxis.tick_bottom()

ax2_top.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax2_top.xaxis.tick_top()
ax2_top.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

ax.plot(Cond_time, h_t_interp, color='red', label='h(t)')
ax.plot(t_h, h_t, color='yellow')
ax.legend()
# ax.plot(Cond_time, h_t_theorique, c='black', label='$\sqrt{D_{eff}t}$')

ax.fill_between(t_h, hp_t, hm_t, color='red', alpha=0.3, lw=0)


ax.tick_params(axis='y', which='both', left=True, right=True, labelleft=True)
ax.set_xlabel('temps(s)')
ax.set_ylabel('Z(cm)')


ax2.plot(Phi_ice_regul, np.array(h_t_phi_ice_m_regul)*100, label="$\phi_{ice}(z)$", color="blue")
ax2.fill_betweenx(np.array(h_t_phi_ice_m_regul)*100, np.array(Phi_ice_regul)+Delta_phi_ice, np.array(Phi_ice_regul)-Delta_phi_ice, color='blue', alpha=0.3, lw=0)
ax2.set_xlabel('$\phi_{ice}(z)$', color='black')
ax2.legend()
ax2.set_xlim(0, np.max(Phi_ice)*1.2)
# Set the y-axis limits to be the maximum and minimum of both datasets
y_min = 0

y_max = 0.8 #max(max(h_t_interp), max(h_t_phi_ice))*1.1
ax.set_ylim(y_min, y_max)
ax2.set_ylim(y_min, y_max)
# ax.yaxis.set_label_coords(-0.1,1)


ax1_top.plot(Cond_time, Hmax_t_interp, c='red')
ax1_top.set_ylim(2.4, 3.25)
ax1.legend()


ax2.tick_params(axis='y', which='both', left=True, labelleft=True, pad=15)
plt.subplots_adjust(wspace=0.05)
fig.tight_layout()
plt.title("Hauteur front et densité de mousse gelée {} 800sec".format(nom_manip))
plt.legend()

# plt.savefig('/Users/krishanbumma/Desktop/Mesure Phi(z)/Manips/Plots/2023-12-22_Phi5_Manip8_solidification_Phiice_ht.pdf')
plt.show()

#%%

#plt.plot(Cond_time,Cond, label="$\phi_{ice}(z)$", color="blue")
#plt.xlabel('time')
#plt.ylabel('cond')


#plt.plot(Cond_time,Phi_t_lisse, label="$\phi_{ice}(z)$", color="blue")
#plt.xlabel('time') 


plt.plot(t_Hmax,Hmax_t)
