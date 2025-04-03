# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 17:09:53 2023

@author: PetitDell
"""

print('Cellule 1 executée')

import libtiepie
import numpy as np
import os
import time
import pandas as pd
from matplotlib.animation import FuncAnimation
import sys
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import keyboard 


libtiepie.network.auto_detect_enabled = True

# Search for devices:
libtiepie.device_list.update()

if len(libtiepie.device_list) > 0:
    print()
    print('[Info] TiePie disponibles:')

    for item in libtiepie.device_list:
        print('-Nom: ' + item.name)
        print('---Serial number  : ' + str(item.serial_number))
        print('---Available types: ' + libtiepie.device_type_str(item.types))

        if item.has_server:
            print('    Server         : ' + item.server.url + ' (' + item.server.name + ')')
        if not item.can_open(1):
            print("---[ERREUR] Impossible de charger l'oscilloscope\n~~~[Essayer de relancer le noyau]~~~")
            sys.exit()
        if not item.can_open(2):
            print("---[ERREUR] Impossible de charger le générateur")
            sys.exit()
else:
    print("[ERREUR] Pas d'oscilloscopes trouvés")
    sys.exit()
    

    
gen = None
###############################################################################
# #différentiation des TiePie
# if libtiepie.device_list[0].serial_number == 33184:
#     scp = libtiepie.device_list[0].open_oscilloscope() #ouveture oscilloscope
#     gen = libtiepie.device_list[0].open_generator() #ouverture generateur
#     #scp_2 = libtiepie.device_list[1].open_oscilloscope() #ouveture oscilloscope
# else :
#     scp = libtiepie.device_list[1].open_oscilloscope() #ouveture oscilloscope
#     gen = libtiepie.device_list[1].open_generator() #ouverture generateur
#     #scp_2 = libtiepie.device_list[0].open_oscilloscope() #ouveture oscilloscope

scp = libtiepie.device_list[0].open_oscilloscope() #ouveture oscilloscope
gen = libtiepie.device_list[0].open_generator() #ouverture generateur

ch1=scp.channels[0]




#%%
print('Cellule 2 executée')
def setGen( f, amplitude =  0.5 , signal_type=libtiepie.ST_SINE): 

 
    gen._set_signal_type(signal_type) #forme du signal généré
    gen._set_frequency(f) # Hz
    gen._set_amplitude(amplitude) #amplitude signal gen (Volt)
    gen._set_offset(0)#off signal générateur (Volt)
    gen._set_mode(libtiepie.GM_CONTINUOUS) #continuous

    gen._set_output_enable(1)
    gen.start()
#%%
print('Cellule 3 executée')
def getData(freq):
    fe = min(50*freq, 500e6)
    if fe==50*freq : 
        nbe = 2000
    else :
        nbe = 5000000
        
    
    scp._set_measure_mode(libtiepie.MM_STREAM)
    scp._set_sample_rate(fe) 
    scp._set_record_length(nbe) 
    Time = np.linspace(0,nbe/fe,nbe)

    for ch in scp.channels:
        # Enable channel to measure it:
        ch._set_enabled(1)
        # Set range:
        ch._set_range(2)
        # Set coupling:
        ch._set_coupling(libtiepie.CK_ACV)  # DC Volt
        
    # time.sleep(0.5)
    # Start measurement:
    scp.start()
    
    # Wait for measurement to complete:
    while not (scp._get_is_data_ready()):# or scp._get_is_data_overflow):
        # if scp._get_is_data_overflow():
        #     print('Data overflow before data ready')
        time.sleep(0.01)  # 10 ms delay, to save CPU time

            
    # time.sleep(0.5)
    
    # if scp._get_is_data_overflow():
    #     print('Data overflow!')
        
    # time.sleep(1)

    # Get data:
    data = scp.get_data()


    # Stop stream:
    scp.stop()
    
    if fe>2*freq: 
        shannon=True
    else : 
        shanon=False
    return Time, data, shannon



    
#%%
print('Cellule 4 executée')



def fitData(Time, data, frequence=10000, shannon=True) : 
    
    if shannon : 
        GenSignal = data[0]   # Le générateur est branché dans le channel 1
        UrSignal = data[1]    #Le pour diviseur de tension est branché en 2
        
        sineWave =  lambda x, A, phi: A*np.sin(x*2*np.pi*frequence + phi)
        ptGen,covGen = curve_fit(sineWave, Time, GenSignal, p0 = [0.5, 0], bounds = [[0,-np.pi],[1, np.pi]]) 
        ptUr,covUr = curve_fit(sineWave, Time, UrSignal, p0 = [0.1, 0], bounds = [[0,-np.pi],[1, np.pi]]) 
        
        FittedGen = [sineWave(Time[i], ptGen[0], ptGen[1]) for i in range(len(Time))]
        FittedUr = [sineWave(Time[i], ptUr[0], ptUr[1]) for i in range(len(Time))]
        
        
        # fig, ax = plt.subplots()
        # ax.scatter(Time, GenSignal, label='Gen signal', color = 'black', marker = 'x')
        # ax.plot(Time, FittedGen,  label='Gen fitted, A={}, Phi = {}'.format( ptGen[0],  ptGen[1]), color='black')
        
        # ax.scatter(Time, UrSignal, label='Ur signal', color = 'blue', marker = 's')
        # ax.plot(Time, FittedUr,  label='Ur fitted, A={}, Phi = {}'.format( ptUr[0],  ptUr[1]), color='blue')
        # ax.set_xlabel('time')
        # ax.set_ylabel('amplitude(V)')
        # plt.legend()
        # plt.plot()
        
        
        print('Gen fitted, A={}, Phi = {}'.format( ptGen[0],  ptGen[1]))
        print('Ur fitted, A={}, Phi = {}'.format( ptUr[0],  ptUr[1]))
               
        return  ptGen[0],  ptGen[1],  ptUr[0],  ptUr[1],  np.sqrt(covGen[0][0]),  np.sqrt(covGen[1][1]),  np.sqrt(covUr[0][0]),  np.sqrt(covUr[1][1])
    
    
    else : 
        GenSignal = data[0]   # Le générateur est branché dans le channel 1
        UrSignal = data[1]   #Le pour diviseur de tension est branché en 2
        
        return max(GenSignal), max(UrSignal), 0, 1, 0,0,0,0
        
#%%
print('Cellule 5 executée') 




def FreqSwipe():
    
    savePath = 'Path'
    
    
    Gen_amplitude = []
    Ur_amplitude = []

    Gen_phi= []
    Ur_phi= []
    
    eGen_amplitude = []
    eUr_amplitude = []

    eGen_phi= []
    eUr_phi= []

    i=0
    for frequence in Freq_list : 
        print('frequence ={}'.format(frequence))
        
        setGen(f=frequence)
        Time, data, shannon = getData(frequence)
        V0, Phi0, Ur, Phir, eV0, ePhi0, eUr, ePhir  =  fitData(Time, data, frequence, shannon)
        
        Gen_amplitude.append(V0)
        Ur_amplitude.append(Ur)
        Gen_phi.append(Phi0)
        Ur_phi.append(Phir)
        eGen_amplitude.append(eV0)
        eUr_amplitude.appende(Ur)
        eGen_phi.append(ePhi0)
        eUr_phi.append(ePhir)
        gen.stop()
        gen._set_output_enable(0)

        i+=1
    Gain_dB = (np.array(Ur_amplitude)/np.array(Gen_amplitude))
    Abs_phi = np.array(Ur_phi)-np.array(Gen_phi)
    Abs_phi=((Abs_phi+np.pi) % (2*np.pi))/(np.pi)-1
    
    # #Fait diagramme de bode
    # fig, ax = plt.subplots()
    # ax2=ax.twinx()
    # ax.scatter(Freq_list[:i+1], Gain_dB,marker='x', color='black')
    # ax2.scatter(Freq_list[:i+1], Abs_phi, marker='o', color='red')
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.set_title('Diagramme de Bode')
    # ax.set_ylabel('Gain (dB)', color='black')
    # ax2.set_ylabel('Phi/pi', color='red')
    # ax.set_xlabel('Freq (Hz)')
    # plt.plot()

        
    return Gain_dB, Abs_phi, Gen_amplitude, Ur_amplitude, Gen_phi, Ur_phi, eGen_amplitude, eUr_amplitude, eGen_phi, eUr_phi
        
        

def Measure(print_values=True): 
    frequence=5*10**3
    setGen(f=frequence)
    time.sleep(0.1)
    Time, data, shannon = getData(frequence)
    V0, Phi0, Ur, Phir, eV0, ePhi0, eUr, ePhir  =  fitData(Time, data, frequence, shannon)
    if print_values == True :
        print()
        print('Generateur = {} +- {}'.format(V0, eV0))
        print('Resistance = {} +- {}'.format(Ur, eUr))
    
    return V0, Ur, eV0, eUr



def CalcResCell(Urest,Ugent,R=998): 
    return R*(Ugent-Urest)/Urest



#D:/Krishan/TESTJustine

def Measure_time(tmax = None, savepath = '' ): 
    
    print('espace pour commencer, x pour arreter')
    
    #Déclanchement 
    while True:  # making a loop
        try:  # used try so that if user pressed other than the given key error will not be shown
            if keyboard.is_pressed(' '):  # if key 'q' is pressed 
                print("c'est parti!!!")
                print("")
                break  # finishing the loop
        except:
            break  # if user pressed a key other than the given key the loop will break
    
    
    X_pressed=False
    
    t0 = time.time()
    t=0
    Time=[]
    Ugen=[]
    Ures=[]
    eUgen=[]
    eUres=[]
    
             
    Rcell=[CalcResCell(Ures[i], Ugen[i]) for i in range(len(Ugen))]
    eRcell=[(5/998+eUres[i]/Ures[i]+eUgen[i]/Ugen[i])*Rcell[i] for i in range(len(Ugen))]
    Condcell=1/np.array(Rcell)
    eCondcell=[(5/998+eUres[i]/Ures[i]+eUgen[i]/Ugen[i])*Condcell[i] for i in range(len(Ugen))]
    
    
    if tmax!=None :

        while t<tmax and X_pressed==False: 
            time.sleep(0.5)
            V0, Ur, eV0, eUr = Measure(print_values=False)
            t=time.time()-t0
            Time.append(t)
            Ugen.append(V0)
            Ures.append(Ur)
            eUgen.append(eV0)
            eUres.append(eUr)
            
       
            print('measuring...')
        

            if keyboard.is_pressed('x'):  # if key 'q' is pressed 
                X_pressed=True
                print()
                print('stop avant la fin voulue')
                print("t=",t)
                print("")
                break  # finishing the loop
                
    
            
    else : 
        j=0

        while X_pressed==False: 
            time.sleep(0.5)
            V0, Ur, eV0, eUr = Measure(print_values=False)
            t=time.time()-t0
            Time.append(t)
            Ugen.append(V0)
            Ures.append(Ur)
            eUgen.append(eV0)
            eUres.append(eUr)
                

            if int(t//10) == j : 
                j+=1
                print(t)
            print('measuring...')
            if keyboard.is_pressed('x'):  # if key 'q' is pressed 
                X_pressed=True
                print()
                print('stop')
                print("t=",t)
                print("")
                break  # finishing the loopxxxxxxx

    
                 
    Rcell=[CalcResCell(Ures[i], Ugen[i]) for i in range(len(Ugen))]
    eRcell=[(5/998+eUres[i]/Ures[i]+eUgen[i]/Ugen[i])*Rcell[i] for i in range(len(Ugen))]
    Condcell=1/np.array(Rcell)
    eCondcell=[(5/998+eUres[i]/Ures[i]+eUgen[i]/Ugen[i])*Condcell[i] for i in range(len(Ugen))]
    
    


    d=dict()
    d['temps(s)']=Time
    d['U_generateur']=Ugen
    d['errU_generateur']=eUgen
    d['U_resistance']=Ures
    d['errU_resistance']=eUres
    d['R_cell(Ohm)']=Rcell
    d['errR_cell(Ohm)']=eRcell
    d['Cond_cell(Ohm^-1)']=Condcell
    d['errCond_cell(Ohm^-1)']=eCondcell
    
    
    
    #saving data
    DF=pd.DataFrame(d)
        
    csv=open(savepath, 'w')
    csv.close()
    
    DF.to_csv(savepath, mode='w')

    
    
    fig, ax=plt.subplots()
    ax2=ax.twinx()
    ax.plot(Time, Ugen, color='black', label='U_gen')
    ax.errorbar(Time, Ugen, yerr=eUgen, color='black')
    ax.plot(Time, Ures, color='blue', label='U_R')
    ax.errorbar(Time, Ures, yerr=eUres, color='blue')
    ax2.plot(Time, Condcell, color='red', label='Conductivité de cellule')
    ax2.errorbar(Time, Condcell, yerr=eCondcell, color='red')
    ax.set_xlabel('Time(s)')
    ax.set_ylabel('volts')
    ax2.set_ylabel('Ohm^-1')
    plt.legend()
    plt.show()
    
    
    
    return DF
    




def Measure_withPhi(): 
    save_file = open('Etallonage_cellule_7_'+str(time.time())+'.csv', 'w')
    save_file.write('Phi,U_generator,eU_generator,U_resistance,eUresistance,R_cell,eR_cell,Cond_cell,eCond_cell\n')
    i=0
    
    X_pressed=False
    while X_pressed==False:  # making a loop
        time.sleep(0.1)
        print('press n for next, x to exit')
        print()
        if keyboard.is_pressed('x'):  # if key 'x' is pressed 
            X_pressed=True  # finishing the loop
            break
        else : 
            while X_pressed==False:  # making a loop
                time.sleep(0.1)
                
                
                
                
                
                if  keyboard.is_pressed('n'): # if key 'n' is pressed   # finishing the loop
                    i+=1
                    print("Measurement number {}".format(i))
                    V0, Ur, eV0, eUr = Measure(print_values=True)
                    Rcell=CalcResCell(Ur, V0)
                    eRcell=(5/998+eUr/Ur+eV0/V0)*Rcell
                    Condcell=1/Rcell
                    eCondcell=Condcell*eRcell/Rcell
    
                    Phi_l=i
                    
                    save_file.write(str(Phi_l)+',')
                    save_file.write(str(V0)+',')
                    save_file.write(str(eV0)+',')
                    save_file.write(str(Ur)+',')
                    save_file.write(str(eUr)+',')
                    save_file.write(str(Rcell)+',')
                    save_file.write(str(eRcell)+',')
                    save_file.write(str(Condcell)+',')
                    save_file.write(str(eCondcell)+',')
                    save_file.write('\n')
                    print()
                    print('press n for next, x to exit')
                    print()
                    
                
                elif  keyboard.is_pressed('x'): 
                    X_pressed=True
                    break


    save_file.close()
    

#%%

# temperature_t_csv=pd.read_csv('D:/Krishan/3eme annee/Test mesure Phi z/MesT.txt', delimiter='	')
# T=temperature_t_csv['Temp.']
# T=[float(T[i]) for i in range(len(T))]
# Cond=DATA_mes_Temp['Cond_cell(Ohm^-1)']
# time=DATA_mes_Temp['temps(s)']



# fig, ax=plt.subplots()
# ax.plot(T)
# plt.show()


# fig, ax=plt.subplots()
# ax.plot(time, Cond)
# plt.show()




