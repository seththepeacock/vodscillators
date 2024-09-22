#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXplotSOAEwfP6.py

Purpose: (Python) Analysis of SOAE waveform data, including 
novel analysis of phase info to extract measure of phase coherence

To Do
+ Modify coherogram calc. by looking at phase diff before and after and then
averaging those two
+ Use updated code to test re EXtestPhaseCoherence2.py

# ==== Notes ====
o "phase coherence" = vector strength (as defined by V&D08 eq.20) 
o computed several different ways
1. time-delayed (tdPC): for a given freq. bin, ref = same bin for previous 
    time buffer
2. nearest-neighbor (nnPC): ref. = neighboring (lower) freq. bin
3. (re coherogram)
o NOTE: keep in mind that vector strength is a statistical quantity and that 
calcs. such as that directly below ill just yield unity.... (as expected!)
vsI= np.sqrt(np.sin(phaseDIFF2)**2 + np.cos(phaseDIFF2)**2)
 ---
 o determines max number of avgs possible for averaging (M)
 o re time-delayed phase coherence, Npts=512 seems to work for anoles (& owl?),
 and 2048 for humans
 ---
 o allows data to be read in as .mat file or .txt file
 o when converting txt files to .mat (via Matlab; roughly halves 
   the filesize), in Matlab type:
    > wf=load('fileName.txt');
    > save('fileName.mat','wf')
o tree cricket files are not SOAE (i.e., acoustic), but vibrometry where
the first column is time [s] and the second measured spontaneous 
oscillation velocity [m/s]; SR is determined directly from 1st column
o can use on human model data from Vaclav via > fileN = 'Model/MCIn1Y0.mat'
(seems to work fine, though I don't have a control for the noise only)
o can use on V&D data (see ) via "README re V&D sims by CB.txt"
> fileN = 'Model/coherenceD1summed.txt' (note SR=128000)
--> sans noise, coherence high about peaks and drops off between valleys
(but stays around 0.5 for higher freqs)
--- re FIG creation
o For coherogram, try using TH21RearwaveformSOAE and Npts=256*4 via
the imaginary part of VS (yields nice horizontal banding)
--- re sample rate
o 44100 Hz, except Groeningen is 48000; cricket SR determ. below, Vaclav model
data is 40000 Hz (coded in below) and V&D model is 128000

      
Created on Mon May 16 10:59:20 2022
@author: CB
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from numpy.fft import rfft
from numpy.fft import irfft

# -------------------------------- 
root= "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\SOAE Data\\"         # root path to file
#fileN = 'anole_ACsb32learSOAEwfG1.mat'   # file name
fileN = 'TH14RearwaveformSOAE.mat'   # file name
#fileN = 'cricket_177.txt'
# fileN = 'Model/coherenceD1summed.txt'


Npts= 256*8;     # length of fft window (# of points) [should ideally be 2^N]
                   # [time window will be the same length]
#mark= '.'  # marker style ('.' for dots, '' for none)
markB= 1  # boolean to turn off markers for plots (0=no markers) {1}
SR= 44100;         # sample rate (see above) [Hz] {default = 44100 Hz}
fPlot= [0.2,6]  # freq. limits for plotting [kHz] {[0,6]}
cPlot= [0,1]  # coherence vert. plot limits {[0,1]}
magL=[-1,1]     # mag limits for plotting {[-5,5]}
downSample = 0  # boolean to downsample the data by 1/2
# -------------------------------- 

# ==== bookeeping I (re file loading)
fname = os.path.join(root,fileN)
if (fileN[-3:]=='mat' and fileN[0:5]!='Model'):   # load in data as a .mat file
    data = scipy.io.loadmat(fname)  # loading in a .mat file
    wf= data['wf']   # grab actual data
elif(fileN[-3:]=='mat' and fileN[0:8]=='Model/MC'):  # load Vaclav's model
    data = scipy.io.loadmat(fname)
    wf= data['oae']  # can also run the 'doae'
    SR= 40000  # SR specified by Vaclav
else:   # load in as a .txt file
    wf = np.loadtxt(fname)  # loading in a .xtt file
  # --- markers for plotting (if used)  
if markB==0:
    markA=''
    markB=''
else:
     markA='.'
     markB='+'   
    
# --- deal w/ cricket file structure (and SR)
# NOTE: files from Natasha contain this header (which is deleted in ./Data/x.txt))
#Source File Name:	F177_LL_FFT_1.5 to 30 khz SOAE 1255_ 23.4Amb_23.74Pla
#Signal:	Time - Vib Velocity - Samples
#Time	Time Signal
#[ s ]	[ m/s ]
#if (wf.shape[1]>1):
if (fileN[0:7]=='cricket'):
    SR= round(1/np.mean(np.diff(wf[:,0])))  # use first column to determine SR
    wf= wf[:,1]  # grab second column
    
# ==== downsample by 1/2?
if (downSample==1):
    wf= wf[1::2]  # skip every other point
    SR= SR/2
    
# --- determine numb. of segments for spectral averaging 
# (and use as much wf as possible)
M= int(np.floor(len(wf)/Npts))  # numb. of time segments
print(f'# of avgs = {str(M-1)} ')
print(f'Delay window length = {1000*Npts/SR} ms')
# --- allocate some buffers
storeM= np.empty([int(Npts/2+1),M]) # store away spectral magnitudes
storeP= np.empty([int(Npts/2+1),M])  # store away spectral phases
storeWF= np.empty([int(Npts),M])  # waveform segments (for time-averaging)
storePd2= np.empty([int(Npts/2+1),M-2])  # smaller buffer for phase diffs
storePd= np.empty([int(Npts/2),M])  # phase diff re lower freq bin
storeWFcorr= np.empty([int(Npts),M-1])  # phase-corrected wf
storeVS= np.empty([int(Npts/2+1),M-2])  # time-delay coherence (for coherogram)
storeT= np.empty([M-2])  # time array for spectrograms

# ==== bookeeping II
Nt= len(wf)  # total numb. of time points
t= np.linspace(0,(Nt-1)/SR,Nt)   # time array
df = SR/Npts;   # freq bin width
freq= np.arange(0,(Npts+1)/2,1)    # create a freq. array (for FFT bin labeling)
freq= SR*freq/Npts
indxFl= np.where(freq>=fPlot[0]*1000)[0][0]  # find freq index re above (0.2) kHz
indxFh= np.where(freq<=fPlot[1]*1000)[0][-1]  # find freq index re under (7) kHz

# ==== spectral averaging loop
for n in range(0,M):
    indx= n*Npts  # index offset so to move along waveform
    signal=  np.squeeze(wf[indx:indx+Npts]);  # extract segment
    # --- deal w/ FFT
    spec= rfft(signal)  
    mag= abs(spec)  # magnitude
    phase= np.angle(spec) # phase
    phaseUW= np.unwrap(phase)   # unwrapped phase (NOT USED)
    phaseDIFF= np.diff(phaseUW) # phase diff re lower freq bin (Nearest Neighbor)
    # --- store away vals
    storeM[:,n]= mag  # spectral mags
    storeP[:,n]= phase # raw phases
    storeWF[:,n]= signal  # waveform segment
    storePd[:,n]= phaseDIFF  # phase diff re lower freq bin
    # ==== deal w/ time-delayed phase diff. (re last buffer)
    if (n>=1 and n<=M-2):
        indxL= (n-1)*Npts  # previous segment index 
        tS= t[indxL]   # time stamp for that point
        signalL=  np.squeeze(wf[indxL:indxL+Npts]);  # re-extract last segment
        specL= rfft(signalL) 
        phaseL= np.angle(specL)
        # --- grab subsequent time segment (re coherogram)
        indxH= (n+1)*Npts  # previous segment index 
        tSh= t[indxH]   # time stamp for that point
        signalH=  np.squeeze(wf[indxH:indxH+Npts]);  # re-extract last segment
        specH= rfft(signalH) 
        phaseH= np.angle(specH)
        # ==== now compute phase diff re last segment (phaseDIFF2) or next (phaseDIFF3)
        phaseDIFF2= phase-phaseL 
        phaseDIFF3= phaseH-phase
        # ==== perform "phase correction" re last time buffer
        corrSP= mag*np.exp(1j*phaseDIFF2)    # 
        corrWF= irfft(corrSP)   # convert to time domain
        # ==== compute vector strength (across freq) for this instance in one of two ways
        # (first method seems to make more sense and yield consistent results)
        if (1==1):
            # 1. compute avg. phase diff over these two intervals and associated VS
            #avgDphi= 0.5*(phaseDIFF2+phaseDIFF3)
            zzA= 0.5*(np.sin(phaseDIFF2)+np.sin(phaseDIFF3))
            zzB= 0.5*(np.cos(phaseDIFF2)+np.cos(phaseDIFF3))
            vsI= np.sqrt(zzA**2 + zzB**2)
        else:
            # 2. Use Roongthumskul 2019 format, but not taking the mag
            # (seems like taking the imaginary part yields most useful)
            Fj= spec/specL
            vsI= np.imag(Fj/abs(Fj))
            #vsI= np.angle(Fj/abs(Fj))/(2*np.pi);
            ##vsI= abs(Fj/abs(Fj));
        # --- store
        storePd2[:,n-1]= phaseDIFF2 # phase diff re previous time segment
        storeWFcorr[:,n-1]= corrWF # phase-corrected wf
        storeVS[:,n-1]= vsI  # time-delayed coherence
        storeT[n-1]= tS  #
        
# ==== tdPC: Phase coherence via vector strength re previous segment
# vC= sqrt(mean(sin(phiC-phi0)).^2 +mean(cos(phiC-phi0)).^2);
xx= np.average(np.sin(storePd2),axis=1)
yy= np.average(np.cos(storePd2),axis=1)
coherence= np.sqrt(xx**2 + yy**2)

# ==== nnPC: Phase coherence via vector strength re adjacent freq bin
xxNN= np.average(np.sin(storePd),axis=1)
yyNN= np.average(np.cos(storePd),axis=1)
coherenceNN= np.sqrt(xxNN**2 + yyNN**2)
freqAVG= freq[1:]- 0.5*np.diff(freq)


# ==== bookeeping III
tP = np.arange(indx/SR,(indx+Npts-0)/SR,1/SR); # time assoc. for segment (only for plotting)
specAVGm= np.average(storeM,axis=1)  # spectral-avgd MAGs
specAVGmDB= 20*np.log10(specAVGm)    # "  " in dB
specAVGp= np.average(storeP,axis=1)  # spectral-avgd PHASEs
# --- time-averaged version (sans phase corr.)
timeAVGwf= np.average(storeWF,axis=1)  # time-averaged waveform
specAVGwf= rfft(timeAVGwf)    # magnitude
specAVGwfDB= 20*np.log10(abs(specAVGwf))  # "  " in dB
# --- time-averaged: phase-corrected version
timeAVGwfCorr= np.average(storeWFcorr,axis=1)  # time-averaged waveform
specAVGwfCorr= rfft(timeAVGwfCorr)  # magnitude
specAVGwfCorrDB= 20*np.log10(abs(specAVGwfCorr))  # "  " in dB

# =============================================
# ==== visualize
plt.close("all")
# ==== ** coherence (tdPC and nnPC; along w/ magnitude)
if 1==0:
    fig5, ax5  = plt.subplots(2,1)
    # --- mags. on top
    ax5[0].plot(freq/1000,specAVGmDB,linestyle='-', marker=markA, 
                   color='k',label='Spectral Avg.')
    ax5[0].set_xlabel('Frequency [kHz]')  
    ax5[0].set_ylabel('Magnitude [dB]')
    #ax5[0].legend(loc="upper right")
    ax5[0].set_title(fileN) 
    ax5[0].set_xlim(fPlot)
    ax5[0].grid()
    ax5[0].set_ylim([np.min(specAVGmDB[indxFl:indxFh])+magL[0],
                    np.max(specAVGmDB[indxFl:indxFh])+magL[1]])
    # --- coherence on bottom
    ax5[1].plot(freq/1000,coherence,linestyle='-', 
                   marker=markA,color='k',label='tdPC')
    ax5[1].plot(freqAVG/1000,coherenceNN,'r',lw=1,linestyle='--',
                   marker=markB,label='nnPC',markersize=4)
    ax5[1].set_xlabel('Frequency [kHz]')  
    ax5[1].set_ylabel('Phase Coherence') 
    ax5[1].set_title(fileN) 
    ax5[1].grid()
    ax5[1].set_xlim(fPlot)
    ax5[1].set_ylim(cPlot)
    ax5[1].legend(loc="upper right")
    plt.tight_layout()
    plt.show()

# ==== ** Comparison of spec-avgd and (phase-corr) time-avgd
if 1==0:
    fig2, ax2 = plt.subplots()
    sp1 = plt.plot(freq/1000,specAVGmDB,linestyle='-',marker=markA, 
                   color='k',label='Spectral Avg.')
    fig2= plt.xlabel('Frequency [kHz]')
    fig2= plt.ylabel('Spectral-Avgd. Magnitude [dB]') 
    fig2= plt.title(fileN) 
    fig2= plt.grid()
    fig2= plt.xlim(fPlot)
    fig2= plt.ylim([np.min(specAVGmDB[indxFl:indxFh])-5,
                    np.max(specAVGmDB[indxFl:indxFh])+5])
    ax2b = ax2.twinx()  # second ordinate (same abscissa)
    sp2 = ax2b.plot(freq/1000,specAVGwfCorrDB,linestyle='--', markersize=4,
                   marker=markB,color='r',label='(Phase-Corr) Time Avg.')
    spT = sp1+sp2
    labelS = [l.get_label() for l in spT]
    ax2.legend(spT,labelS,loc="lower right")
    ax2b.set_ylabel('Time-Avgd. Magnitude [dB]')
    plt.tight_layout()

# ==== ** spectrogram & coherogram pair
if 1==0:
    # --- a few derived vals
    tG,fG = np.meshgrid(storeT,freq/1000)
    spectrogV= 20*np.log10(storeM[:,2:])  # grab last M-1 spec. mags and convert to dB
    # ---
    fig6, (ax1, ax2) = plt.subplots(nrows=2)
    # --- spectrogram
    cs= ax1.pcolor(tG,fG,spectrogV,cmap='jet')
    cbar1= fig6.colorbar(cs,ax=ax1)
    cbar1.ax.set_ylabel('Magnitude [dB]')
    cs.set_clim(-20,10)  # comment to let limits auto-set
    ax1.set_ylim(fPlot)
    ax1.set_title('Spectrogram')
    ax1.set_ylabel('Frequency [kHz]')
    ax1.set_xlabel('Time [s]') 
    # --- coherogram
    cs2= ax2.pcolor(tG,fG,storeVS,cmap='viridis', vmin=0, vmax=1)
    cbar2=fig6.colorbar(cs2,ax=ax2)
    cbar2.ax.set_ylabel('Coherence')
    ax2.set_ylim(fPlot)
    cs2.set_clim(0.3,1)
    ax2.set_title('Coherogram')
    ax2.set_ylabel('Frequency [kHz]')
    ax2.set_xlabel('Time [s]') 
    plt.tight_layout()

# --- * averaged time-averaged. MAG
if 1==0:
    fig3 = plt.subplots()
    fig3= plt.plot(freq/1000,specAVGwfDB,linestyle='-', 
                   marker=markA,color='k',label='Uncorr.')
    fig3= plt.plot(freq/1000,specAVGwfCorrDB,linestyle='--', 
                   marker=markB,color='r',label='Corrected')
    fig3= plt.xlim(fPlot)
    #fig3= plt.ylim([np.min(specAVGwfDB[indxFl:indxFh])-5,
    #                np.max(specAVGwfDB[indxFl:indxFh])+5])
    fig3= plt.xlabel('Frequency [kHz]')
    fig3= plt.ylabel('Magnitude [dB]') 
    fig3= plt.title(fileN) 
    fig3= plt.grid()
    plt.ylim([-60,10])
    fig3= plt.legend()
    plt.tight_layout()

# --- single waveform and assoc. spectrum
if 1==0:
    fig1, ax1 = plt.subplots(2,1)
    ax1[0].plot(tP,signal,linestyle='-', marker=markA, 
                color='k',label='wf')
    ax1[0].set_xlabel('Time [s]')  
    ax1[0].set_ylabel('Signal [arb]') 
    ax1[0].set_title('Last waveform used for averaging')
    ax1[0].grid()
    ax1[1].plot(freq/1000,20*np.log10(spec),'r',label='X',marker=markB)
    ax1[1].set_xlabel('Frequency [kHz]')  
    ax1[1].set_ylabel('Magnitude [dB]') 
    ax1[1].set_title('Spectrum')
    ax1[1].grid()
    ax1[1].set_xlim([0,8])
    fig1.tight_layout(pad=1.5)

plt.show()