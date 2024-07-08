#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXtestPhaseCoherence.py

[IN PROGRESS; functional, but could use further testing/refinement]
Purpose: Test the time-delayed phase coherence measure being 
developed for SOAE analysis (e.g., EXplotSOAEwfP4.py) for a 
noisy sinusoid to verify predicted behavior. In additon to purely
additive noise, stochastics can also be added into the amplitude,
frequency, and/or phase to examine effects

Predictions 
o Coherence will be zero(-ish) for noise-dominated freqs. and 
non-zero for places where there willl be sinusoidal energy (at fs)
o Quantizing the freq re the FFT window should affect the coherence,
making it localized to fs
o AM would not affect coherence, but FM would (given that it 
would lead to phase diffusion)
o Adding a (small?) amount of frequency noise will adversely 
affect the coherence
o Adding a (small?) amount of phase noise will adversely 
affect the coherence
o Adding a amplitude noise will adversely affect the coherence, but
to a lesser effect (as this would average out)
    
Notes
o Combines elements of EXsnr.py and (chiefly) EXplotSOAEwfP4.py
o If the sinusoid is quantized, the coherence is localized to 
unity at that frequency (unless the amplitude is very small, in 
which case the coherence decreases but is still localized)                 
o If the sinusoid is non-quantized, the coherence spreads out 
about fs for places where there is energy above the NF, which makes
sense given that those sinusoidal components need to be there 
given the periodic BC
o Frequency noise quickly plummets the coherence
o Phase noise affects things as well, but not as much as freq. noise
    
    
Created on Sat Jul  6 07:21:04 2024
@author: CB
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from numpy.fft import rfft
from numpy.fft import irfft

# -------------------------------- 
fs= 1343   # sinusoidal freq [Hz] {1343}
As= 10.5      # amplitude of sinusoid {1}
Npts= 512*1;     # length of fft window (# of points) [should ideally be 2^N]                 
# --
quantF= 1   # boolean to quantize the sinusoidal freq {1?}
# --
freqNoise=0  # boolean to add frequency noise {0}
alphaF=0.0001  # scaling factor for freq noise {0.0001}
# --
phaseNoise=0  # boolean to add phase noise {?}
alphaP=0.4  # scaling factor for freq noise
# --
amplNoise=0  # boolean to add amplitude noise {0}
alphaA=0.4  # scaling factor for freq noise
# --
mark= '.'  # marker style ('.' for dots, '' for none)
Length= 60   # total length of window to create noisy sinusoid [s] {60?}
SR= 44100;         # sample rate [Hz] {44100 except Groeningen is 48000}
fPlot= [0,7]  # freq. limits for plotting [kHz] {[0,7]}
downSample = 0  # boolean to downsample the data by 1/2
# -------------------------------- 

# ===============================================
# 1. Waveform Generation

# --- bookeeping I
NptsT= Length*SR   # total number of time points
t= np.linspace(0,(NptsT-1)/SR,NptsT)   # time array
dt= 1/SR;  # spacing of time steps
df = SR/Npts;  
fQ= np.ceil(fs/df)*df;   # quantized natural freq.
# === quantize sinusoids freq?
if (quantF==1): 
    fs=fQ
# === add in freq. noise?
if (freqNoise==1): 
    fs=fs*(1+ alphaF*np.random.randn(NptsT))
# === add in amplitude. noise?
if (amplNoise==1): 
    As=As*(1+ alphaA*np.random.randn(NptsT))

# === generate noise + sinusoid
noiseT= np.random.normal(0,1,NptsT)   

if (phaseNoise==1): 
    wf= noiseT+ As*np.sin(fs*2*np.pi*t + 2*np.pi*alphaP*np.random.randn(NptsT))
else  :  
    wf= noiseT+ As*np.sin(fs*2*np.pi*t)

# ===============================================
# 2. (SOAE-like) Analysis

# ==== downsample by 1/2?
if (downSample==1):
    wf= wf[1::2]  # skip every other point
    SR= SR/2
    

# --- determine numb. of segments for spectral averaging 
# (and use as much wf as possible)
M= int(np.floor(len(wf)/Npts))
print(f'# of avgs = {str(M)} ')
# --- allocate some buffers
storeM= np.empty([int(Npts/2+1),M])
storeP= np.empty([int(Npts/2+1),M])
storeWF= np.empty([int(Npts),M])
storePd2= np.empty([int(Npts/2+1),M-1])  # smaller buffer for phase diffs

# ==== bookeeping II
#df = SR/Npts;  
freq= np.arange(0,(Npts+1)/2,1)    # create a freq. array (for FFT bin labeling)
freq= SR*freq/Npts;
indxFl= np.where(freq>=fPlot[0]*1000)[0][0]  # find freq index re above (0.2) kHz
indxFh= np.where(freq<=fPlot[1]*1000)[0][-1]  # find freq index re under (7) kHz
# ==== spectral averaging loop
for n in range(0,M):
    indx= n*Npts  # index offset so to move along waveform
    signal=  np.squeeze(wf[indx:indx+Npts]);  # extract segment
    # --- deal w/ FFT
    spec= rfft(signal)  # magnitude
    mag= abs(spec)
    phase= np.angle(spec)
    phaseUW= np.unwrap(phase)
    # --- store away
    storeM[:,n]= mag
    storeP[:,n]= phase
    storeWF[:,n]= signal
    # ==== deal w/ phase diff. re last buffer
    if (n>=1):
        indxL= (n-1)*Npts  # previous segment index 
        signalL=  np.squeeze(wf[indxL:indxL+Npts]);  # re-extract last segment
        specL= rfft(signalL) 
        phaseL= np.angle(specL)
        # --- now compute phase diff re last segment (phaseDIFF2) and store
        phaseDIFF2= phase-phaseL
        storePd2[:,n-1]= phaseDIFF2

# ==== compute phase coherence via vector strength
# vC= sqrt(mean(sin(phiC-phi0)).^2 +mean(cos(phiC-phi0)).^2);
xx= np.average(np.sin(storePd2),axis=1)
yy= np.average(np.cos(storePd2),axis=1)
coherence= np.sqrt(xx**2 + yy**2)


# ====  
tP = np.arange(indx/SR,(indx+Npts-0)/SR,1/SR); # time assoc. for segment (only for plotting)
specAVGm= np.average(storeM,axis=1)  # spectral-avgd MAGs
specAVGp= np.average(storeP,axis=1)  # spectral-avgd PHASEs
# --- time-averaged version
timeAVGwf= np.average(storeWF,axis=1)  # time-averaged waveform
specAVGwf= rfft(timeAVGwf)

# =============================================
# ==== visualize
plt.close("all")
# --- single waveform and assoc. spectrum
if 1==1:
    fig1, ax1 = plt.subplots(2,1)
    ax1[0].plot(tP,signal,linestyle='-', marker=mark, 
                color='k',label='wf')
    ax1[0].set_xlabel('Time [s]')  
    ax1[0].set_ylabel('Signal [arb]') 
    ax1[0].set_title('Last waveform used for averaging')
    ax1[0].grid()
    ax1[1].plot(freq/1000,20*np.log10(spec),'r-',label='X')
    ax1[1].set_xlabel('Frequency [kHz]')  
    ax1[1].set_ylabel('Magnitude [dB]') 
    ax1[1].set_title('Spectrum')
    ax1[1].grid()
    ax1[1].set_xlim([0,8])
    fig1.tight_layout(pad=1.5)

# ==== averaged spect. MAG
specAVGmDB= 20*np.log10(specAVGm)
fig2 = plt.subplots()
fig2= plt.plot(freq/1000,specAVGmDB,linestyle='-', marker=mark, 
            color='k',label='wf')
fig2= plt.xlabel('Frequency [kHz]')
fig2= plt.ylabel('Magnitude [dB]') 
fig2= plt.title('Noisy Sinusoid (EXtestPhaseCoherence.py)') 
fig2= plt.grid()
fig2= plt.xlim(fPlot)
fig2= plt.ylim([np.min(specAVGmDB[indxFl:indxFh])-5,
                np.max(specAVGmDB[indxFl:indxFh])+5])



# ==== coherence
if 1==1:
    fig5 = plt.subplots()
    fig5= plt.plot(freq/1000,coherence,linestyle='-', 
                   marker=mark,color='k',label='wf')
    fig5= plt.xlabel('Frequency [kHz]')  
    fig5= plt.ylabel('Phase Coherence (i.e., vector strength)') 
    fig5= plt.title('Noisy Sinusoid (EXtestPhaseCoherence.py)') 
    fig5= plt.grid()
    fig5= plt.xlim(fPlot)

<<<<<<< HEAD:Chris Ex Code/EXtestPhaseCoherence.py
plt.show()

=======

plt.show()
>>>>>>> 34f991d (y):EXtestPhaseCoherence.py
