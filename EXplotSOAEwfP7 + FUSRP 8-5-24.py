#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXplotSOAEwfP7.py

Purpose: (Python) Analysis of SOAE waveform data, including 
novel analysis of phase info to extract measure of phase coherence

# ==== Notes ====
o v7: Adding in option to add noise to wf (helps w/ model sims but doesn't really
affect actual SOAE data) as well as option to apply a Hanning window
o "phase coherence" = vector strength (as defined by V&D08 eq.20), 
computed several different ways:
1. time-delayed (tdPC): for a given freq. bin, ref = same bin for previous 
    time buffer
2. nearest-neighbor (nnPC): ref. = neighboring (lower) freq. bin
3. "coherogram" re plus/minus one time buffer
 ---
 o determines max number of avgs possible for averaging (M)
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
o 44100 Hz, except Groeningen is 48000; cricket SR determ. below, Vaclav model
data is 40000 Hz (coded in below) and V&D model is 128000
--- re FIG creation
o For coherogram, try using TH21RearwaveformSOAE and Npts=256*4 via
the imaginary part of VS (yields nice horizontal banding)
o re tdPC Npts=512 seems to work for anoles (& owl?) and 2048 for humans
---
2024 SOAE coherence MS Fig.1=
> human_TH14RearwaveformSOAE.mat (Npts= 256*8,fPlot= [0.2,6],cPlot= [0,1])
> owl_TAG4learSOAEwf1.mat (Npts= 256*4,fPlot= [4,11],cPlot= [0,.8])
> anole_ACsb24rearSOAEwfA1.mat (Npts= 256*4,fPlot= [0.2,6],cPlot= [0,.6])
> cricket_177.txt (Npts= 256*5,fPlot= [0.2,6],cPlot= [0,.8])

> fileN = 'Model/coherenceD1summed.txt'  #coherenceD2summed

Created on Mon May 16 10:59:20 2022
@author: CB
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from numpy.fft import rfft
from numpy.fft import irfft
# ====================================================
root= './Data/'         # root path to file
#fileN = 'Model/coherenceD2summed.txt'  #coherenceD2summed
#fileN = 'anole_ACsb32learSOAEwfG1.mat'   # file name
#fileN = 'human_TH21RearwaveformSOAE.mat'   # file name
fileN = 'human_TH14RearwaveformSOAE.mat'  # SR=128000


Npts= 256*16;     # length of fft window (# of points) [should ideally be 2^N]
                   # [time window will be the same length]                
#mark= '.'  # marker style ('.' for dots, '' for none)
markB= 1  # boolean to turn off markers for plots (0=no markers) {1}
SR= 44100;         # sample rate (see above) [Hz] {default = 44100 Hz}
fPlot= [0,5]  # freq. limits for plotting [kHz] {[0.2,6]}
cPlot= [0,1]  # coherence vert. plot limits {[0,1]}
magL=[-1,1]     # mag limits for plotting {[-5,5]}
downSample = 0  # boolean to downsample the data by 1/2
addNoise=0    # boolean to add noise to waveform before analysis {0}
windowB=0   # boolean to allow for application of a Hanning window {0}

# optionally define window size in terms of seconds
if 1==1:
    t_win = 0.1
    Npts = int(t_win * SR) 

# ====================================================
# ==== bookeeping I (re file loading)
#fname = os.path.join(root,fileN)
fname = r"C:\Users\Owner\OneDrive\Documents\GitHub\vodscillators\SOAE Data\TH14RearwaveformSOAE.mat"
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
# ==== add in noise to waveform (useful re model sims?)
if (addNoise==1):
    #wf= wf.flatten()+ 2.1*np.random.randn(len(wf))
    #wf= wf.flatten()+ float(np.mean(wf.flatten()))*10000*np.random.randn(len(wf))
    wf= wf.flatten()+ float(np.mean(np.abs(wf.flatten())))*0.5*np.random.randn(len(wf))

# --- determine numb. of segments for spectral averaging (and use as much wf as possible)
alpha= len(wf)
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
Npts= int(np.floor(Npts))
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
    # =======================
    # option to apply a windowing function
    if (windowB==1):
        signal=signal*np.hanning(len(signal))  
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
            Fj= spec/specL;
            vsI= np.imag(Fj/abs(Fj));
            #vsI= np.angle(Fj/abs(Fj))/(2*np.pi);
            ##vsI= abs(Fj/abs(Fj));
        # --- store
        storePd2[:,n-1]= phaseDIFF2 # phase diff re previous time segment
        storeWFcorr[:,n-1]= corrWF # phase-corrected wf
        storeVS[:,n-1]= vsI  # time-delayed coherence (aka Gamma)
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
"""
PSD NORMALIZING FACTOR? Do you average before or after squaring/taking log? 
The following is what we did - get the canonical PSD for each window and then average that over windows, and finally convert to dB at the end
"""
if 1==1:
    normalizing_factor = SR * Npts
    specAVGmDB= 10*np.log10(np.mean((storeM**2)/normalizing_factor, axis=1))
specAVGp= np.average(storeP,axis=1) # spectral-avgd PHASEs
# --- time-averaged version (sans phase corr.)
timeAVGwf= np.average(storeWF,axis=1)  # time-averaged waveform
specAVGwf= rfft(timeAVGwf)    # magnitude
specAVGwfDB= 20*np.log10(abs(specAVGwf))  # "  " in dB
# --- time-averaged: phase-corrected version
timeAVGwfCorr= np.average(storeWFcorr,axis=1)  # time-averaged waveform
specAVGwfCorr= rfft(timeAVGwfCorr)  # magnitude
specAVGwfCorrDB= 20*np.log10(abs(specAVGwfCorr))  # "  " in dB
# --- complex-averaged vers. of phase-corrected version
# (alternative reality check re performing the irfft and rfft to get specAVGwfCorr)
specAVGpcS= np.average(storeM[:,1:-1]*np.exp(1j*storePd2),axis=1)  # 

# =============================================
# ==== visualize
plt.close("all")

# ==== ** coherence (tdPC and nnPC; along w/ magnitude)
if 1==0:
    fig5, ax5  = plt.subplots(2,1)
    # --- mags. on top
    ax5[0].plot(freq/1000,specAVGmDB,linestyle='-', marker=markA, 
                   color='k',label='Spectral Avg.')
    #ax5[0].set_xlabel('Frequency [kHz]')  
    ax5[0].set_ylabel('Magnitude [dB]',fontsize=12)
    #ax5[0].legend(loc="upper right")
    ax5[0].set_title(fileN,fontsize=10,loc='right') 
    ax5[0].set_xlim(fPlot)
    ax5[0].grid()
    ax5[0].set_ylim([np.min(specAVGmDB[indxFl:indxFh])+magL[0],
                    np.max(specAVGmDB[indxFl:indxFh])+magL[1]])
    # --- coherence on bottom
    ax5[1].plot(freq/1000,coherence,linestyle='-', 
                   marker=markA,color='k',label=r'$C_{\tau}$')
    ax5[1].plot(freqAVG/1000,coherenceNN,'r',lw=1,linestyle='--',
                   marker=markB,label=r'$C_{\theta}$',markersize=4)
    ax5[1].set_xlabel('Frequency [kHz]',fontsize=12)  
    ax5[1].set_ylabel('Phase Coherence',fontsize=12) 
    #ax5[1].set_title(fileN) 
    ax5[1].grid()
    ax5[1].set_xlim(fPlot)
    ax5[1].set_ylim(cPlot)
    ax5[1].legend(loc="upper right")
    plt.tight_layout()
# ==== ** Comparison of spec-avgd and (phase-corr) time-avgd
if 1==0:
    fig2, ax2 = plt.subplots()
    sp1 = plt.plot(freq/1000,specAVGmDB,linestyle='-',marker=markA, 
                   color='k',label='Spectral Avg.')
    fig2= plt.xlabel('Frequency [kHz]',fontsize=12)
    fig2= plt.ylabel('Spectral-Avgd. Magnitude [dB]',fontsize=12) 
    fig2= plt.title(fileN,fontsize=10,loc='right') 
    fig2= plt.grid()
    fig2= plt.xlim(fPlot)
    fig2= plt.ylim([np.min(specAVGmDB[indxFl:indxFh])-5,
                    np.max(specAVGmDB[indxFl:indxFh])+5])
    ax2b = ax2.twinx()  # second ordinate (same abscissa)
    sp2 = ax2b.plot(freq/1000,specAVGwfCorrDB,linestyle='--', markersize=4,
                   marker=markB,color='r',label='(Phase-Corr) Time Avg.')
    spT = sp1+sp2
    # NOTE: comment out below unless checking equivalence on specAVGwfCorr and specAVGpcS
    if (1==0):
        sp3 = ax2b.plot(freq/1000,20*np.log10(abs(specAVGpcS)),
                         label='(Phase-Corr) Spec. Avg.')
        spT = sp1+sp2+sp3
    
    labelS = [l.get_label() for l in spT]
    ax2.legend(spT,labelS,loc="lower right")
    ax2b.set_ylabel('Time-Avgd. Magnitude [dB]',fontsize=12)
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
    #plt.plot(freq/1000,20*np.log10(abs(specAVGpcS)),label='Corrected Spec-Avg.',color='m')
    fig3= plt.xlim(fPlot)
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
    
    
# plot <|phase diffs|> for C_theta
if 1==1:
    # fonts and such
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVuSans"
    "font."
    })
    
    # unwrap phase with respect to frequency bin axis (0th axis), not the time window axis (1st axis)
    phasesUW = np.unwrap(storeP, axis=0)
    # get phase differences between frequency bins (0th axis)
    phase_diffsUW = np.diff(phasesUW, axis=0)
    # finally, get <|phase diffs|> averaging over time window axis (1st)
    means = np.mean(np.abs(phase_diffsUW), axis=1)

    # redefine frequency axis
        # note frequency bins have width 1/t_win 
        # also, since np.diff is a[i+1] - a[i], we need to shift the axis to the right
    t_win = Npts / SR
    bin_size = 1/t_win
    # so we'll take off the final bin and then shift everything else to the right by half a bin
    freqAVG2 = freq[:-1] + (1/2)*(bin_size)
    # finally, convert  to kHz
    freqAVG2 = freqAVG2/1000
    freq = freq/1000

    # set up plots and plot parameters
    fig1 = plt.figure(1)
    axt = plt.subplot(2, 1, 1)
    axb = plt.subplot(2, 1, 2)
    
    # marker for PC/PSD
    markA = "+"
    # marker for means
    markB = "."
    # set font size for all labels
    fs = "18"
    # define the <|phase diffs|> latex string
    means_label = r"$\langle|\phi_j^{{\theta}}|\rangle$"
    # twin the axes so we can add a second y axis to each subplot
    axt2 = axt.twinx()
    axb2 = axb.twinx()
    
    # TOP SUBPLOT
    # plot PC on top
    axt2.plot(freqAVG2, coherenceNN, label=r"$C_{{\theta}}$", marker=markA, color='m', lw=1)
    axt2.set_ylabel('Vector Strength', fontsize=fs)
    axt2.legend(loc="upper right", fontsize=fs)
    
    # plot means on top
    axt.plot(freqAVG2, means, label=means_label, color='black', marker=markB, lw=1)
    axt.set_ylabel(means_label, fontsize=fs)
    axt.legend(loc="upper left", fontsize=fs)

    # BOTTOM SUBPLOT
    # plot psd on bottom
    axb2.plot(freq, specAVGmDB, label="PSD", color='r', marker=markA, lw=1)
    axb2.set_ylabel('PSD [dB]', fontsize=fs)
    axb2.legend(loc="lower right", fontsize=fs)
    
    # plot means on bottom
    axb.plot(freqAVG2, means, label=means_label, color='black', marker=markB, lw=1)
    axb.set_ylabel(means_label, fontsize=fs)
    axb.legend(loc="lower left", fontsize=fs)

    # set title, xlims, and xlabels
    title = r"$\langle|\phi_j^{{\theta}}|\rangle$" + ", " + r"$C_{{\theta}}$" + ", and PSD for "
    axt.set_xlabel("Frequency [kHz]", fontsize=fs)
    axb.set_xlabel("Frequency [kHz]", fontsize=fs)
    axt.set_title(title + fileN, fontsize="22")
    axt.set_xlim(fPlot)
    axb.set_xlim(fPlot)

    # tighten up layout and show
    plt.tight_layout()
    fig1.set_size_inches(18, 10)
    # save fig
    fig1.savefig('abs_avg_pd_TH14_ChrisCode.png', dpi=500, bbox_inches='tight')
    # plt.show()