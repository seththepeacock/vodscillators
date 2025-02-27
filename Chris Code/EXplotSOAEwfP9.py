#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXplotSOAEwfP9.py

Purpose: (Python) Analysis of SOAE waveform data, including 
analysis of phase info to extract measure of phase coherence

# ==== Notes ==== [vers updates + addit. notes at bottom]
o v.9 calculates C_xi, a sliding window time-delayed coherence measure
as well as the associated avgd. phase; also cleaned up code overall and
fixed some minor issues/bugs (e.g., wf is now flattened)
o v.9 not only also now plots the avgd.phase assoc. with the xi-method
( call is xi-phase), but also computews the product of the gradients
(across freq.) of the spectrally-avgd. mags and xi-phase as an initial
attempt to combine (and thereby reinforce) each measure to improve(?)
freq. regions of SOAE activity
o v.8 allows for the signal to be high-pass filtered
o Good files for analysis/visualization (i.e., fileN = ):
    'anole_ACfls27rearSOAEwfD01.mat' (Npts= 256*4 and xiS= 256*1)
    'owl_TAG4learSOAEwf1.mat'  (Npts= 256*8 and xiS= 256*2)
    'human_TH14RearwaveformSOAE.mat' (Npts= 256*14 and xiS= 256*4)
    
    'vib_ACfls19_DualSO_Righttymp_01_SO.mat'
    'anole_AC6rearSOAEwfB1.mat' (Npts= 256*8 and xiS= 256*0.5)


Created on Mon May 16 10:59:20 2022
@author: CB
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from numpy.fft import rfft
from numpy.fft import irfft
#from scipy.signal import butter,filtfilt
from scipy import signal
# ====================================================
# --- file info
root= './Data/'         # root path to file
fileN = 'owl_TAG4learSOAEwf1.mat'   # file name
#fileN = 'Other/ve10re01.mat'
# ---
Npts= 256*8;     # tau = length of time window into FFT (ideally be 2^N) [# of points] {256*4}
xiS= 256*0.5        # number of points to shift over in time to grab next buffer
# --- Filter the signal?
HPfilter= 1  # boolean to turn on/off the high-pass filtering (0-no, 1-yes)
cutoffF= 0.15  #  cutoff freq. [kHz]
order = 2    # filter order
# ---
SR= 50000         # sample rate (see above) [Hz] {default = 44100 Hz}
fPlot= [2,12]  # freq. limits for plotting [kHz] {[0.2,6], [2,12] for owl}
cPlot= [0, 0.4]  # coherence vert. plot limits {[0,1]}
magL=[-1,1]     # mag limits for plotting {[-5,5]}
markB= 1  # boolean to turn off markers for plots (0=no markers) {1}
downSample = 0  # boolean to downsample the data by 1/2
addNoise= 0    # boolean to add noise to waveform before analysis {0}
windowB= 0   # boolean to allow for application of a Hanning window {0}
FTnorm= "forward"   # normalization meth. for FFT
# ====================================================
#
# ==== bookeeping I (re file loading and plotting logistics)
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
# --- deal w/ cricket file structure (and SR; see notes at bottom)
#if (wf.shape[1]>1):
if (fileN[0:7]=='cricket'):
    SR= round(1/np.mean(np.diff(wf[:,0])))  # use first column to determine SR
    wf= wf[:,1]  # grab second column 
# --- markers for plotting (if used)  
if markB==0:
    markA=''
    markB=''
    markC= ''
else:
    markA='.'  # C_tau
    markB='+'  # C_theta
    markC='s'  # C_xi 
#
# ==== bookeeping II (re modify whole waveform?)
# NOTE: windowing is done on shorter segments (and thus in analysis loops)
# --- flatten the array if needed (needed for .mat versus .txt read in)
wf= wf.flatten()
# --- downsample by 1/2?
if (downSample==1):
    wf= wf[1::2]  # skip every other point
    SR= SR/2
# --- add in noise to waveform (useful re model sims?)
if (addNoise==1):
    nAmpl= 0.5   # scaling factor for the noise
    wf= wf.flatten()+ float(np.mean(np.abs(wf.flatten())))* nAmpl*np.random.randn(len(wf))
# --- high-pass filtering
if HPfilter==1:
    # REF: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
    sos = signal.butter(order, cutoffF*1000,'hp',fs=SR,output='sos')
    wf = signal.sosfilt(sos, wf)
#
# ==== bookeeping III (determine derived analysis params)
# --- determine numb. of segments for spectral averaging (and use as much wf as possible)
M= int(np.floor(len(wf)/Npts))  # numb. of tau time segments (no overlap)
xiS= int(xiS)  # convert xi shift {pts} to int (as may be needed)
Mxi= int(np.floor(len(wf)/(xiS)-np.ceil(Npts/xiS)-5))  # max> numb. of xi-shifted tau time segments (w/ overlap)
# NOTE: the -10 here is a bit kludgy
# ---
Npts= int(np.floor(Npts))
Nt= len(wf)  # total numb. of time points
t= np.linspace(0,(Nt-1)/SR,Nt)   # create assoc. time array
df = SR/Npts   # freq bin width
freq= np.arange(0,(Npts+1)/2,1)    # create freq. array (for FFT bin labeling)
freq= SR*freq/Npts
indxFl= np.where(freq>=fPlot[0]*1000)[0][0]  # find freq index re above (0.2) kHz
indxFh= np.where(freq<=fPlot[1]*1000)[0][-1]  # find freq index re under (7) kHz
# --- send some vals to screen for ref
print(f'# of spectral avgs = {str(M-1)} ')
print(f'Delay window length = {1000*Npts/SR:.2f} ms')
print(f'Xi window shift = {1000*xiS/SR:.2f} ms (or {SR/xiS:.2f} Hz)')
# --- allocate some buffers
storeM= np.empty([int(Npts/2+1),M]) # store away spectral magnitudes
storeP= np.empty([int(Npts/2+1),M])  # store away spectral phases
storeWF= np.empty([int(Npts),M])  # waveform segments (for time-averaging)
storePDtau= np.empty([int(Npts/2+1),M-2])  # smaller buffer for phase diffs re windows
storePDtheta= np.empty([int(Npts/2),M])  # phase diff re lower freq bin
storeWFcorr= np.empty([int(Npts),M-1])  # phase-corrected wf
storeVS= np.empty([int(Npts/2+1),M-2])  # time-delay coherence (for coherogram)
storeT= np.empty([M-2])  # time array for spectrograms

# ---------------------------------------------------------------
# ==== Loop I: spectral averaging + coherence (NO OVERLAPPING)
for n in range(0,M):
    indx= n*Npts  # index offset so to move along waveform
    signal= np.squeeze(wf[indx:indx+Npts])  # extract segment
    # =======================
    # option to apply a windowing function
    if (windowB==1):
        signal=signal*np.hanning(len(signal))  
    # --- deal w/ FFT
    spec= rfft(signal,norm=FTnorm)  
    mag= abs(spec)  # magnitude
    phase= np.angle(spec) # phase
    # --- store away vals
    storeM[:,n]= mag  # spectral mags
    storeP[:,n]= phase # raw phases
    storeWF[:,n]= signal  # waveform segment
    storePDtheta[:,n]= np.diff(phase)  # phase diff re adjacent freq bin (i.e., \phi_j^{{\theta}})
    # ==== deal w/ time-delayed phase diff. (re last buffer)
    if (n>=1 and n<=M-2):
        indxL= (n-1)*Npts  # previous segment index 
        tS= t[indxL]   # time stamp for that point
        signalL=  np.squeeze(wf[indxL:indxL+Npts])  # re-extract last segment
        specL= rfft(signalL,norm=FTnorm) 
        phaseL= np.angle(specL)
        # --- grab subsequent time segment (re coherogram)
        indxH= (n+1)*Npts  # previous segment index 
        tSh= t[indxH]   # time stamp for that point
        signalH=  np.squeeze(wf[indxH:indxH+Npts])  # re-extract last segment
        specH= rfft(signalH,norm=FTnorm) 
        phaseH= np.angle(specH)
        # ==== now compute phase diff re last segment (phaseDIFF2) or next (phaseDIFF3)
        phaseDIFF2= phase-phaseL # (i.e., \phi_j^{{\tau}})
        phaseDIFF3= phaseH-phase
        # ==== perform "phase correction" re last time buffer
        corrSP= mag*np.exp(1j*phaseDIFF2)    # 
        corrWF= irfft(corrSP, norm=FTnorm)   # convert to time domain
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
            #vsI= np.angle(Fj/abs(Fj))/(2*np.pi)
            ##vsI= abs(Fj/abs(Fj))
        # --- store
        storePDtau[:,n-1]= phaseDIFF2 # phase diff re previous time segment (i.e., \phi_j^{{\tau}})
        storeWFcorr[:,n-1]= corrWF # phase-corrected wf
        storeVS[:,n-1]= vsI  # time-delayed coherence (aka Gamma)
        storeT[n-1]= tS  #
#
tP = np.arange(indx/SR,(indx+Npts-0)/SR,1/SR) # time assoc. for last segment (only for plotting)
# --- tdPC: Phase coherence via vector strength re previous segment
xx= np.average(np.sin(storePDtau),axis=1)
yy= np.average(np.cos(storePDtau),axis=1)
coherence= np.sqrt(xx**2 + yy**2)
# --- nnPC: Phase coherence via vector strength re adjacent freq bin
xxNN= np.average(np.sin(storePDtheta),axis=1)
yyNN= np.average(np.cos(storePDtheta),axis=1)
coherenceNN= np.sqrt(xxNN**2 + yyNN**2)
freqAVG= freq[1:]- 0.5*np.diff(freq)
# --- calculate avgd. spectra
specAVGm= np.average(storeM,axis=1)  # spectral-avgd MAGs
specAVGmDB= 20*np.log10(specAVGm)    # "  " in dB
specAVGp= np.average(storeP,axis=1)  # spectral-avgd PHASEs
#
# ---------------------------------------------------------------
# ==== Loop II: xi-shifting coherence (WITH OVERLAPPING)
storePDxi= np.empty([int(Npts/2+1),Mxi])  # allocate buffer to store phase diffs
storePxi= np.empty([int(Npts/2+1),Mxi])  # allocate buffer to store phases
for n in range(0,Mxi):
    indxNow= n*xiS  # index offset so to move along waveform re NOW
    indxNext= (n+1)*(xiS)  # index offset re NEXT
    signalNow= np.squeeze(wf[indxNow:indxNow+Npts])  # extract "now" segment
    signalNext= np.squeeze(wf[indxNext:indxNext+Npts])  # extract "next" segment
    # --- option to apply a windowing function
    if (windowB==1):
        signalNow= signalNow*np.hanning(len(signalNow)) 
        signalNext= signalNext*np.hanning(len(signalNext)) 
    # --- deal w/ FFT
    specNow= rfft(signalNow,norm=FTnorm)  
    magNow= abs(specNow)  # magnitude
    phaseNow= np.angle(specNow) # phase
    specNext= rfft(signalNext,norm=FTnorm)  
    magNext= abs(specNext)  # magnitude
    phaseNext= np.angle(specNext) # phase
    # --- determine phase diff and store away
    phaseDIFFxi= phaseNext-phaseNow
    storePDxi[:,n-1]= phaseDIFFxi
    # -- store away phase, but correct for xi-shifting
    storePxi[:,n-1]= np.angle(np.exp(1j*(phaseNow-(2*np.pi*freq*n*(xiS/SR)))))
#
# --- do the averaging of the phase to calculate the xi-based coherence
xxXi= np.average(np.sin(storePDxi),axis=1)
yyXi= np.average(np.cos(storePDxi),axis=1)
coherenceXi= np.sqrt(xxXi**2 + yyXi**2)
#
# ---------------------------------------------------------------       
# ==== deal w/ phase processing (indepenedent of vector stength)
# --- unwrap phase to compute avgd. \phi_j^{{\theta}} 
# unwrapping puts phases differences in [-pi, pi] rather than [-2pi, 2pi]
phaseUWtheta= np.unwrap(storeP, axis=0)  # first unwrap w.r.t. frequency axis
phaseDtheta= np.diff(phaseUWtheta, axis=0)  #  ** second compute diff. re adjacent bin
phaseUWtau= np.unwrap(storeP, axis=1) # ditto w.r.t. time window axis
phaseDtau = np.diff(phaseUWtau, axis=1) # **
# --- also apply to the xi-shifted vers.
phaseUWxi= np.unwrap(storePxi, axis=1)
phaseDxi = np.diff(phaseUWxi, axis=1) 
#
# --- now compute avgd vers
storePDthetaAVG= np.average(np.abs(phaseDtheta),axis=1)  # lastly avg. the abs val. over windows
storePDtauAVG= np.average(np.abs(phaseDtau),axis=1)
storePDxiAVG= np.average(np.abs(phaseDxi),axis=1) # no further corr. needed (since handled earlier)
#storePDxiAVGcorr= np.angle(np.exp(1j*(storePDxiAVG-(2*np.pi*freq*(xiS/SR)))))
# --- time-averaged version (sans phase corr.)
timeAVGwf= np.average(storeWF,axis=1)  # time-averaged waveform
specAVGwf= rfft(timeAVGwf,norm=FTnorm)    # magnitude
specAVGwfDB= 20*np.log10(abs(specAVGwf))  # "  " in dB
# --- time-averaged: phase-corrected version
timeAVGwfCorr= np.average(storeWFcorr,axis=1)  # time-averaged waveform
specAVGwfCorr= rfft(timeAVGwfCorr,norm=FTnorm)  # magnitude
specAVGwfCorrDB= 20*np.log10(abs(specAVGwfCorr))  # "  " in dB
# --- complex-averaged vers. of "phase-corrected" version
# (alternative reality check re performing the irfft and rfft to get specAVGwfCorr)
specAVGpcS= np.average(storeM[:,1:-1]*np.exp(1j*storePDtau),axis=1)  # 

# --------------------------------------------------------------- 
# Compute a combined measure of the mags & phases
#mGrad= np.gradient(abs(specAVGwf))  # gradient of avgd. spec. mags
mGrad= np.gradient(20*np.log10(abs(specAVGwf)))  # gradient of avgd. spec. mags (in dB)
pGrad= np.gradient(abs(storePDxiAVG))  # gradient of avgd. xi-phase
# --- rectify? (i.e., make all vals positive) 
# {1 helps visually, but perhpas at loss of info/generality)
if (1==1):
    mGrad= np.abs(mGrad)
    pGrad= np.abs(pGrad)

mpGradProd= mGrad*pGrad  # product of their slopes
mpGradProd[0:10]=0  # set first several vals to zero so to ignore (kludgy)
combinedMP= mpGradProd

# =============================================
# ==== visualize
plt.close("all")
# - create a title string contain. filename + key coherence params
titC= fileN[:-4]+": "+r'$ \tau=$'+str(1000*Npts/SR)+" & "+r'$ \xi=$'+str(1000*xiS/SR)+" ms"
# --- FIG - single waveform and assoc. spectrum
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
# ==== **FIG - coherence (tdPC and nnPC; along w/ magnitude)
if 1==1:
    fig5, ax5  = plt.subplots(2,1)
    # --- mags. on top
    ax5[0].plot(freq/1000,specAVGmDB,linestyle='-', marker=markA, 
                   color='k',label='Spectral Avg.')
    #ax5[0].set_xlabel('Frequency [kHz]')  
    fracLS= 0.05  # scaling fraction to help w/ vert. lims {0.05}
    ax5[0].set_ylabel('Magnitude [dB]',fontsize=12)
    ax5[0].set_title(titC,fontsize=10,loc='right') 
    ax5[0].set_xlim(fPlot)
    ax5[0].grid()
    ax5[0].set_ylim([np.min(specAVGmDB[indxFl:indxFh])+magL[0],
                    np.max(specAVGmDB[indxFl:indxFh])+magL[1]])
    # --- coherence on bottom
    Lb1= ax5[1].plot(freq/1000,coherence,linestyle='-', 
                   marker=markA,color='k',label=r'$C_{\tau}$')
    Lb2= ax5[1].plot(freqAVG/1000,coherenceNN,'r',lw=1,linestyle='--',
                   marker=markB,label=r'$C_{\theta}$',markersize=4)
    # --- use right y-axis for C_xi
    ax5b = ax5[1].twinx()
    Lb3= ax5b.plot(freq/1000,coherenceXi,'b',lw=1,linestyle='-.',
                   marker=markC,label=r'$C_{\xi}$',markersize=2)
    ax5b.set_ylim([(1-fracLS)*np.min(coherenceXi), (1+fracLS)*np.max(coherenceXi)])
    ax5[1].set_xlabel('Frequency [kHz]',fontsize=12)  
    ax5[1].set_ylabel(r'Coherence ($C_{\tau}$ & $C_{\theta}$)',fontsize=12) 
    ax5b.set_ylabel(r'$C_{\xi}$',fontsize=12,color='b') 
    ax5[1].grid()
    ax5[1].set_xlim(fPlot)
    #ax5[1].set_ylim(cPlot)
    ax5[1].set_ylim([-0.05,np.max([(1+fracLS)*np.max(coherence),
                                   (1+fracLS)*np.max(coherenceNN)])]) 
    LbT = Lb1+Lb2+Lb3  # deal w/ legend given multiple ordinates
    labelF5 = [l.get_label() for l in LbT]
    ax5[1].legend(LbT,labelF5,loc="best")
    #ax5[1].legend(loc="upper right")
    plt.tight_layout()
# ==== ** FIG - spectral mags: spectral-avgd vs time-avgd w/ phase corr
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
    # NOTE: set boolean to 0 unless checking equivalence on specAVGwfCorr and specAVGpcS
    if (1==0):
        sp3 = ax2b.plot(freq/1000,20*np.log10(abs(specAVGpcS)),
                         label='(Phase-Corr) Spec. Avg.')
        spT = sp1+sp2+sp3
    
    labelS = [l.get_label() for l in spT]
    ax2.legend(spT,labelS,loc="lower right")
    ax2b.set_ylabel('Time-Avgd. Magnitude [dB]',fontsize=12)
    plt.tight_layout()
# --- * time-averaged. spectral mags: phase corr vs no corr.
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
    #plt.ylim([-60,10])
    fig3= plt.legend()
    plt.tight_layout()
# --- ** Combine gradient of spec.-avgd peaks and xi-phase 
if 1==1:
    fig55, ax55   = plt.subplots(2,1)
    ax55[0].plot(freq/1000,specAVGmDB,linestyle='-', marker=markA, 
                   color='k',label='Spectral Avg.')
    fracLS= 0.05  # scaling fraction to help w/ vert. lims {0.05}
    ax55[0].set_ylabel('Magnitude [dB]',fontsize=12)
    ax55[0].set_title(titC,fontsize=10,loc='right') 
    ax55[0].set_xlim(fPlot)
    ax55[0].grid()
    ax55[0].set_ylim([np.min(specAVGmDB[indxFl:indxFh])+magL[0],
                    np.max(specAVGmDB[indxFl:indxFh])+magL[1]])
    # Note: Initial analysis suggests no need to plot combinedMP in dB
    ax55[1].plot(freq/1000,combinedMP,linestyle='-', markersize=2,
                   marker=markC,color='b',label='Uncorr.') 
    ax55[1].set_xlim(fPlot)
    ax55[1].set_xlabel('Frequency [kHz]')
    ax55[1].set_ylabel('Comb. Mag & Phase metric') 
    ax55[1].grid()
    plt.tight_layout()
    
# ==== **FIG -  spectrogram & coherogram pair
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
    #cs.set_clim(-20,10)  # comment to let limits auto-set
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

# ==== *** FIG -  Avg. phase diffs: adjacent freq bin vs adjacent window
if 1==1:
    fig7, ax7 = plt.subplots()
    sp1 = plt.plot(freqAVG/1000,storePDthetaAVG,linestyle='--',marker=markB, 
                   color='r',label=r'$\phi^{\theta}$')
    sp2 = plt.plot(freq/1000,storePDtauAVG,linestyle='-',marker=markA, 
                   color='k',label=r'$\phi^{\tau}$')
    sp3 = plt.plot(freq/1000,storePDxiAVG,linestyle='-.',marker=markC, 
                   color='b',label=r'$\phi^{\xi}$',markersize=2)
    fig2= plt.xlabel('Frequency [kHz]',fontsize=12)
    fig2= plt.ylabel(r"$\langle|\phi_j|\rangle$ [rads]",fontsize=12) 
    ax7.legend(loc="lower right")
    fig2= plt.title(titC,fontsize=10,loc='right') 
    fig2= plt.grid()
    fig2= plt.xlim(fPlot)

# === FIG (c/o Seth) - Avg. phase diff for C_theta vs C_theta vs mags (SI figure 1+2)
if 1==0:
    # set up plots and plot parameters
    fig, ax = plt.subplots(2, 1)
    fs = "12" # set font size for all labels
    means_label = r"$\langle|\phi_j^{{\theta}}|\rangle$" # define the <|phase diffs|> latex string
    # -- twin the axes so we can add a second y axis to each subplot
    axt2 = ax[0].twinx()
    axb2 = ax[1].twinx()
    # TOP SUBPLOT
    axt2.plot(freqAVG/1000, coherenceNN, label=r"$C_{{\theta}}$", marker=markB, color='m', lw=1)
    axt2.set_ylabel('Vector Strength', fontsize=fs)
    axt2.legend(loc="upper right", fontsize=fs)
    ax[0].plot(freqAVG/1000, storePDthetaAVG, label=means_label, color='black', marker=markA, lw=1)
    ax[0].set_ylabel(means_label, fontsize=fs)
    ax[0].legend(loc="upper left", fontsize=fs)
    # BOTTOM SUBPLOT
    axb2.plot(freq/1000, specAVGmDB, label="Magnitude", color='r', marker=markB, lw=1)
    axb2.set_ylabel('Magnitude [dB]', fontsize=fs)
    axb2.legend(loc="lower right", fontsize=fs)
    ax[1].plot(freqAVG/1000, storePDthetaAVG, label=means_label, color='black', marker=markA, lw=1)
    ax[1].set_ylabel(means_label, fontsize=fs)
    ax[1].legend(loc="lower left", fontsize=fs)
    # set title, xlims, and xlabels
    ax[0].set_title(fileN, fontsize="14")
    for i in [0, 1]:
        ax[i].set_xlabel("Frequency [kHz]", fontsize=fs)
        ax[i].set_xlim(fPlot)
        ax[i].set_ylim(0, np.pi)
    plt.tight_layout()
    plt.show()


    
"""
==== Notes (cont) ====
o "phase coherence" = vector strength (as defined by V&D08 eq.20), 
computed several different ways:
1. time-delayed (C_tau): for a given freq. bin, ref = same bin for previous 
    time buffer
2. nearest-neighbor (C_theta): ref. = neighboring (lower) freq. bin
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
> fileN = 'Model/coherenceD2summed.txt' (note SR=128000)
--> sans noise, coherence high about peaks and drops off between valleys
(but stays around 0.5 for higher freqs)
o 44100 Hz, except Groeningen is 48000; cricket SR determ. below, Vaclav model
data is 40000 Hz (coded in below) and V&D model is 128000
--- re FIG creation
o For coherogram, try using TH21RearwaveformSOAE and Npts=256*4 via
the imaginary part of VS (yields nice horizontal banding)
o re tdPC Npts=512 seems to work for anoles (& owl?) and 2048 for humans
 v9: 
    + 
o v8: 
    + Includes updates from Seth Peacock + CB (minor) tweaks
    + v.8 allows for the signal to be high-pass filtered
    + Plots can be turned on/off below via booleans (i.e., ""if 1==0" --> if 1==1")
o v7: 
    + Adding in option to add noise to wf (helps w/ model sims but doesn't really
    affect actual SOAE data) as well as option to apply a Hanning window. 
    + Also now plots an averaged phi_j^{{theta}} in a separate window to show structure 
    there
    + also deals w/ FFT normalization (FTnorm = "forward" or "backward"). THe default
    is the latter, but perhaps....
--- [cricket-specific notes]
# NOTE: files from Natasha contain this header (which is deleted in ./Data/x.txt))
#Source File Name:	F177_LL_FFT_1.5 to 30 khz SOAE 1255_ 23.4Amb_23.74Pla
#Signal:	Time - Vib Velocity - Samples
#Time	Time Signal
#[ s ]	[ m/s ]   
---
2024 SOAE coherence MS Fig.1=
> human_TH14RearwaveformSOAE.mat (Npts= 256*8,fPlot= [0.2,6],cPlot= [0,1])
> owl_TAG4learSOAEwf1.mat (Npts= 256*4,fPlot= [4,11],cPlot= [0,.8])
> anole_ACsb24rearSOAEwfA1.mat (Npts= 256*4,fPlot= [0.2,6],cPlot= [0,.6])
> cricket_177.txt (Npts= 256*5,fPlot= [0.2,6],cPlot= [0,.8])
Other figures utilize
> Model/coherenceD1summed.txt
"""
