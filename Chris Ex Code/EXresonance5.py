#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXresonance5.py
 
[IN PROGRESS; 
 + need to fix scaling as it is currently kludged due to sensitivity to gamma
 + App.5 eigensol mag seem dependent on tE(??)
 + App.6 mag. scaling not correct
 + create averaging loop for Apps.5&6
 + tweak some other bits/aspects (e.g., how the drive amplitude A is handled
 rather than using kludge scaling re magA1[0]
]                               
                                                                    
o Note: There is a (minor?) bug somewhere such that changing w0
 sometimes leads to oddball scaling for the eigensolution]
 
o Purpose: Code to demonstrate equivalence of the "resonance" response 
of the damped driven harmonic oscillator (DDHO) emerging from 
several different *approaches* (translation of EXhoResonanceF.m from
Matlab; see that code for more details)
    
    
o Eqn. of motion (for sin-driven cases)
dx/dt = y
dy/dt = -gamma*y - (w^2)*x + A*sin(wd*t)

[methods currently included re EXhoResonanceF.m]
+ App.1: Numerically integrate the DDHO using blackbox numpy.odeint,
         extract steady-state portion, and use Fourier transform to
         pick off magnirude and pahse
+ App.2: Analytic solution (from French, 1971; see PHYS 2010 notes too)
+ App.3: Transfer function (via integrating to determine impulse resp. IR)
+ App.4: Numeric computation of Transfer function (TF) - Obtained by determining
% the impulse response and then taking the FFT to numerically get the TF
+ App.5: Eigensolution
+ App.6: Noise-driven Transfer Function (nTF)
+ App.7: Convolve IR & noise
+ App.10: Chirp-driven Transfer Function (chTF)
    
o Notes
> ** The magnitudes are scaled relatively (not absolutely), so ultimately
need to determine the proper scaling coefficients (e.g., depend. upon A,
gamma, etc...) for each App. and weave in                                                  
> App.1 is kinda slow (depends upon wdNUM), so code may take a bit to
run (e.g., ~1 min to run for wdNUM=51)

[other general notes are at bottom of code]                                     
                                            
Created on 2021.12.01 (updated 2023.09.10)
@author:CB
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft
from numpy.fft import irfft
from scipy.integrate import odeint
from scipy.interpolate import CubicSpline

# ================= [User Params]  =================
# --- osc. params
gamma = 20.0      # damping term {20}
w0 = 1000     # natural freq (rads/s) {1000}
# --- sinusoidal drive
A = 100      # drive amplitude {100}
wdNUM= 50     # # of drive freqs to run {50}
wdR = w0*np.linspace(0.25,1.75,wdNUM+1)    # (ang.) drive freqs (rads/s)
                                      # w*np.linspace(0.2,2.0,wdNUM+1) 
# --- chirp params
Ad = 10      # (unscaled) drive amplitude (i.e., A/m)
fS= 20;   # starting freq. [Hz]
fE= 2000;    # ending freq. [Hz]  
# --- noise params
Nxi=3000  # number of base points for noise waveform
                                    
# --- time & spec params
tE = 5        # integration time (assumes tS=0) {5; 6.5536?}
SR= 5*10**3     # "sample rate" (i.e., step-size = 1/SR) [Hz] {5*10**3}
Npoints= 2*8192  # of point for "steady-state" waveform to compute 
                 # spectrum of {16384}
# ==================================================

# ===== bookkeeping =====
# --- ICs
x0 = 0.0       # initial x
y0 = 0.0       # initial y
ICs = [x0,y0]   # repackage ICs
Q= w0/gamma   # -- "quality factor" 
tS = 0         # start time
N = tE*SR     # num of time points
h = (tE-tS)/N   # step-size
tpoints = np.arange(tS,tE,h)   # time point array
L= len(tpoints)   # total numb of time points
VTW = L-(Npoints);  # create offset indx point extracting FFT window
tW= tpoints[L-Npoints:L]  #(shorter/later) time interval for FFT window
tI= tpoints[0:Npoints]  # time window for impulse resp
# --- FFT stuff
df = SR/Npoints
freq= np.arange(0,(Npoints+1)/2,1)    # create a freq. array (for FFT bin labeling)
freq= SR*freq/Npoints;
w= 2*np.pi*freq  # angular freqs
# === get drive freqs to match FFT bin freqs (for peak-picking re App.1)
# -- indicies of drive freqs > 1st drive freq    
xx1=np.argwhere(2*np.pi*freq > wdR[0])
# -- indicies of drive freqs > 2nd drive freq    
xx2=np.argwhere(2*np.pi*freq > wdR[1]) 
xxD= xx2[0,0]-xx1[0,0]  # diff in array indicies
for nn in range(0,len(wdR)):
    wdR[nn] = 2*np.pi*freq[xx1[0,0]+nn*xxD]

# === grabbing driving freqs. from freq array (for instances w/ discrete drive freqs)
#Vindx=  np.where(freq>=wdR[0]/(2*np.pi) & freq<=wdR[-1]/(2*np.pi));    # find relevant indicies
#VindxB= round(linspace(V.indx(1),V.indx(end),P.wDriveN));    # one means to get the desired subset

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# App.1: Numeric integration of ODE for discrete freqs, allowing for steady-state and
# subsequently using the FFT to obtain relevant values in spectral domain

# ==== [define ODE system re App.1]
def dxdt(x,t,gamma,w0,A,wd):
    x1, x2 = x
    dx1dt = x2
    dx2dt = -gamma*x2 - (w0**2)*x1 + A*np.sin(wd*t)
    return dx1dt, dx2dt

# --- initialize arrays for storing SS amplitude and phase
maxA, phaseA = ([] for i in range(2))
# ==== loop to go through each drive freq.
for n in range(0,len(wdR)):
    wd= wdR[n]  # extract drive freq for this loop iteration
    x, y= ([] for i in range(2))  # re-initialize
    # ---- use blackbox solver (rather than RK4)
    x, y = odeint(dxdt,ICs,tpoints,args=(gamma,w0,A,wd)).T    
    # ==== now deal w/ FFT & whatnot
    indx= len(x)
    signal = x[indx-Npoints:indx]  # grab last nPts
    spec= rfft(signal)
    specM = abs(spec)  # extract mag
    specMdb = 20*np.log10(specM)  # convert to dB
    specP = np.angle(spec)  # extract angle
    # === peak-picking to obtain mag., phase and freq
    indxA= np.argmax(specM)
    freqA= freq[indxA]
    maxAt= specM[indxA]
    phaseAt= specP[indxA]
    # === ** correct phase for SS window delay **
    phaseAt= np.angle(np.exp(1j*(phaseAt-wd*tpoints[indx-Npoints])) )
    # --- store away
    maxA= np.insert(maxA,n,maxAt)
    phaseA= np.insert(phaseA,n,phaseAt)
# ==== final bits
phaseUW= np.unwrap(phaseA)/(2*np.pi)  # unwrap the phase
magA1= maxA*np.sqrt(2)  # scale as RMS val
phaseA1= phaseUW-phaseUW[0]  # norm. to create vert. offset (kludge?)

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# App.2: Analytic solution [KLUDGE re  mag scaling]
# REF: see French, 1971; eqns.4.11 on pg.85
# -- create ang. freq. array for plotting analytic solution
wA= np.linspace(wdR[0],wdR[-1],500); 
# --- mag
magA2= A/np.sqrt((w0**2-wA**2)**2 + ((gamma*wA)**2))
# --- phase
phaseA2= np.arctan2((gamma*wA),(-w0**2+wA**2))
phaseA2= np.unwrap(phaseA2)/(2*np.pi)
phaseA2= phaseA2-0.5  # fix 1/2 vert offset
# -- scale mag. (KLUDGE)
normR2= magA2[0]/magA1[0]  # compare ratio at lowest freq to scale
magA2= magA2/normR2

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# App.3: "Impulse response & Transfer function" [KLUDGE re  mag scaling]
# Two versions of TF here (yIp and yIv) which differ in how the "impulse" is
# implemented (i.e., position versus velocity) --> yIv is used
# -- set ICs such that
init0v = [0, A] # ... there is a velocity "impulse" at t=0
init0p = [A, 0] # ... there is a position "impulse" at t=0
# -- make sure to turn off the sin. drive!
Atemp= A
A= 0
# ===  use blackbox solver to integrate both cases
# -- impulse re velocity drive
xIv, yIv= ([] for i in range(2))  # re-initialize
xIv, yIv = odeint(dxdt,init0v,tpoints,args=(gamma,w0,A,wd)).T
# -- impulse re position
xIp, yIp= ([] for i in range(2))  # re-initialize
xIp, yIp = odeint(dxdt,init0p,tpoints,args=(gamma,w0,A,wd)).T  
# -- make sure to turn back on the sin. drive!
A= Atemp
# -- extract relevant #s
yIvS= xIv[0:Npoints]
yIpS= xIp[0:Npoints]
# == set IR
IR= yIvS
# -- spectral analysis 
specIv= rfft(yIvS)
specIp= rfft(yIpS)
specIR= rfft(IR)  # kinda redundant (but useful later on)
magA3= abs(specIR)
phaseA3= np.unwrap(np.angle(specIR))/(2*np.pi);
# -- scale mag. (KLUDGE)
normR3= magA3[xx1[0,0]]/magA2[0]  # compare ratio at lowest freq to scale
magA3= magA3/normR3

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# App.4: Convolve IR w/ sinusoids
# --- initialize arrays for storing SS amplitude and phase
maxA4, phaseA4 = ([] for i in range(2))
# ==== loop to go through each drive freq.
for n in range(0,len(wdR)):
    wd= wdR[n]  # extract drive freq for this loop iteration
    sinWF= A*np.sin(wd*tpoints)
    xcS= np.convolve(sinWF,IR)
    CTvss= xcS[VTW:VTW+Npoints];  # extract steady-state portion of convolv. for FFT
    CTvssS= rfft(CTvss);    # compute spectral representation of convolution
    # === peak-picking to obtain mag., phase and freq
    spec4M = abs(CTvssS) # mag
    spec4P = np.angle(CTvssS) # phase
    indxD= np.argmax(spec4M) # grab max val (at peak)
    freqD= freq[indxD]
    maxDt= spec4M[indxD]
    phaseDt= spec4P[indxD]
    # === ** correct phase for SS window delay **
    #phaseDt= np.angle(np.exp(1j*(phaseDt-wd*tpoints[indx-Npoints])) )
    phaseDt= np.angle(np.exp(1j*(phaseDt-wd*tpoints[len(tpoints)-Npoints])) )
    # --- store away
    maxA4= np.insert(maxA4,n,maxDt)
    phaseA4= np.insert(phaseA4,n,phaseDt)
# === normalize (RMS) mags and unwrap phase (+ref)
magA4= maxA4*np.sqrt(2)
magA4n= magA4*(magA1[0]/magA4[0])

phaseA4UW= np.unwrap(phaseA4)/(2*np.pi)
phaseA4UW= phaseA4UW-phaseA4UW[0]  # ref re first val

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# App.5: Eigensolution meth. [KLUDGE re  mag scaling]
    
# ==== calculate various derived quantities
# -- Eigenvalues, for x=0 (undriven)
lambdaP= 0.5*(-gamma+ np.sqrt((gamma**2-4*w0**2)+0j))  
lambdaM= 0.5*(-gamma- np.sqrt((gamma**2-4*w0**2)+0j))
# -- "correct" eigensolution 
yEP= A*np.e**(tW*lambdaP)
yEPi= yEP.real    
yEPr= yEP.imag  
# -- compute Fourier transform
specEPi= rfft(yEPi) 
magA5= abs(specEPi)
phaseA5= np.unwrap(np.angle(specEPi))/(2*np.pi);
phaseA5= phaseA5-0.5  # fix 1/2 vert offset
# -- scale mag. (KLUDGE)
normR5= magA5[xx1[0,0]]/magA1[0]  # compare ratio at lowest freq to scale
magA5= magA5/normR5

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# App.6: SDE version, i.e., drive DHO w/ ("determinisitic") Brownian noise
# dx/dt = y
# dy/dt = -gamma*y - (w^2)*x + A*Xi
# where gamma=b/m, w= sqrt(k/m), and Xi is the Brownian noise

# ----------- [define ODE system re App.6 (noise)]
def fN(rN,t,xiVal):
    xn = rN[0]
    yn = rN[1]
    fxn = yn
    fyn = -gamma*yn - (w0**2)*xn + A*xiVal
    return np.array([fxn,fyn],float)
# ----------- [define RK4 re noise]
def rk4n(rN,t,h):
    k1 = h*fN(rN,t,xiVal)
    k2 = h*fN(rN+0.5*k1,t+0.5*h,xiVal)
    k3 = h*fN(rN+0.5*k2,t+0.5*h,xiVal)
    k4 = h*fN(rN+k3,t+h,xiVal)
    rN += (k1+2*k2+2*k3+k4)/6
    return rN

# ---
xXi= []; yXi = [];
rN= [x0,y0]
# --- create a unique (low-pass) Brownian noise
nBase= np.random.randn(Nxi,1)  # create noise waveform
tXibase= np.linspace(1,N,num=Nxi)/SR  # time array for base noise
nFine= CubicSpline(tXibase,nBase)  
NoiseR= nFine(tpoints)
Noise= np.concatenate(NoiseR,axis=0)  # need to recast array of arrays to single array
# conversely tried noise in spectral doamin to guarentee it is flat (see stimT=7 from EXspecREP3.m)
# but can't seem to get working correctly (slight bug somewhere?) 
    # mm= 20 # KLUDGE!
    # Asize= len(tpoints)/2 +mm
    # phaseN= 2*np.pi*np.random.rand(int(Asize),1)
    # phaseN= np.concatenate(phaseN,axis=0)  # need to recast array of arrays to single array
    # Noise= irfft(-1j*np.exp(phaseN),norm="backward")
    # Noise= A*Noise[int(mm)-2:-int(mm)]+1
    
# -- FFT-related stuff
dfCh = SR/len(Noise)
freqCh= np.arange(0,(len(Noise)+1)/2,1)    # create a freq. array (for FFT bin labeling)
freqCh= SR*freqCh/len(Noise)
wCh= 2*np.pi*freqCh
# --- integration loop
indx = 0
for t in tpoints:
    xXi = np.insert(xXi,indx,rN[0])
    yXi = np.insert(yXi,indx,rN[1])
    xiVal= Noise[indx]
    rN = rk4n(rN,t,h)
    indx = indx+1
# ===== deal w/ FFT & compute TF
specyXi= rfft(xXi);
specXi= rfft(Noise);
TFnoise= specyXi/specXi  # noise-derived transfer function 
magA6= abs(TFnoise)
phaseA6= np.unwrap(np.angle(TFnoise))/(2*np.pi);
# -- normalize the mag (kludgy)
magA6= magA6*np.sqrt(2)
magA6= magA6*(magA1[0]/magA6[0])


# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# App.7: Convolve IR w/ noise

xcNIR= np.convolve(Noise,IR)

# ===== deal w/ FFT 
xcNIRss= xcNIR[VTW:VTW+Npoints];  # extract steady-state portion of convolv. for FFT
specNIR= rfft(xcNIRss)

magA7= abs(specNIR)
# -- normalize the mag (kludgy)
magA7= magA7*(magA1[0]/magA7[0])
# -- deal w/ phase and ref. re SS portion of the noise (kludge???)
phaseA7raw= np.angle(specNIR)
phaseREFN= np.angle(rfft(Noise[VTW:VTW+Npoints])) # create ref. phase re SS noise portion
phaseA7= phaseA7raw- phaseREFN;  # now correct the phase re the ref
phaseA7= np.unwrap(phaseA7)/(2*np.pi);

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# App.10: Numerically integrate the DDHO using a hard-coded RK4 for a 
# chirp stimulus
# dx/dt = y
# dy/dt = -gamma*y - (w^2)*x + A*sin(wC(t)*t)
# where gamma=b/m, w= sqrt(k/m), and wC(t) linearly changes w/ time
# ----------- [define ODE system re App.10 (chirp)]
def f(r,t):
    xc = r[0]
    yc = r[1]
    fxc = yc
    # assume freq. changes linearly w/ time
    fCi= fS - ((fS-fE)/(2*tpoints[-1]))*t;
    wCi= 2*np.pi*fCi
    fyc = -gamma*yc - (w0**2)*xc + A*np.sin(wCi*t)
    return np.array([fxc,fyc],float)
# ----------- [define RK4 re chirp]
def rk4(r,t,h):
    k1 = h*f(r,t)
    k2 = h*f(r+0.5*k1,t+0.5*h)
    k3 = h*f(r+0.5*k2,t+0.5*h)
    k4 = h*f(r+k3,t+h)
    r += (k1+2*k2+2*k3+k4)/6
    return r
# ---
xCh= []; yCh = [];
r= [x0,y0]
# --- assume freq. changes linearly w/ time
fC= fS - ((fS-fE)/(2*tpoints[-1]))*tpoints; # note factor of 1/2
wC= 2*np.pi*fC
chirp= Ad*np.sin(wC*tpoints)  # chirp waveform itself (used to create TF)
# -- FFT-related stuff
dfCh = SR/len(chirp)
freqCh= np.arange(0,(len(chirp)+1)/2,1)    # create a freq. array (for FFT bin labeling)
freqCh= SR*freqCh/len(chirp)
wCh= 2*np.pi*freqCh
# --- integration loop
indx = 0
for t in tpoints:
    xCh = np.insert(xCh,indx,r[0])
    yCh = np.insert(yCh,indx,r[1])
    r = rk4(r,t,h)
    indx = indx+1

# ===== deal w/ FFT & compute TF
specyCh= rfft(xCh);
specCh= rfft(chirp);
TFch= specyCh/specCh;  # chirp-derived transfer function 
magA10= abs(TFch)
phaseA10= np.unwrap(np.angle(TFch))/(2*np.pi);
# -- normalize the mag (kludgy)
magA10= magA10*np.sqrt(2)
magA10= magA10*(magA1[0]/magA10[0])


# ==================================================
# ===== visualize
plt.close("all")
# --- Fig.1: (last wd run) time waveforms & phase space
if 1==0:
    fig1, ax1 = plt.subplots(2,1)
    ax1[0].plot(tpoints,x,'b-',label='X')
    ax1[0].plot(tpoints[indx-Npoints:indx],signal,'r.',label='SS bit')
    ax1[0].set_xlabel('Time')  
    ax1[0].set_ylabel('Position x') 
    ax1[0].set_title('DDHO responses')
    ax1[0].grid()
    #ax1[0].legend(loc=1)
    # ~~ Fig.1B: phase space
    ax1[1].plot(x,y,'r-',label='X')
    ax1[1].set_xlabel('x')  
    ax1[1].set_ylabel('dx/dt') 
    ax1[1].grid()
    fig1.tight_layout(pad=1.5)

# --- Fig.2: (last wd run) spectrum
if 1==0:
    fig2 = plt.subplots()
    fig2= plt.plot(freq/(w0/(2*np.pi)),specMdb,'r.-',label='X')
    fig2= plt.xlabel('Normaliz. Freq. [f/fo]')  
    fig2= plt.ylabel('Magnitude [dB]') 
    fig2= plt.grid()
    fig2= plt.xlim([0,3])
    
# --- Fig.3: Resonance curve
fig3, ax3 = plt.subplots(2,1)
# --- magnitude
ax3[0].plot(wdR/w0,magA1,'ko-',alpha=0.8,\
                markersize=4,label='App.1: Int. ODE')
ax3[0].plot(wA/w0,magA2,'r.-.',alpha=0.5,label='App.2: Analytic SS')
ax3[0].plot(w/w0,magA3,'b:',lw=2,alpha=0.5,label='App.3: Transfer function')
ax3[0].plot(wdR/w0,magA4n,'+',lw=4,alpha=0.75,label='App.4: Convolv. IR&sins',color='#00FF00')
ax3[0].plot(w/w0,magA5,'c--',alpha=0.5,label='App.5: Eigensol')
ax3[0].plot(wCh/w0,magA6,'-',lw=1,alpha=0.25,color='#FF00FF',label='App.6: TF re Noise')
ax3[0].plot(w/w0,magA7,'-',lw=2,alpha=0.15,color='#DBB40C',label='App.7: Conv. IR & Noise')
ax3[0].plot(wCh/w0,magA10,'k-',lw=8,alpha=0.15,label='App.10: TF re Chirp')
ax3[0].grid()
ax3[0].legend(fontsize="8",loc="upper right")
ax3[0].set_xlim([wdR[0]/w0, wdR[-1]/w0])
ax3[0].set_ylim([0, 1.3*np.max(magA1)])
ax3[0].set_xlabel('Normaliz. Freq. [f/fo]')  
ax3[0].set_ylabel('Steady-state amplitude') 
#ax3[0].set_yscale("log")  # plot on log scale (comment to turn off)
# --- phase
ax3[1].plot(wdR/w0,phaseA1,'ko-',alpha=0.8,markersize=4)
ax3[1].plot(wA/w0,phaseA2,'r.-.',alpha=0.5)
ax3[1].plot(w/w0,phaseA3,'b:',lw=2,alpha=0.5)
ax3[1].plot(wdR/w0,phaseA4UW,'+',lw=6,alpha=0.75,color='#00FF00')
ax3[1].plot(w/w0,phaseA5,'c--',alpha=0.55)
ax3[1].plot(wCh/w0,phaseA6,'-',lw=1,alpha=0.25,color='#FF00FF')
ax3[1].plot(w/w0,phaseA7,'-',lw=2,alpha=0.15,color='#DBB40C')
ax3[1].plot(wCh/w0,phaseA10,'k-',lw=8,alpha=0.15)
ax3[1].grid()
ax3[1].set_xlabel('Normaliz. Freq. [f/fo]')  
ax3[1].set_ylabel('Steady-state phase [cycs]') 
ax3[1].set_ylim([-0.6, 0.1])
ax3[1].set_xlim([wdR[0]/w0, wdR[-1]/w0])
fig3.tight_layout(pad=1.5)
    
# --- Fig.4: Impulse response(s)
if 1==0:
    fig4 = plt.subplots()
    fig4= plt.plot(tI,yEPr,'r.-',label='Eigensolution (yEPr)')
    fig4= plt.xlabel('Time')  
    fig4= plt.ylabel('Impulse response') 
    fig4= plt.grid()
    fig4= plt.xlim([0,3])
    fig4= plt.legend()
    
# --- Fig.5: Noise stuff
if 1==0:
    fig5 = plt.subplots()
    fig5= plt.plot(tpoints,Noise,'k-')
    fig5= plt.xlabel('Time')  
    fig5= plt.ylabel('Noise') 
    fig5= plt.grid()
    
    
    
"""
=================================================
[Other notes]

o built off of EXddho.py (formerly EXresonance.py). See that code for RK4
context (replaced those bits here to use np.odeint to presumably
spped things up)
o Note: Be careful about the relative val. of the params. (e.g., if w is 
large, than A needs to be large too) as otherwise the relative
force terms are small and transients become problematic in the
spectrum  
o Need to "find" a spectral peak for each wd. Using np.max, but can 
also use np.findpeaks. Ref. page is:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
o REF re blackbox solver is "Listing 8.11" of Hill (2012; pg.366) 

"""