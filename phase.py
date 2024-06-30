#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# -------------------------------- 
#root= './Files/'         # root path to file
##fileN = 'AC6rearSOAEwfB1.mat'   # file name
#fileN = 'TH21RearwaveformSOAE.mat'   # file name
#
#SR= 44100;         # sample rate [Hz]
#Npts= 8192*1;     # length of fft window (# of points) [should ideally be 2^N]
                   # [time window will be the same length]
# -------------------------------- 

# ==== bookeeping I
#fname = os.path.join(root,fileN)
#if (fileN[-3:]=='mat'):   # load in data as a .mat file
#    data = scipy.io.loadmat(fname)  # loading in a .mat file
#    wf= data['wf']   # grab actual data
#else:   # load in as a .txt file
#    wf = np.loadtxt(fname)  # loading in a .xtt file

# --- determine numb. of segments for spectral averaging 
# (and use as much wf as possible)
#M= int(np.floor(len(wf)/Npts))
#print(f'# of avgs = {str(M)} ')
## --- allocate some buffers
#storeM= np.empty([int(Npts/2+1),M])
#storeP= np.empty([int(Npts/2+1),M])
#storePd= np.empty([int(Npts/2),M])
#storeWF= np.empty([int(Npts),M])
#storePd2= np.empty([int(Npts/2+1),M-1])  # smaller buffer for phase diffs

# ==== bookeeping II
#df = SR/Npts;  
#t = np.arange(0,Npts/SR,1/SR)  # create an array of time points, Npoints long
#dt= 1/SR;  # spacing of time steps
#freq= np.arange(0,(Npts+1)/2,1)    # create a freq. array (for FFT bin labeling)
#freq= SR*freq/Npts;
#indxFl= np.where(freq>=200)[0][0]  # find freq index re above (0.2) kHz
#indxFh= np.where(freq<=7000)[0][-1]  # find freq index re under (7) kHz
# ==== spectral averaging loop
for n in range(0,M):
    indx= n*Npts  # index offset so to move along waveform
    signal=  np.squeeze(wf[indx:indx+Npts]);  # extract segment
    # --- deal w/ FFT
    spec= rfft(signal)  # magnitude
    mag= abs(spec)
    phase= np.angle(spec)
    phaseUW= np.unwrap(phase)
    phaseDIFF= np.diff(phaseUW) # phase diff re lower freq bin
    # --- store away
    storeM[:,n]= mag
    storeP[:,n]= phase
    storePd[:,n]= phaseDIFF
    storeWF[:,n]= signal
    # ==== 
    if (n>=1):
        indxL= (n-1)*Npts  # previous segment index 
        signalL=  np.squeeze(wf[indxL:indxL+Npts]);  # re-extract last segment
        specL= rfft(signalL) 
        phaseL= np.angle(specL)
        # --- now compute phase diff re last segment (phaseDIFF2) and store
        phaseDIFF2= phase-phaseL
        storePd2[:,n-1]= phaseDIFF2

# ====
# vC= sqrt(mean(sin(phiC-phi0)).^2 +mean(cos(phiC-phi0)).^2);
xx= np.average(np.sin(storePd2),axis=1)
yy= np.average(np.cos(storePd2),axis=1)
coherence= np.sqrt(xx**2 + yy**2)


# ====  
tP = np.arange(indx/SR,(indx+Npts-0)/SR,1/SR); # time assoc. for segment (only for plotting)
specAVGm= np.average(storeM,axis=1)  # spectral-avgd MAGs
specAVGp= np.average(storeP,axis=1)  # spectral-avgd PHASEs
specAVGpd= np.average(storePd,axis=1) # spectral-avgd phase diff.
# --- time-averaged version
timeAVGwf= np.average(storeWF,axis=1)  # time-averaged waveform
specAVGwf= rfft(timeAVGwf)

# ==== visualize
plt.close("all")
# --- single waveform and assoc. spectrum
if 1==0:
    fig1, ax1 = plt.subplots(2,1)
    ax1[0].plot(tP,signal,'k-',label='wf')
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

# --- averaged spect. MAG
specAVGmDB= 20*np.log10(specAVGm)
fig2 = plt.subplots()
fig2= plt.plot(freq/1000,specAVGmDB,'b-',lw=1,label='X')
fig2= plt.xlabel('Frequency [kHz]')
fig2= plt.ylabel('Magnitude [dB]') 
fig2= plt.title(fileN) 
fig2= plt.grid()
fig2= plt.xlim([0, 7])
fig2= plt.ylim([np.min(specAVGmDB[indxFl:indxFh])-5,
                np.max(specAVGmDB[indxFl:indxFh])+5])


# --- averaged time-averaged. MAG
if 1==0:
    specAVGwfDB= 20*np.log10(abs(specAVGwf))
    fig3 = plt.subplots()
    fig3= plt.plot(freq/1000,specAVGwfDB,'b-',lw=2)
    fig3= plt.xlim([0, 7])
    fig3= plt.ylim([np.min(specAVGwfDB[indxFl:indxFh])-5,
                    np.max(specAVGwfDB[indxFl:indxFh])+5])
    fig3= plt.xlabel('Frequency [kHz]')
    fig3= plt.ylabel('Magnitude [dB]') 
    fig3= plt.title('Time-averaged spectrum') 
    fig3= plt.grid()

# --- averaged spect. PHASE (difference re lower freq bin)
# NOTE: Not sure this is coded to completion to ascertain this/that
if 1==0:
    fig4 = plt.subplots()
    fig4= plt.plot(freq[0:-1]/1000,specAVGpd,'b-',lw=1,label='X')
    #fig4= plt.plot(freq/1000,specAVGp,'b-',lw=1,label='X')
    fig4= plt.xlabel('Frequency [kHz]')  
    fig4= plt.ylabel('Phase [rads]') 
    fig4= plt.title(fileN) 
    fig4= plt.grid()
    fig4= plt.xlim([0, 7])
    #fig2= plt.ylim([-0.2,0.2])

# --- coherence
if 1==1:
    fig1 = plt.subplots()
    fig1= plt.plot(freq/1000,coherence,'b-',lw=1,label='X')
    #fig4= plt.plot(freq/1000,specAVGp,'b-',lw=1,label='X')
    fig5= plt.xlabel('Frequency [kHz]')  
    fig5= plt.ylabel('Phase Coherence (i.e., vector strength)') 
    fig5= plt.title(fileN) 
    fig5= plt.grid()
    fig5= plt.xlim([0, 7])
    #fig2= plt.ylim([-0.2,0.2])
