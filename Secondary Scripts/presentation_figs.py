
import numpy as np
from vodscillator import *
from matplotlib import pyplot as plt, patches
from funcs_plotting import *
from funcs_spectral import *
import scipy.io

dpi=300
reso=[7, 9]
bbox="tight"


        

# human coherence vs psd
if 1==0:
    filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\SOAE Data\\"
    filename = 'TH14RearwaveformSOAE.mat'
    mat = scipy.io.loadmat(filepath + filename)
    wf = np.squeeze(mat['wf'])
    sr=44100
    coherence_vs_spectrum(wf, sr, t_win=0.1, t_shift=0.1, xmin=0, xmax=5000, wf_title="Human SOAE Waveform")
    
    plt.gcf().set_size_inches(reso) # set figure's size manually to your full screen (32x18)
    plt.savefig('human soae wf.png', dpi=dpi, bbox_inches=bbox)
    

# human soae wf
if 1==0:
    filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\SOAE Data\\"
    filename = 'TH14RearwaveformSOAE.mat'
    mat = scipy.io.loadmat(filepath + filename)
    wf = np.squeeze(mat['wf'])
    sr=44100
    x = np.arange(0, 1, 1/sr)
    plt.title("Human SOAE Waveform")
    plt.plot(x, wf[0:len(x)])
    plt.xlabel("Time (Seconds)")
    
    plt.gcf().set_size_inches(reso) # set figure's size manually to your full screen (32x18)
    plt.savefig('human soae wf.png', dpi=dpi, bbox_inches=bbox)
    


#vs2, vs3
if 1==1:
    def plot(thetas):
        x = np.cos(thetas)
        y = np.sin(thetas)
        xavg = np.mean(x)
        yavg = np.mean(y)
        plt.figure()
        ax = plt.gca()
        t = np.arange(-10, 10, 0.05)
        ax.plot(np.cos(t), np.sin(t), linewidth=0.5, color="black")
        ax.plot(np.zeros(len(t)), t, linewidth=0.5, color="black")
        ax.plot(t, np.zeros(len(t)), linewidth=0.5, color="black")
        # use colormap
        colormap = np.array(['r', 'g', 'b', 'c', 'm', 'y'])
        ax.scatter(x, y, c=colormap)
        
        for i in range(len(x)):
            ax.arrow(0, 0, x[i], y[i], width=0.001, head_width=0.06, length_includes_head=True, color=colormap[i])
            
        
        ax.scatter(xavg, yavg, label="Average Vector", color="black")
        ax.arrow(0, 0, xavg, yavg, width=0.03, head_width=0.12, length_includes_head=True, color="black")
        ax.legend()
        ax.axis('equal')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        
    thetas = [1.05, 1, 1.1, 1.2, 1.3, 6]
    plot(thetas)
    plt.gcf().set_size_inches(reso) # set figure's size manually to your full screen (32x18)
    plt.savefig('vs2.png', dpi=dpi, bbox_inches=bbox)
    
    thetas = [0.1, 6, 2, 2.3, 4.5, 1.4]   
    plot(thetas)
    plt.gcf().set_size_inches(reso) # set figure's size manually to your full screen (32x18)
    plt.savefig('vs3.png', dpi=dpi, bbox_inches=bbox)



# vs1
if 1==0:
    plt.figure(4)
    sr=100
    d = 1/sr
    tf = 50
    x = np.arange(0, tf, d)
    N = len(x)
    color_inc="purple"
    color_c="green"
    
    fmin=0
    fmax = 3
    F = 1
    t_win = 5
     
    while(True):
        # get sin and noise
        amp_inc = 2
        noise_inc = 2*amp_inc*np.random.sample(N) - amp_inc
        sin_inc = amp_inc*np.sin(F*2*np.pi*x + 0.1*np.random.sample(N)) + noise_inc / 10
        
        # construct incoherent waveform
        M = 6
        Ni = np.zeros(M, int)
        K = 75
        for i in range(M):
            Ni[i] = int(i*(N/M) + np.random.randint(-K, K))
            if i ==0:
                Ni[i] = int(i*N/M)
        
        y_inc = np.concatenate((noise_inc[0:Ni[0]], sin_inc[Ni[0]:Ni[1]], noise_inc[Ni[1]:Ni[2]], sin_inc[Ni[2]:Ni[3]], noise_inc[Ni[3]:Ni[4]], sin_inc[Ni[3]:Ni[4]], noise_inc[Ni[4]:Ni[5]]))
        
        x = x[0:len(y_inc)]
        amp_c=2
        noise_c = 2*amp_c*np.random.sample(len(x)) - amp_c
        y_c = 1/2*amp_inc*np.sin(F*2*np.pi*x) + noise_c / 10
        
        if len(x)==len(y_inc):
            break
    
    
    # get FTs
    N=len(x)
    f = rfftfreq(N, d)
    ft_inc = rfft(y_inc)
    ft_c = rfft(y_c)
    
    f_ind = int(F*N*d)
        
    # Mag FTs
    ylim = 2.4
    plt.subplot(2, 2, 1)
    plt.plot(x, y_inc, color=color_inc)
    plt.ylim(-ylim, ylim)
    plt.title("Incoherent Waveform")
    
    plt.subplot(2, 2, 2)
    plt.plot(x, y_c, color=color_c)
    plt.ylim(-ylim, ylim)
    plt.title("Coherent Waveform")
    
    plt.subplot(2, 2, 3)
    plt.plot(f, np.abs(ft_inc), color=color_inc)
    # put a marker on the relevant point
    plt.plot(f[f_ind], np.abs(ft_inc[f_ind]), "o", color=color_inc)
    plt.ylabel("Magnitude of FT")
    plt.xlabel("Frequency")
    plt.xlim(fmin, fmax)
    plt.ylim(0, 2400)
    
    plt.subplot(2, 2, 4)
    plt.plot(f, np.abs(ft_c), color=color_c)
    # put a marker on the relevant point
    plt.plot(f[f_ind], np.abs(ft_c[f_ind]), "o", color=color_c)
    plt.ylabel("Magnitude of FT")
    plt.xlabel("Frequency")
    plt.xlim(fmin, fmax)
    plt.ylim(0, 2400)
    
    plt.gcf().set_size_inches(reso) # set figure's size manually to your full screen (32x18)
    plt.savefig('coherent vs incoherent ft.png', dpi=dpi, bbox_inches=bbox)
    
    plt.show()
    
    # coherence plot (for final)
    
    # get coherence
    inc = get_coherence(y_inc, sample_rate=sr, t_win=t_win, return_all=True, ref_type="next_win")
    c = get_coherence(y_c, sample_rate=sr, t_win=t_win, return_all=True, ref_type="next_win")
    f_pc = inc["freq_ax"]
    pc_inc = inc["coherence"]
    pc_c = c["coherence"]
    
    ylim = 2.4
    plt.subplot(2, 2, 1)
    plt.plot(x, y_inc, color=color_inc)
    plt.ylim(-ylim, ylim)
    plt.title("Incoherent Waveform")
    
    plt.subplot(2, 2, 2)
    plt.plot(x, y_c, color=color_c)
    plt.ylim(-ylim, ylim)
    plt.title("Coherent Waveform")
    
    plt.subplot(2, 2, 3)
    plt.plot(f_pc, pc_inc, color=color_inc)
    plt.plot(f_pc[t_win], pc_inc[t_win], "o", color=color_inc)
    plt.ylabel("Phase Coherence")
    plt.xlabel("Frequency")
    plt.xlim(fmin, fmax)
    plt.ylim(0, 1.1)
    
    plt.subplot(2, 2, 4)
    plt.plot(f_pc, pc_c, color=color_c)
    plt.plot(f_pc[t_win], pc_c[t_win], "o", color=color_c)
    plt.ylabel("Phase Coherence")
    plt.xlabel("Frequency")
    plt.xlim(fmin, fmax)
    plt.ylim(0, 1.1)
    
    plt.gcf().set_size_inches(reso) # set figure's size manually to your full screen (32x18)
    plt.savefig('coherent vs incoherent pc.png', dpi=dpi, bbox_inches=bbox)
    
    plt.show()

    
    # just waveforms plot
    plt.subplot(2, 1, 1)
    plt.plot(x, y_inc, color=color_inc)
    plt.ylim(-amp_inc*1.2, amp_inc*1.2)
    plt.title("Incoherent Waveform")
    
    plt.subplot(2, 1, 2)
    plt.plot(x, y_c, color=color_c)
    plt.ylim(-amp_inc*1.2, amp_inc*1.2)
    plt.title("Coherent Waveform")
    
    plt.gcf().set_size_inches(reso) # set figure's size manually to your full screen (32x18)
    plt.savefig('coherent vs incoherent wf.png', dpi=dpi, bbox_inches=bbox)


if 1==0:
    plt.figure(3)
    sr=100
    x = np.arange(0, 100, 1/sr)
    N = len(x)
    d = 1/sr
    y1 = np.cos(2*np.pi*x)
    label1="Cosine at 1Hz"
    
    W=100
    
    plt.subplot(2, 1, 1)
    plt.title(label1)
    plt.plot(x[0:W], y1[0:W], label="Window 1", color="blue")
    plt.plot(x[W:2*W], y1[W:2*W], label="Window 2", color="green")
    plt.plot(x[2*W:3*W], y1[2*W:3*W], label="Window 3", color="purple")
    plt.plot(x[3*W:4*W], y1[3*W:4*W], label="Window 4", color="red")
    plt.vlines([1, 2, 3, 4], -1, 1, linestyles='dashed', color='black')
    plt.legend(loc="lower left")
    plt.xlim(0, 4)
    
    
        
    W=125
    plt.subplot(2, 1, 2)
    plt.title(label1)
    plt.plot(x[0:W], y1[0:W], label="Window 1", color="blue")
    plt.plot(x[W:2*W], y1[W:2*W], label="Window 2", color="green")
    plt.plot(x[2*W:3*W], y1[2*W:3*W], label="Window 3", color="purple")
    plt.plot(x[3*W:4*W], y1[3*W:4*W], label="Window 4", color="red")
    plt.legend(loc="lower left")
    plt.vlines([W/100, W/100*2, W/100*3, W/100*4], -1, 1, linestyles='dashed', color='black')
    plt.xlim(0, 4)
    

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(reso) # set figure's size manually to your full screen (32x18)
    plt.savefig('windowing.png', dpi=dpi, bbox_inches=bbox)


if 1==0:
    plt.figure(2)
    sr=100
    x = np.arange(0, 100, 1/sr)
    N = len(x)
    d = 1/sr
    y1 = np.cos(4*np.pi*x)
    label1="cos(4pi*x)"
    y2 = np.cos(2*np.pi*x + np.pi/2)
    label2="cos(2pi*x + pi/2)"

    ts = y1+y2

    ft = rfft(ts)
    f = rfftfreq(N, d)
    
    
    plt.subplot(2, 2, 2)
    plt.title("Freq = 2Hz, Phase = 0")
    plt.plot(x, y1, label=label1, color="blue")
    plt.xlim(0, 5)
    plt.legend(loc="upper right")

    plt.subplot(2, 2, 1)
    plt.title("Freq = 1Hz, Phase = pi/2")
    plt.plot(x, y2, label=label2, color="red")
    plt.xlim(0, 5)
    plt.legend(loc="upper right")

    plt.subplot(2, 2, 3)
    plt.title("Magnitude of Fourier Transform")
    plt.plot(f, np.abs(ft), label="|FT(x(t))|", color="black")
    plt.legend()
    plt.ylabel("Magnitude")
    plt.xlabel("Frequency [Hz]")
    plt.xlim(0, 3)
    
    plt.subplot(2, 2, 4)
    plt.title("Fourier Transform (of Sum)")
    plt.plot(f, np.angle(ft), label="Phase", color='green')
    print(np.where(f==1.0))
    print(N*d)
    z1 = ft[np.where(f==1.0)]
    z2 = ft[np.where(f==2.0)]
    plt.plot([1], np.angle([z1]), 'ro')
    plt.plot([2], np.angle([z2]), 'bo')
    plt.ylabel("Phase")
    plt.xlabel("Frequency [Hz]")
    plt.legend()
    plt.xlim(0, 3)
    

    
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(reso) # set figure's size manually to your full screen (32x18)
    plt.savefig('wf ft angle.png', dpi=dpi, bbox_inches=bbox)
    


if 1==0:
    plt.figure(1)
    sr=100
    x = np.arange(0, 100, 1/sr)
    y1 = np.cos(4*np.pi*x)
    label1="cos(4pi*x)"
    y2 = np.cos(2*np.pi*x + np.pi/2)
    label2="cos(2pi*x + pi/2)"

    ts = y1+y2

    ft = rfft(ts)
    f = rfftfreq(len(ts), 1/sr)

    plt.subplot(2, 2, 2)
    plt.title("Freq = 2Hz, Phase = 0")
    plt.plot(x, y1, label=label1, color="blue")
    plt.xlim(0, 5)
    plt.legend(loc="upper right")

    plt.subplot(2, 2, 1)
    plt.title("Freq = 1Hz, Phase = pi/2")
    plt.plot(x, y2, label=label2, color="red")
    plt.xlim(0, 5)
    plt.legend(loc="upper right")

    plt.subplot(2, 2, 3)
    plt.title("Summed Time Series x(t)")
    plt.xlim(0, 5)
    plt.plot(x,ts, label=label1 + " + " + label2, color="purple")
    plt.legend(loc="lower left")

    plt.subplot(2, 2, 4)
    plt.title("Magnitude of Fourier Transform")
    plt.plot(f, np.abs(ft), label="|FT(x(t))|", color="black")
    plt.ylabel("Magnitude")
    plt.xlabel("Frequency [Hz]")
    plt.xlim(0, 3)

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(reso) # set figure's size manually to your full screen (32x18)
    plt.savefig('wf ft.png', dpi=dpi, bbox_inches=bbox)




