import numpy as np
from plots import *
import scipy.io
import pandas as pd
from pathlib import Path

def plot_mags():
    m = get_mags(wf, sr=44100, t_win=1, num_wins=30, dict=True)
    mags = m['mags']
    freq_ax = m['freq_ax']
    plt.plot(freq_ax, np.log10(mags)*20)
    plt.title(fn)
    plt.show()
    
# get the main directory in my computer
main_path_str = "C:\\Users\\Owner\OneDrive\\Desktop\\SOAE Data\\"
# we'll process each subfolder separately since each is likely to have its own quirks
    
    
# We'll build our dataframe by making a dictionary of lists and appending to them
data = {
    'filepath': [],
    'wf': [],  
    'species': [],
    'sr': [],
}



subfolder = "Curated Data"
# First navigate to our directory
directory_path = Path(main_path_str + subfolder)
# now loop through all files in that collection
for fp in directory_path.rglob('*'):
    if fp.is_file():  # Check if it's a file
        # Cut off the beginning of the filepath since it's unnecessary for our dataframe (fps = file path shortened)
        main_path = Path(main_path_str)
        fps = fp.relative_to(main_path)
        
        # Get the filename itself (without its containing folders)
        fn = fp.name
        # now we actually open the waveform here
        # Check if it's a .txt or .mat file
        try:
            if fp.suffix == '.mat':
                mat = scipy.io.loadmat(fp)
                if 'wf' in mat:
                    wf = np.squeeze(mat['wf'])
                else: 
                    print(f"Not sure how to process {fp}")
            if fp.suffix == '.txt':
                wf = np.loadtxt(fp)
            # Let's make sure this waveform is a 1D array
            if len(wf.shape) > 1:
                if fn == 'cricket_177.txt':
                    wf = wf[:, 1]
                    print(wf)
                print(f"Waveform from {fps} isn't 1D!")
        except:
            f"Uh oh! Issue when loading {fp}"
            
        # if str(fps).split("\\")[1]=='Tree Cricket':
        #     plot_mags()
        
        
        # try and get the species name
        fn_species = fn.split("_")[0]
        
        match fn_species:
            case 'anole':
                species = "Anole"
            case 'cricket':
                species = "Cricket"
            case 'human':
                species = "Human"
            case 'owl':
                species = "Owl"
            case _:
                species = None
        
        if species != "Owl":
            sr = 44100
        else:
            sr = None
            
                
        # add everything to our df dict
        data['filepath'].append(fps)
        data['wf'].append(wf)
        data['species'].append(species)
        data['sr'].append(sr)

# turn this into a pandas dataframe
df = pd.DataFrame(data)
# save this as an hdf5 file
df.to_hdf('soae_data.h5', key='df', mode='w')

        
    
    

        





