import pickle

filename = "wf - V&D fig 4, loc=0.1, glob=0, sr=128.pkl"
filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Chris's Pickle Jar\\"
with open(filepath + filename, 'rb') as picklefile:
    wf = pickle.load(picklefile)
    
print(wf)