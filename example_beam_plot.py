import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pathlib
from tqdm import tqdm
import scipy.stats as stats
from scipy.optimize import curve_fit

# pd.options.mode.copy_on_write = True

# Read in lookup file
data_dir = pathlib.Path.cwd() / 'DATA'
lookupfile = data_dir / 'DetIDmap.csv'
lookup_table = pd.read_csv(lookupfile, skiprows=2, index_col=None)

# convert bin centers to edges
def bin_centers_to_edges(bin_centers):
    """Convert bin centers to bin edges."""
    bin_width = np.diff(bin_centers)[0]  
    bin_edges = np.concatenate(([bin_centers[0] - bin_width / 2],
                                bin_centers + bin_width / 2))
    return bin_edges

# Gaussian function for fitting
def gaussian(x, a, b, c, d):
    return a*np.exp(-((x-b)**2)/(2*c**2)) + d


def plot(run):
    # read in data to dataframe
    df = pd.read_csv(data_dir / f'MNO_GPSANS_{run}.txt', sep=',', skiprows=4)
    # keep only GPSANS data
    df = df[df[' Detector ID'] < 197]
    # reset row index
    df.reset_index(drop=True,inplace=True)

    # Create new columns for tubes and pixels
    df['Tubes'] = df[' Detector ID']
    df['Pixels'] = df[' Value']

    # combine front and back tubes
    df.loc[df['Tubes'] >= 100, 'Tubes'] -= 96
    
    #  convert tubes/pixels to x,y in meters
    tube_positions = dict(zip(lookup_table['#Detector ID Number'], lookup_table['Values associated with Notes']))
    df['X'] = df[' Detector ID'].map(tube_positions)
    df['Y'] = df[' Value']*0.004086

    # define ROI
    if run == 89947 or run == 'sample': # full beam
        roi = [-0.52525, 0.52525, -0.52525, 0.52525] # x_min, x_max, y_min, y_max
    else:
        # 20 cm ROI
        roi = [-0.1, 0.1, -0.1, 0.1]

    # get bin centers from lookup_table file (so that the binsize is constant)
    xbins = np.sort(lookup_table[lookup_table['...'] < 0]['Values associated with Notes'].values)
    xbins = [x for x in xbins if roi[0] < x < roi[1]]
    xedges = bin_centers_to_edges(xbins)
    
    # create 256 pixels from -0.52525 to 0.52525 m
    ybins = np.linspace(-0.52525, 0.52525, 256, endpoint=True)
    ybins = [y for y in ybins if roi[2] < y < roi[3]]
    yedges = bin_centers_to_edges(ybins)




    x = df[(df['X'] >= roi[0]) & (df['X'] <= roi[1])]['X']
    y = df[(df['Y'] >= roi[0]) & (df['Y'] <= roi[1])]['Y']

    #gaussian fit x
    x_h = np.histogram(x, xedges)
    xparams, xcov = curve_fit(gaussian, xbins, x_h[0], p0=[max(x_h[0]), x.mean(), x.std(), 100])
    x_fit = gaussian(xbins, *xparams)

    #gaussian fit y
    y_h = np.histogram(y, yedges)
    yparams, ycov = curve_fit(gaussian, ybins, y_h[0], p0=[max(y_h[0]), y.mean(), y.std(), 100])
    y_fit = gaussian(ybins, *yparams)

    fig, (ax, x_ax, y_ax) = plt.subplots(1, 3, figsize=(14, 4.5), num=f"run {run}", constrained_layout=True)
    fig.set_constrained_layout_pads(
    h_pad=0.2, w_pad=0.2,
    hspace=0.1, wspace=0.1
    ) 
    # 2D histogram
    h = ax.hist2d(x, y, bins=(xbins,ybins))
    ax.set_ylabel(r'Y (m)', fontsize=12)
    ax.set_xlabel(r'X (m)', fontsize=12)
    ax.set_title(f'GPSANS Full Beam', fontsize=18)
    cbar = fig.colorbar(h[3], ax=ax, shrink=0.9)
    cbar.spacing = 'proportional'
    ax.set_aspect('equal')

    #plot x projection  
    x_ax.hist(x, bins=xbins, label='X Projection', alpha=0.7)
    x_ax.plot(xbins, x_fit, label='Gaussian Fit', color='red')
    x_ax.set_xlabel('X (m)', fontsize=12)
    x_ax.set_ylabel('Counts', fontsize=12)
    x_ax.set_title('X Projection', fontsize=14)
      
    #plot y projection  
    y_ax.hist(y, bins=ybins, label='Y Projection', alpha=0.7)
    y_ax.plot(ybins, y_fit, label='Gaussian Fit', color='red')
    y_ax.set_xlabel('Y (m)', fontsize=12)
    y_ax.set_ylabel('Counts', fontsize=12)
    y_ax.set_title('Y Projection', fontsize=14)

    
    


# Main -------------------------------------------------------------------------------------
# change this list to plot different runs
# runs = [89947, 89948, 89949]
runs = ['sample'] 
for run in runs:
    plot(run)

plt.show()
    
    
