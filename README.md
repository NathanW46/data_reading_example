# GPSANS Beam Data Plotter

This Python script processes neutron beam data from the MNO_GPSANS .txt files. It reads run files, maps detector bin IDs to physical X,Y positions using a lookup table, then generates 1D and 2D histograms of the counts distribution. Back tubes are shifted to be inline with front tubes for plotting.

---

## ️Structure

```
GPSANS_data_reading/
├── DATA/
│   ├── DetIDmap.csv               # Lookup table for Detector ID to XY-position
│   ├── MNO_GPSANS_<run>.txt       # Run data files 
├── example_beam_plot.py           # plotting script
└── requirements.txt               # dependencies
```

---

## Setup Instructions

### Clone the Repository

```bash
git clone https://github.com/NathanW46/GPSANS_data_reading
cd GPSANS_data_reading
```


### Install Dependencies

```bash
pip install -r requirements.txt
```

## Add Data files

Add MNO data from n-n' experiments to DATA directory.
A sample file has been provided based on run 89947 from the 2024 experiment due to the filesize limit.

## Running the Script

Edit the script to specify the run numbers you want to plot:

```python
runs = [89947]  # Or multiple runs like [89947, 89948] or ['sample']
```

Then run:

```bash
python example_beam_plot.py
```

This will:
- Load the detector data
- Convert detector tube/pixel IDs to X/Y coordinates (in meters)
- Generate:
  - X projection histogram
  - Y projection histogram
  - 2D histogram of (X, Y) positions
 
Everything should work on windows powershell as well (probably...)
