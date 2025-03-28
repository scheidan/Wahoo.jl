# Data Description

This are simulated example data!


### Bathymetry

Geotiff defining the bathymetry with 200m spatial
resolution. Depth is in meters.

The bathymetry data is subset in lower resolution derived from the following survey:

Howe JA, Anderton R, Arosio R, et al. The seabed geomorphology and geological structure of the Firth of Lorn, western Scotland, UK, as revealed by multibeam echo-sounder survey. Earth and Environmental Science Transactions of the Royal Society of Edinburgh. 2014;105(4):273-284. doi:10.1017/S1755691015000146 

### Acoustic Observations

The signals are to be interpreted as:
- `0 = 'no detection'`
- `1 = 'detection'`
- `-1 = 'inactive'`

The locations of the receivers are in the `acoustic_moorings.csv`.


### Depth Observations

Depth measured in meters.
