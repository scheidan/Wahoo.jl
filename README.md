# Wahoo

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://scheidan.github.io/Wahoo.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://scheidan.github.io/Wahoo.jl/dev/)
[![Build Status](https://github.com/scheidan/Wahoo.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/scheidan/Wahoo.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/scheidan/Wahoo.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/scheidan/Wahoo.jl)

## Installation

`] add git@github.com:scheidan/Wahoo.jl.git`

## Usage

Minimal example using example data that come with `Wahoo.jl`:

```Julia
using Wahoo

import GeoArrays
using DelimitedFiles: readdlm

# This is optional. If *both* packages are imported,
# some computations use the GPU. Otherwise, the CPU is used for everything.
import CUDA
import cuDNN


# -----------
# 1) Read data
# -----------

# Read example data that come with Wahoo
pathdata = joinpath(pkgdir(Wahoo), "example_data")

# -- bathymetry

bathymetry_map = GeoArrays.read(joinpath(pathdata, "bathymetry_200m.tif"))
GeoArrays.bbox(bathymetry_map)           # shows coordinates of corners


# -- depth observationa

# Vector{Float64}
depth_obs = readdlm(joinpath(pathdata, "depth_observations.csv"), ',', header=true)[1][:,2]


# -- acoustic observations

# Vector{Tuple{Float64, Float64}}
moorings = readdlm(joinpath(pathdata, "acoustic_moorings.csv"), ',', header=true)[1]
acoustic_pos = tuple.(moorings[:,2], moorings[:,3])

# Matrix{Int}: n_receivers x n_time steps
acoustic_obs = readdlm(joinpath(pathdata, "acoustic_observations.csv"), ',', header=true)[1][:,2:3]
acoustic_obs = Int.(acoustic_obs')


# -----------
# 2) Define parameters
# -----------

# initial values: Matrix{Float64}
p0 = zeros(size(bathymetry_map)[1], size(bathymetry_map)[2])
idx = GeoArrays.indices(bathymetry_map, (709757.111649658, 6.26772603565296e6)) # last known location of the fish
bathymetry_map[idx]
p0[idx] = 1

tsave = 1:5:720             # time steps to save
h = 200                     # spatial resolution [m]
D = 200^2                   # Diffusion coefficient, i.e. variance of one time step movement [m^2]


# -----------
# 3) Define parameters
# -----------

res = track(p0, bathymetry_map; tsave = tsave,
            D = D, h = h,
            depth_obs = depth_obs,
            acoustic_obs = acoustic_obs, acoustic_pos = acoustic_pos,
            smoothing = true);

# Resulting probabilities
# Array{Float32, 4}: Ny × Nx × 1 × time
res.pos_filter       # Prob(s_t | y_{1...t})
res.pos_smoother     # Prob(s_t | y_{1...T})
res.tsave            # time points
```

To use GPU for compuations, the packages `CUDA.jl` _and_ `cuDNN.jl` must be imported.


## References

The filter and smoother implementation is based on:

Thygesen, Uffe Høgsbro, Martin Wæver Pedersen, and Henrik
Madsen. 2009. “Geolocating Fish Using Hidden Markov Models and Data Storage Tags.” In Tagging and Tracking of Marine Animals with Electronic Devices, 277–93. Dordrecht: Springer Netherlands. https://doi.org/10.1007/978-1-4020-9640-2_17.
