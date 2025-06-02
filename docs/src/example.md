# Example

The following example uses example data that come with `Wahoo.jl`.

```julia
using Wahoo

import GeoArrays
using DelimitedFiles: readdlm

# This is optional. If *both* packages are imported,
# some computations use the GPU. Otherwise, the CPU is used for everything.
import CUDA
import cuDNN
```

### 1) Bathymetry

Load the bathymetry map which provides depth information for each grid cell.

```julia
pathdata = joinpath(pkgdir(Wahoo), "example_data")
bathymetry_map = GeoArrays.read(joinpath(pathdata, "bathymetry_200m.tif"))
GeoArrays.bbox(bathymetry_map)
```

### 2) Depth observations

Read the depth measurements and define the exponential
likelihood model. This model implies the fish is more likely close to the
seabed.

```julia
# read depth signals
depth_signals = readdlm(joinpath(pathdata, "depth_observations.csv"), ',', header=true)[1][:,2]

# define likelihood
function p_obs_depth_exponential(signals, t, waterdepth, dist; scale=30f0)
    signal = signals[t]
    if signal > waterdepth         # water is too shallow
        return zero(waterdepth)
    else
        # exponential
        Z = 1 - exp(-waterdepth/scale) # normalisation due to truncation
        exp(-(waterdepth - signal)/scale)/(scale * Z)
    end
end
```

### 3) Acoustic observations

Load acoustic detection data and specify the probability model
for acoustic signals given the distance.

```julia
# read acoustic signals
acoustic_signals = readdlm(joinpath(pathdata, "acoustic_observations.csv"), ',', header=true)[1][:,2:3]
acoustic_signals = Int.(acoustic_signals')
acoustic_obs = [acoustic_signals[1,:], acoustic_signals[2,:]]

# read sensor positions
moorings = readdlm(joinpath(pathdata, "acoustic_moorings.csv"), ',', header=true)[1]
acoustic_pos = tuple.(moorings[:,2], moorings[:,3])

# define likelihood
function p_obs_acoustic(signals, t::Int, depth::Number, distance::Number)
    Wahoo.p_acoustic_sigmoid(signals[t], depth, distance)
end
```


### 4) Define parameters

We define initial distribution of the fish location and configures the model parameters
such as time steps, movement capabilities of the fish, and spatial
resolution of the bathymetry.

```julia

# initial values: Matrix{Float64}
p0 = zeros(size(bathymetry_map)[1], size(bathymetry_map)[2])
idx = GeoArrays.indices(bathymetry_map, (709757.111649658, 6.26772603565296e6)) # last known location of the fish
p0[idx] = 1

tsave = 1:2:720             # time steps to save
movement_std = 100          # standard deviation of the fish movement for one time step [m]
spatial_resolution = 200    # spatial resolution [m]
```

### 5) Run inference

Finally, we run the model inference combine all observations and
assumptions.

```julia
res = track(pos_init = p0, bathymetry = bathymetry_map,
            tsave = tsave,
            spatial_resolution = spatial_resolution,
            movement_std = movement_std,
            observations = [depth_signals, acoustic_obs...],
            observation_models = [p_obs_depth_exponential, acoustic_obs_models...],
            sensor_positions = [nothing, acoustic_pos...],
            n_trajectories = 2)

# Resulting probabilities
# Array{Float32, 4}: Ny × Nx × 1 × time
res.pos_smoother     # Prob(s_t | y_{1...T})
res.pos_filter       # Prob(s_t | y_{1...t}), only if `save_filter = true` was used
res.residence_dist   # 1/T Σ Prob(s_t | y_{1...T})
res.trajectories     # Vector of trajectories sampled from Prob(s_{1...T} | y_{1...T})
res.log_p            # Prob(y_t)
res.tsave            # time points
```

The result of the smoother `res.pos_smoother` can be visualized:

```@raw html
<img src="../assets/smoothing_animated.gif" style="width: 75%; height: auto;" />
```

Note, the code for visualization is not part of Wahoo.
