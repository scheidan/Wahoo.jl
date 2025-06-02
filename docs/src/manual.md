# Manual

## Installation

Install Julia, at least version 1.9. If you plan to use GPU support,
it is advisable to use the latest stable version.

Within Julia install Wahoo:

`] Wahoo`


## Defining observation models


The user must define an observation model for every sensor. This is a
function that computes the probability (density) of
the observed signal given the location, `p(y_t | s_t)`.

The function must have the following signature:
```
 p_obs(signals, t::Int, bathymetry_depth::Number, dist::Number)
```
where `bathymetry_depth` is the water depth at `s_t` and `dist` is the Euclidean
distance from `s_t` to the sensor location. Note, the function must
accept all four arguments, even if some are not used.

Note that if GPU use is planned, the function must be type-stable!



## Using GPU

To use the GPU for computations, the packages `CUDA.jl` _and_ `cuDNN.jl`
must be imported. Currently, only CUDA-compatible GPUs are supported.


## Export results

The results can be exported in different ways. We recommend
`hdf5` if interoperability is required or `JLD2`  for postprocessing in
Julia.

### HDF5

[HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) is a generic format for array like data that can be read from
most languages. The function below export the result of `track` as
hdf5. Note, you must install `HDF5.jl` additionally to `Wahoo`.

```Julia
import HDF5

"""
Write the result from `Wahoo.track()` to `file` in the hdf5 format.
"""
function export_hdf5(res, file)

    # Open (and create) the HDF5 file in write mode
    HDF5.h5open(file, "w") do f

        # --- model outputs
        write(f, "timesteps", collect(res.tsave))

        f["pos", compress=3] =  dropdims(res.pos_smoother, dims=3)
        HDF5.attributes(f["pos"])["dimensions"] = "(y_coord, x_coord, timestep) of size $(size(res.pos_smoother))"

        f["residence_distribution", compress=3] = res.residence_dist
        HDF5.attributes(f["residence_distribution"])["dimensions"] = "(y_coord, x_coord) of size $(size(res.residence_dist))"

        if isdefined(res, :pos_filter)
            f["pos_filtered", compress=3] =  dropdims(res.pos_filter, dims=3)
            HDF5.attributes(f["pos_filtered"])["dimensions"] = "(y_coord, x_coord, timestep) of size $(size(res.pos_filter))"
        end

        write(f, "log_p", res.log_p)

        if !isnothing(res.trajectories)
            # Create a group for the 'trajectories' data
            grp = HDF5.create_group(f, "trajectories")
            HDF5.attributes(f["trajectories"])["description"] = "Each trajectory is a 2d array of shape (2 x time)." *
                                                                " Note, it contains all time steps, i.e. from 1:maximum(timesteps)." *
                                                                " The first row are the y-coordinates, the second the x-coordinates."

            # Write each matrix in the track array as its own dataset
            for (i, tr) in enumerate(res.trajectories)
                write(grp, "traj$(i)", tr)
            end
        end
    end
end
```

##### Read HDF5 with Python

The code below give an example who the data could be read with
Python. Note, the order of the indices differ.

```python
# /// script
# dependencies = [
#   "h5py",
#   "numpy"
# ]
# ///

import h5py
import numpy as np

# -------
# load data from hdf5 file

with h5py.File('Wahoo_results.hdf5', 'r') as f:
    print("structure of the file:")
    print(f.keys())
    print(f['trajectories'].keys())

    pos = f['pos'][:]
    timesteps = f['timesteps'][:]
    trajectories_group = f['trajectories']
    trajectories = [trajectories_group[key][:] for key in trajectories_group.keys()]

    residence_distribution = f['residence_distribution'][:]


print('\nProbabilities of the fish position, saved at `timesteps`:')
print('shape (time, x, y):', np.shape(pos))
# For example, pos[:,:,2] is the distribution of the position for at time = timesteps[2]

print('\nAll trajectories:')
for i, arr in enumerate(trajectories, start=1):
    print(f" - trajectory {i} shape: {np.shape(arr)}")
# For example, trajectories[1][2] is the position for time = 3
```


### JLD2

[JLD2](https://github.com/JuliaIO/JLD2.jl) saves and loads Julia data
structures in a format comprising a subset of HDF5. It is the
recommended format if the data will be used by another Julia script. Note, `JLD2.jl` and
`CodecZlib.jl` must be installed additionally to `Wahoo`.

The function below stores the inference results together with the corresponding inputs.

```julia
import JLD2
import CodecZlib                # for compression

function export_jld2(res, file; bathymetry, spatial_resolution,
                     acoustic_obs, acoustic_pos, depth_obs)

    # Open (and create) the JLD2 file in write mode
    JLD2.jldopen(file, "w"; compress = true) do f

        # --- model inputs
        f["bathymetry"] = bathymetry
        f["acoustic_obs"] = acoustic_obs
        f["acoustic_pos"] = acoustic_pos
        f["depth_obs"] = depth_obs
        f["spatial_resolution"] = spatial_resolution

        # --- model outputs
        f["timesteps"] = collect(res.tsave)
        f["pos"] = res.pos_smoother
        f["residence_distribution"] = res.residence_dist
        f["log_p"] = res.log_p

        if isdefined(res, :pos_filter)
            f["pos_filtered"] =  res.pos_filter
        end

        if !isnothing(res.trajectories)
            f["trajectories"] = res.trajectories
        end
    end
    file
end
```

## Visualizations

todo
