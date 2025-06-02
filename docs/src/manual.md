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

todo


## Visualizations

todo
