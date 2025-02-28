module Wahoo

include("observation_models.jl")
include("inference.jl")
include("trajectory_sampling.jl")


"""
Tracks the location of the fish

```
track(;pos_init::Matrix,
       tsave::AbstractVector = 1:100,
       bathymetry::GeoArrays.GeoArray,
       observations::Vector,
       observation_models::Vector{Function},
       sensor_positions::Vector,
       spatial_resolution,
       movement_std,
       save_filter::Bool = false,
       n_trajectories::Int = 0,
       show_progressbar::Bool = !is_logging(stderr),
       precision = Float32)
```

Infers the location of the animal based on a diffusion model and smoothing.

# Keyword Arguments

- `pos_init::Matrix`: Initial probability distribution of the fish position
- `tsave::AbstractVector`: Time steps at which the probability map is saved.
- `bathymetry`: Bathymetric data as `GeoArray`
- `spatial_resolution`: the spatial resolution [m] of the `bathymetry`.
- `movement_std`: Standard deviation of the fish movement within one time step [m]
- `observations`: Vector holding all observations. Each element contains the observation of a separate sensor.
- `observation_models::Vector{Function}`: Vector containing the observation model for each sensor.
- `sensor_positions`: Vector of tuples of coordinates or `nothing`, i.e. `Vector{Union{Nothing, Tuple{Real, Real}}}`
- `save_filter`: if `true` the probabilities from the filter run are returned.
- `n_trajectories=0`: Number of trajectories to sample
- `show_progressbar = !is_logging(stderr)`: defaults to `true` for interactive use.
- `precision = Float32`: numerical floating point type used for computations

Note, the elements of the vectors `observations`, `observation_models`, and `sensor_positions` must be sorted in the same way, i.e.
the elements at the same position in the Vectors refer to the same sensor.

# Return

A named tuple with the following elements:
- `pos_smoother`: Smoothed probability distribution of the fish positions for all timesteps in `tsave`.
- `residence_dist`: Residence distribution.
- `trajectories`: Sampled trajectories if `n_trajectories` > 0, otherwise `nothing`.
- `log_p`: Log probability of the observations.
- `tsave`: Vector of time steps at which the results are saved.
- `pos_filter`: Filtered  probability distribution of the fish positions, included if `save_filter = true`.

"""
function track(;pos_init::Matrix, bathymetry::GeoArrays.GeoArray,
               observations::Vector,
               observation_models::Vector{Function},
               sensor_positions::Vector,
               tsave::AbstractVector = 1:100,
               spatial_resolution,
               movement_std,
               save_filter::Bool=false,
               n_trajectories::Int=0,
               show_progressbar::Bool = !is_logging(stderr),
               precision=Float32)

    @assert size(pos_init) == size(bathymetry)[1:2]

    nx, ny = size(pos_init)


    # convolution kerel
    H, n_hops = make_kernel(D=movement_std^2, h=spatial_resolution, precision=precision)

    # precompute distances to sensors
    distances = build_distances(sensor_positions, bathymetry, spatial_resolution)

    # run filter

    cudaext = Base.get_extension(@__MODULE__, :CUDAExt)
    if !isnothing(cudaext) # check if we have CUDA.jl loaded
        H, bathymetry, observations, distances = cudaext.move_to_GPU(H, bathymetry, observations, distances, precision)
    else                   # use CPU
        bathymetry = precision.(bathymetry[:,:,1])
        pos_init = precision.(pos_init)
    end

    @info "Using $precision for computations"

    pos_filter, log_p = run_filter(pos_init, H,
                                   bathymetry,
                                   observations,
                                   observation_models,
                                   distances;
                                   hops_per_step = n_hops, tsave = tsave,
                                   show_progressbar = show_progressbar)


    pos_smoother, residence_dist = run_smoother(pos_filter, H,
                                                bathymetry,
                                                observations,
                                                observation_models,
                                                distances;
                                                hops_per_step = n_hops,
                                                tsave = tsave,
                                                show_progressbar = show_progressbar)

    if n_trajectories > 0
        trajectories =  sample_trajectories(pos_filter, H,
                                            bathymetry,
                                            observations,
                                            observation_models,
                                            distances;
                                            tsave = tsave,
                                            n_trajectories =  n_trajectories,
                                            hops_per_step = n_hops,
                                            show_progressbar = show_progressbar)
    else
        trajectories = nothing
    end

    if save_filter
        return (pos_smoother = Array(pos_smoother),
                pos_filter = Array(pos_filter),
                residence_dist = Array(residence_dist),
                trajectories = trajectories,
                log_p = Array(log_p),
                tsave = tsave)
    else
        return (pos_smoother = Array(pos_smoother),
                residence_dist = Array(residence_dist),
                trajectories = trajectories,
                log_p = Array(log_p),
                tsave = tsave)
    end

end

# Little helper:
# Check if a stream is logged. From: https://github.com/timholy/ProgressMeter.jl
is_logging(io) = isa(io, Base.TTY) == false || (get(ENV, "CI", nothing) == "true")


end
