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
       smoother::Bool = true,
       filter::Bool = false,
       n_trajectories::Int = 0,
       show_progressbar::Bool = !is_logging(stderr),
       precision = Float32)
```

Infers the location of the animal based on a diffusion model and smoothing.

# Keyword Arguments

- `pos_init::Matrix`: Initial probability distribution of the fish position.
- `tsave::AbstractVector`: Time steps at which the probability map is saved.
- `bathymetry`: Bathymetric data as `GeoArray`.
- `spatial_resolution`: the spatial resolution [m] of the `bathymetry`.
- `movement_std`: Standard deviation of the fish movement within one time step [m].
- `observations`: Vector holding all observations. Each element contains the observation of a separate sensor.
- `observation_models::Vector{Function}`: Vector containing the observation model for each sensor.
- `sensor_positions`: Vector of tuples of coordinates or `nothing`, i.e. `Vector{Union{Nothing, Tuple{Real, Real}}}`.
- `smoother = true`: if `true` the probabilities from the smoother run are returned.
- `filter = false`: if `true` the probabilities from the filter run are returned.
- `n_trajectories = 0`: Number of trajectories to sample.
- `show_progressbar = !is_logging(stderr)`: defaults to `true` for interactive use.
- `precision = Float32`: numerical floating point type used for computations.

Note, the elements of the vectors `observations`, `observation_models`, and `sensor_positions` must be sorted in the same way, i.e.
the elements at the same position in the Vectors refer to the same sensor.

# Return

A named tuple with the following elements:
- `log_p`: Log probability of the observations, ``\\log p(\\mathbf{y}_{1:T})``.
- `tsave`: Vector of time steps at which the results are saved.
- `trajectories`: Sampled trajectories if `n_trajectories` > 0, otherwise `nothing`.

Additionally, if `smoother = true`:
- `pos_smoother`: ``\\Pr(\\mathbf{s}_t \\mid \\mathbf{y}_{1:T})``, the smoothed probability distribution of the fish positions for all timesteps in `tsave`.
- `residence_dist`: Residence distribution, ``\\frac{1}{T}\\sum_{t=1}^{T}\\Pr(\\mathbf{s}_t\\mid \\mathbf{y}_{1:T})``.

Additionally, if `filter = true`:
- `pos_filter`: ``\\Pr(\\mathbf{s}_t \\mid \\mathbf{y}_{1:t})``, the filtered probability distribution of the fish positions.


"""
function track(;pos_init::Matrix, bathymetry::GeoArrays.GeoArray,
               observations::Vector,
               observation_models::Vector{Function},
               sensor_positions::Vector,
               tsave::AbstractVector = 1:100,
               spatial_resolution,
               movement_std,
               filter::Bool = false,
               smoother::Bool = true,
               n_trajectories::Int = 0,
               show_progressbar::Bool = !is_logging(stderr),
               precision=Float32)

    # Ensure that observations, observation_models, and sensor_positions have the same length
    if !(length(observations) == length(observation_models) == length(sensor_positions))
        error("The lengths of `observations` (length = $(length(observations))), `observation_models` " *
            "(length = $(length(observation_models))), and `sensor_positions` (length = $(length(sensor_positions))) must be the same.")
    end

    # Check if pos_init and bathymetry sizes match
    if size(pos_init) != size(bathymetry)[1:2]
        error("The size of `pos_init` $(size(pos_init)) must match the first two dimensions of `bathymetry` $(size(bathymetry)[1:2]).")
    end

    nx, ny = size(pos_init)

    # convolution kerel
    H, n_hops = make_kernel(D = movement_std^2/2, h = spatial_resolution, precision = precision)

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

    if smoother
        pos_smoother, residence_dist = run_smoother(pos_filter, H,
                                                    bathymetry,
                                                    observations,
                                                    observation_models,
                                                    distances;
                                                    hops_per_step = n_hops,
                                                    tsave = tsave,
                                                    show_progressbar = show_progressbar)
    end

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


    results =  (log_p = Array(log_p),
                tsave = tsave,
                trajectories = trajectories)
    if smoother
        results = merge(results, (pos_smoother = Array(pos_smoother),
                                  residence_dist = Array(residence_dist)))
    end

    if filter
        results = merge(results, (pos_filter = Array(pos_filter),))
    end

    return results
end

# Little helper:
# Check if a stream is logged. From: https://github.com/timholy/ProgressMeter.jl
is_logging(io) = isa(io, Base.TTY) == false || (get(ENV, "CI", nothing) == "true")


end
