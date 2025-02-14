module Wahoo

include("observation_models.jl")
include("inference.jl")
include("trajectory_sampling.jl")


"""
Tracks the location of the fish

```
track(;pos_init, bathymetry, observations,  observation_models, sensor_positions, tsave, h, D, smoothing=true)
```

Uses forward filtering based on a diffusion model and optionally smoothing.

# Arguments

- `pos_init::Matrix`: Initial probability distribution of the fish positions
- `bathymetry`: Bathymetric data of the environment
- `observations`: Vector of all observations
- `sensor_positions`: Vector of tuples or `nothing` containing the positions of the receivers
- `tsave::AbstractVector`: Time steps at which to save the probability map
- `h`: spatial resolution [m]
- `D`: Diffusion coefficient, i.e. variance for one time step movement [m^2]
- `smoothing`: Boolean flag to enable smoothing
- `n_trajectories=0`: Number of trajectories to sample
- `show_progressbar = !is_logging(stderr)`: defaults to `true` for interactive use.
- `precision = Float32`: numerical floating point type used for computations

"""
function track(;pos_init::Matrix, bathymetry::GeoArrays.GeoArray,
               observations::Vector,
               observation_models::Vector,
               sensor_positions::Vector,
               tsave::AbstractVector = 1:100,
               h, D,
               smoothing::Bool=true,
               n_trajectories::Int=0,
               show_progressbar::Bool = !is_logging(stderr),
               precision=Float32)

    @assert size(pos_init) == size(bathymetry)[1:2]

    nx, ny = size(pos_init)


    # convolution kerel
    H, n_hops = make_kernel(D=D, h=h, precision=precision)

    # precompute distances to sensors
    distances = build_distances(sensor_positions, bathymetry, h)

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

    if smoothing

        pos_smoother, residence_dist = run_smoother(pos_filter, H,
                                                    bathymetry,
                                                    observations,
                                                    observation_models,
                                                    distances;
                                                    hops_per_step = n_hops,
                                                    tsave = tsave,
                                                    show_progressbar = show_progressbar)

        if n_trajectories >0
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

        return (pos_smoother = Array(pos_smoother),
                pos_filter = Array(pos_filter),
                residence_dist = Array(residence_dist),
                trajectories = trajectories,
                log_p = Array(log_p),
                tsave = tsave)
    else
        return  (pos_filter = Array(pos_filter), log_p = Array(log_p), tsave = tsave)
    end

end

# Little helper:
# Check if a stream is logged. From: https://github.com/timholy/ProgressMeter.jl
is_logging(io) = isa(io, Base.TTY) == false || (get(ENV, "CI", nothing) == "true")


end
