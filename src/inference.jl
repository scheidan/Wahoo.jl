
export track

import NNlib
import GeoArrays
import ProgressMeter

"""
Compute convolution kernel and the required number of hope per time steps

D: diffusion coefficient
h: spatial resolution
"""
function make_kernel(;D, h)

    # compute 1/Δ, so that 4*D*Δ/h^2 < 1
    n_hops = ceil(Int, 4*D/(h^2*0.99))
    Δ = 1/n_hops
    @assert 4*D*Δ/h^2 < 1

    # convolution kernel
    H = Float32[0 0 0;
                0 1 0;
                0 0 0]

    H = H + D*Δ/h^2 .* [0  1  0;
                        1 -4  1;
                        0  1  0]
    H = reshape(H, 3, 3, 1, 1)

    H, n_hops
end



"""
Maps time step to the index of `pos_filter` and `pos_smoother`
"""
time2index(t, tsave) = findfirst(==(t), tsave)


# ---
# filter algorithm

function run_filter(pos_init, H,
                    bathymetry,
                    signals, distances;
                    hops_per_step,
                    tsave=1:100,
                    show_progressbar)

    tmax = maximum(tsave)

    pos_init ./= sum(pos_init)
    nx, ny = size(bathymetry)[1:2]

    pos = similar(H, nx, ny, 1, length(tsave))

    pos_tmp = similar(H, nx, ny, 1, 1)
    pos_tmp[:,:,1,1] = pos_init
    pos[:,:,1,1] .= pos_tmp[:,:,1,1]

    pmeter = ProgressMeter.Progress(tmax - 1; desc = "Filtering...:",
                                    output = stderr, enabled = show_progressbar)

    log_p = zero(eltype(H))

    for t in 2:tmax

        # --- solve focker plank
        for k in 1:hops_per_step
            pos_tmp[:,:,1,1] = NNlib.conv(pos_tmp[:,:,1:1,1:1], H, pad=1)
        end

        # --- add signals
        for (k, sig) in enumerate(signals)
            pos_tmp[:,:,1,1] .*= sig.p_obs.(sig.observations[t], bathymetry, distances[:,:,k])
        end


        # --- normalize
        Z = sum(pos_tmp[:,:,1,1])
        isfinite(Z) ||
            error("No solution at time point $(t)! Check for data incompatibilities.")
        pos_tmp[:,:,1,1] .= pos_tmp[:,:,1,1] ./ Z

        log_p += log(Z)

        # --- save results
        if t in tsave
            pos[:,:,1,time2index(t, tsave)] = pos_tmp[:,:,1,1]
        end

        ProgressMeter.next!(pmeter)
    end

    return pos, log_p
end


# ---
# smoothing algorithm


"""
Define division by zero equal to zero.
"""
divzero(a, b) = iszero(b) ? zero(a/b) : a/b


function run_smoother(pos_filter, H,
                      bathymetry,
                      signals, distances;
                      hops_per_step,
                      tsave=1:100,
                      show_progressbar)

    ext = Base.get_extension(@__MODULE__, :CUDAExt)

    tmax = maximum(tsave)
    n_tsave = length(tsave)

    nx, ny = size(pos_filter)[1:2]

    # hold smoothed results, P(s_t | y_{1:T})
    pos_smoother = similar(pos_filter)
    pos_smoother_tmp = similar(pos_filter, nx, ny, 1)

    pos_smoother_tmp[:,:,1] .= pos_smoother[:,:,1,end] .= pos_filter[:,:,1,end]

    # distribution of resdence time over all time steps
    residence_dist = similar(pos_filter, nx, ny)
    residence_dist[:,:] .= pos_filter[:,:,1,end]

    pmeter = ProgressMeter.Progress(n_tsave - 1; desc = "Smoothing...:",
                                    output = stderr, enabled = show_progressbar)

    # jth jump back in time
    for j in (n_tsave-1):-1:1

        tsave_jump = tsave[j]:1:tsave[j+1]    # "internal", reconstructed time-steps

        # -----------
        # 1) recompute all filter steps between j and j+1

        # holds reconstructed filter results (!!! better move out of loop?)
        pos_filter_jump = similar(pos_filter, nx, ny, 1, length(tsave_jump))         # P(s_t | y_{1:t})
        pos_filter_jump_no_obs = similar(pos_filter, nx, ny, 1, length(tsave_jump))   # P(s_{t+1} | y_{1:t})

        pos_filter_jump[:,:,1,1] .= pos_filter_jump_no_obs[:,:,1,1] .= pos_filter[:,:,1,time2index(tsave_jump[1], tsave)]


        for (i,t) in enumerate(tsave_jump[1:(end-1)])

            pos_filter_jump[:,:,1,i+1] .= pos_filter_jump[:,:,1,i]

            # --- solve focker plank
            for k in 1:hops_per_step
                pos_filter_jump[:,:,1,(i+1):(i+1)] = NNlib.conv(pos_filter_jump[:,:,1:1,(i+1):(i+1)], H, pad=1)
            end

            # --- save P(s_{t+1} | y_{1:t})
            pos_filter_jump_no_obs[:,:,1,i+1] .= pos_filter_jump[:,:,1,i+1]

            # --- add signals
            for (k, sig) in enumerate(signals)
                pos_filter_jump[:,:,1,1] .*= sig.p_obs.(sig.observations[t], bathymetry, distances[:,:,k])
            end

            # --- normalize
            Z = sum(pos_filter_jump[:,:,1,i+1])
            isfinite(Z) ||
                error("No solution at time point $(t)!")
            pos_filter_jump[:,:,1,i+1] .= pos_filter_jump[:,:,1,i+1] ./ Z


        end

        # -----------
        # 2) backward smoothing

        pos_smoother_tmp[:,:,1,1] .= pos_smoother[:,:,1,time2index(tsave_jump[end], tsave)]


        for (i, t) in enumerate(reverse(tsave_jump)[2:end])
            idx = length(tsave_jump) - i + 1 # index of pos_filter_jump

            # treat division by zero as special case
            pos_smoother_tmp[:,:,1,1] .= divzero.(pos_smoother_tmp[:,:,1,1], pos_filter_jump_no_obs[:,:,1,idx])

            # --- solve focker plank backwards
            # K = rot180(H) = H if no advections
            for k in 1:hops_per_step
                pos_smoother_tmp[:,:,1,1] = NNlib.conv(pos_smoother_tmp[:,:,1:1,1:1], H, pad=1)
            end

            pos_smoother_tmp[:,:,1,1] .=  pos_filter_jump[:,:,1,idx-1] .* pos_smoother_tmp[:,:,1,1]
            pos_smoother_tmp[:,:,1,1] ./= sum(pos_smoother_tmp[:,:,1,1])

            residence_dist .+= pos_filter[:,:,1,1]

            # --- save
            if t in tsave
                pos_smoother[:,:,1,time2index(t, tsave)] = pos_smoother_tmp[:,:,1,1]
            end

        end

        ProgressMeter.next!(pmeter)

    end

    # normalize
    residence_dist ./= sum(residence_dist)

    return (pos_smoother, residence_dist)
end



"""
Tracks the location of the fish

```
track(pos_init, bathymetry; depth_obs, acoustic_obs, acoustic_pos, tsave, D, h, smoothing, use_gpu)
```

Uses forward filtering based on a diffusion model and optionally smoothing.

# Arguments

- `pos_init::Matrix`: Initial probability distribution of the fish positions
- `bathymetry`: Bathymetric data of the environment
- `depth_obs`: A vector of observed depths at each time step
- `acoustic_obs`: A matrix of  acoustic observations
- `acoustic_pos`: Vector of tuples containing the positions of the acoustic receivers
- `tsave::AbstractVector`: Time steps at which to save the probability map
- `D`: Diffusion coefficient, i.e. variance for one time step movement [m^2]
- `h`: spatial resolution [m]
- `smoothing`: Boolean flag to enable smoothing
- `show_progressbar = !is_logging(stderr)`: defaults to `true` for interactive use.

"""
function track(pos_init::Matrix, bathymetry::GeoArrays.GeoArray;
               depth_obs::Vector,
               acoustic_obs::Matrix{Int}, acoustic_pos::Vector,
               tsave::AbstractVector = 1:100,
               D, h=1,
               smoothing::Bool=false,
               show_progressbar::Bool = !is_logging(stderr))

    @assert size(pos_init) == size(bathymetry)[1:2]
    @assert size(acoustic_obs, 1) == length(acoustic_pos)

    nx, ny = size(pos_init)


    # convolution kerel
    H, n_hops = make_kernel(D=D, h=h)

    # precompute distances to sensors
    distances = build_distances(signals, bathymetry, h)

    # run filter

    cudaext = Base.get_extension(@__MODULE__, :CUDAExt)
    if !isnothing(cudaext) # check if we have CUDA.jl loaded
        H, bathymetry, dist_acoustic = cudaext.move_to_GPU(H, bathymetry, distances, tsave)
    else                   # use CPU
        bathymetry = Float64.(bathymetry[:,:,1])
    end

    pos_filter, log_p = run_filter(pos_init, H, bathymetry,
                            signals, distances;
                            hops_per_step = n_hops, tsave = tsave,
                            show_progressbar = show_progressbar)

    if smoothing

        pos_smoother, residence_dist = run_smoother(pos_filter, H,
                                                    bathymetry,
                                                    signals,
                                                    distances;
                                                    hops_per_step = n_hops,
                                                    tsave = tsave,
                                                    show_progressbar = show_progressbar)

        return (pos_smoother = Array(pos_smoother),
                pos_filter = Array(pos_filter),
                residence_dist = Array(residence_dist),
                log_p = log_p,
                tsave = tsave)
    else
        return  (pos_filter = Array(pos_filter), log_p = log_p, tsave = tsave)
    end

end

# Little helper:
# Check if a stream is logged. From: https://github.com/timholy/ProgressMeter.jl
is_logging(io) = isa(io, Base.TTY) == false || (get(ENV, "CI", nothing) == "true")
