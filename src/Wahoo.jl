module Wahoo

export track

import NNlib
import GeoArrays

"""
Compute convolution kernel and the number of time steps

D: diffusion coefficient
h: spatial resolution
"""
function make_kernel(;D, h)

    # compute 1/Δ, so that 4*D*Δ/h^2 < 1
    n_hops = ceil(Int, 4*D/(h^2*0.99))
    Δ = 1f0/n_hops
    @assert 4*D*Δ/h^2 < 1

    # convolution kernel
    H = Float32[0 0 0;
                0 1 0;
                0 0 0]

    H = H + Float32(D*Δ/h^2) .* Float32[0  1  0;
                                        1 -4  1;
                                        0  1  0]
    H = reshape(H, 3, 3, 1, 1)

    H, n_hops
end


# ---
# depth likelihood

function p_depth(dobs, dmax)
    if dobs > dmax
        return zero(dobs)
    else
        # # uniform depth
        # one(dobs)/dmax

        # exponential
        scale = 30f0
        Z = 1 - exp(-dmax/scale) # normalisation due to truncation
        exp(-(dmax - dobs)/scale)/(scale * Z)
    end
end

# ---
# acoustic likelihood

function p_acoustic(obs::Int, dist::T; d0 = 400f0, k = 100f0)::T where T
    if obs == 0
        # 1/(1 + exp(-(dist - d0)/k))
        return NNlib.sigmoid((dist - d0)/k)
    end
    if obs == 1
        # 1 - 1/(1 + exp(-(dist - d0)/k))
        return 1 - NNlib.sigmoid((dist - d0)/k)
    end

    return one(T)
end



function build_distances(coord::Vector,
                         bathymetry::GeoArrays.GeoArray, h)

    ny, nx = size(bathymetry)[1:2]
    dist_acoustic = zeros(Float32, ny, nx, length(coord))

    for k in 1:length(coord)
        idx = GeoArrays.indices(bathymetry, coord[k])
        dist_acoustic[:,:,k] .= [sqrt((i-idx[1])^2 + (j-idx[2])^2) * h
                                 for i in 1:ny, j in 1:nx]
    end

    dist_acoustic
end


# ---
# filter algorithm

function run_filter!(pos, H,
                     bathymetry, depth_obs,
                     acoustic_obs, dist_acoustic;
                     hops_per_step,
                     tsave=1:100,
                     p_init)

    ext = Base.get_extension(@__MODULE__, :CUDAExt)

    tmax = maximum(tsave)

    p_init ./= sum(p_init)
    nx, ny = size(pos)[1:2]
    pos_tmp = similar(pos, nx, ny, 1, 1)
    pos_tmp[:,:,1,1] = p_init
    pos[:,:,1,1] .= pos_tmp[:,:,1,1]

    for t in 2:tmax

        # --- solve focker plank
        for k in 1:hops_per_step
            if !isnothing(ext) # check if we have CUDA.jl loaded
                pos_tmp[:,:,1,1] = ext.conv(pos_tmp[:,:,1:1,1:1], H)
            else
                pos_tmp[:,:,1,1] = NNlib.conv(pos_tmp[:,:,1:1,1:1], H, pad=1)
            end

        end

        # --- add depth signal
        pos_tmp[:,:,1,1] .*= p_depth.(depth_obs[t], bathymetry)

        # --- add accustic signal
        n_sensors = size(dist_acoustic, 3)
        for s in 1:n_sensors
            pos_tmp[:,:,1,1] .*= p_acoustic.(acoustic_obs[s, t], dist_acoustic[:,:,s])
        end

        # --- normalize
        Z = sum(pos_tmp[:,:,1,1])
        isfinite(Z) ||
            error("No solution at time point $(t)!")
        pos_tmp[:,:,1,1] .= pos_tmp[:,:,1,1] ./ Z

        # --- save results

        if t in tsave
            pos[:,:,1,time2index(t, tsave)] = pos_tmp[:,:,1,1]
        end
    end

end


# ---
# smoothing algorithm


"""
Define division by zero equal to zero.
"""
divzero(a, b) = iszero(b) ? zero(a/b) : a/b


"""
Maps time step to the index of `pos_filter` and `pos_smoother`
"""
time2index(t, tsave) = findfirst(==(t), tsave)


function run_smoother(pos_filter, H,
                      bathymetry, depth_obs,
                      acoustic_obs, dist_acoustic;
                      hops_per_step,
                      tsave=1:100)

    ext = Base.get_extension(@__MODULE__, :CUDAExt)

    tmax = maximum(tsave)
    n_tsave = length(tsave)

    nx, ny = size(pos_filter)[1:2]

    # hold smoothed results, P(s_t | y_{1:T})
    pos_smoother = similar(pos_filter)
    pos_smoother_tmp = similar(pos_filter, nx, ny, 1)

    pos_smoother_tmp[:,:,1] .= pos_smoother[:,:,1,end] .= pos_filter[:,:,1,end]

    # jth jump back in time
    for j in (n_tsave-1):-1:1

        tsave_jump = tsave[j]:1:tsave[j+1]    # "internal", reconstructed time-steps

        # -----------
        # 1) recompute all filter steps between j and j+1

        # holds reconstructed filter results (!!! better move out of loop?)
        pos_filter_jump = similar(pos_filter, nx, ny, 1, length(tsave_jump))         # P(s_t | y_{1:t})
        pos_filter_jump_moved = similar(pos_filter, nx, ny, 1, length(tsave_jump))   # P(s_{t+1} | y_{1:t})

        pos_filter_jump[:,:,1,1] .= pos_filter_jump_moved[:,:,1,1] .= pos_filter[:,:,1,time2index(tsave_jump[1], tsave)]


        for (i,t) in enumerate(tsave_jump[1:(end-1)])

            pos_filter_jump[:,:,1,i+1] .= pos_filter_jump[:,:,1,i]

            # --- solve focker plank
            for k in 1:hops_per_step
                 if !isnothing(ext) # check if we have CUDA.jl loaded
                      pos_filter_jump[:,:,1,(i+1):(i+1)] = ext.conv(pos_filter_jump[:,:,1:1,(i+1):(i+1)], H)
                 else
                    pos_filter_jump[:,:,1,(i+1):(i+1)] = NNlib.conv(pos_filter_jump[:,:,1:1,(i+1):(i+1)], H, pad=1)
                end

            end

            # --- save P(s_{t+1} | y_{1:t})
            pos_filter_jump_moved[:,:,1,i+1] .= pos_filter_jump[:,:,1,i+1]

            # --- add depth signal
            pos_filter_jump[:,:,1,i+1] .*= p_depth.(depth_obs[t+1], bathymetry)

            # --- add accustic signal
            n_sensors = size(dist_acoustic, 3)
            for s in 1:n_sensors
                pos_filter_jump[:,:,1,i+1] .*= p_acoustic.(acoustic_obs[s, t+1], dist_acoustic[:,:,s])
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
            pos_smoother_tmp[:,:,1,1] .= divzero.(pos_smoother_tmp[:,:,1,1], pos_filter_jump[:,:,1,idx])

            # --- solve focker plank backwards
            # K = rot180(H) = H if no advections
            for k in 1:hops_per_step
                 if !isnothing(ext) # check if we have CUDA.jl loaded
                     pos_smoother_tmp[:,:,1,1] = ext.conv(pos_smoother_tmp[:,:,1:1,1:1], H)
                 else
                     pos_smoother_tmp[:,:,1,1] = NNlib.conv(pos_smoother_tmp[:,:,1:1,1:1], H, pad=1)
                 end
            end

            pos_smoother_tmp[:,:,1,1] .=  pos_filter_jump[:,:,1,idx-1] .* pos_smoother_tmp[:,:,1,1]

            # !!! CHECK why is this normalisation needed / allowed? !!!
            pos_smoother_tmp[:,:,1,1] ./= sum(pos_smoother_tmp[:,:,1,1])

            # --- save
            if t in tsave
                pos_smoother[:,:,1,time2index(t, tsave)] = pos_smoother_tmp[:,:,1,1]
            end

        end

    end

    pos_smoother, tsave
end



"""
Tracks the location of the fish

```
track(p_init, bathymetry; depth_obs, acoustic_obs, acoustic_pos, tsave, D, h, smoothing, use_gpu)
```

Uses forward filtering based on a diffusion model and optionally smoothing.

# Arguments

- `p_init::Matrix`: Initial probability distribution of the fish positions
- `bathymetry`: Bathymetric data of the environment
- `depth_obs`: A vector of observed depths at each time step
- `acoustic_obs`: A matrix of  acoustic observations
- `acoustic_pos`: Vector of tuples containing the positions of the acoustic receivers
- `tsave::AbstractVector`: Time steps at which to save the probability map
- `D`: Diffusion coefficient, i.e. variance for one time step movement [m^2]
- `h`: spatial resolution [m]
- `smoothing`: Boolean flag to enable smoothing

"""
function track(p_init, bathymetry;
               depth_obs,
               acoustic_obs, acoustic_pos,
               tsave::AbstractVector = 1:100,
               D, h=1,
               smoothing::Bool=false)

    @assert size(p_init) == size(bathymetry)[1:2]
    @assert size(acoustic_pos, 1) == length(acoustic_pos)
    println("$(size(acoustic_pos, 1)) acoustic sensors")


    nx, ny = size(p_init)


    # convolution kerel
    H, n_hops = make_kernel(D=D, h=h)

    # distances to acoustic sensors
    dist_acoustic = build_distances(acoustic_pos, bathymetry, h)

    # run filter

    ext = Base.get_extension(@__MODULE__, :CUDAExt)
    if !isnothing(ext) # check if we have CUDA.jl loaded
        H, bathymetry, dist_acoustic, pos = ext.move_to_GPU(H, bathymetry, dist_acoustic, tsave)
    else
        pos = similar(p_init, nx, ny, 1, length(tsave))
        bathymetry = bathymetry[:,:,1]
    end

    @info "--- Run filter ---"
    @info " Requires $((maximum(tsave)-1)*n_hops) convolutions"

    run_filter!(pos, H, bathymetry, depth_obs,
                acoustic_obs, dist_acoustic;
                hops_per_step = n_hops, tsave = tsave, p_init)

    if smoothing
        @info "--- Run smoother ---"

        pos_smoother, tsave = run_smoother(pos, H,
                                           bathymetry, depth_obs,
                                           acoustic_obs,
                                           dist_acoustic;
                                           hops_per_step = n_hops,
                                           tsave = tsave)

        if !isnothing(ext) # check if we have CUDA.jl loaded
            return (pos_smoother = Array(pos_smoother), pos_filter = Array(pos), tsave = tsave)
        else
            return (pos_smoother = pos_smoother, pos_filter = pos, tsave = tsave)
        end

    else
        if !isnothing(ext) # check if we have CUDA.jl loaded
            return  (pos_filter = Array(pos), tsave = tsave)
        else
            return (pos_filter = pos, tsave = tsave)
        end
    end

end



end
