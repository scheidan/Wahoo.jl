import NNlib
import GeoArrays
import ProgressMeter

"""
Compute convolution kernel and the required number of hops per time steps

D: diffusion coefficient
h: spatial resolution
"""
function make_kernel(;D, h, precision)

    # compute 1/Δ, so that 4*D*Δ/h^2 < 1
    n_hops = ceil(Int, 4*D/(h^2*0.99))
    Δ = 1/n_hops
    @assert 4*D*Δ/h^2 < 1

    # convolution kernel
    H = [0 0 0;
         0 1 0;
         0 0 0]

    H = H + D*Δ/h^2 .* [0  1  0;
                        1 -4  1;
                        0  1  0]
    H = precision.(reshape(H, 3, 3, 1, 1))

    H, n_hops
end


"""
Maps time step to the index of `pos_filter` and `pos_smoother`
"""
time2index(t, tsave) = findfirst(==(t), tsave)


"""
Predict the joint position and activity state by one time step.
"""
function predict_state!(prediction, state, H, transition_matrix, bathymetry;
                        hops_per_step)

    prediction .= zero(eltype(prediction))
    nx, ny, n_activity = size(state)[1:3]
    moved = similar(first(H), nx, ny, 1, 1)

    for a_prev in 1:n_activity
        moved[:,:,1,1] .= state[:,:,a_prev,1]
        for k in 1:hops_per_step[a_prev]
            moved = NNlib.conv(moved, H[a_prev], pad=1)
        end
        for a_next in 1:n_activity
            prediction[:,:,a_next,1] .+= transition_matrix[a_prev,a_next] .* moved[:,:,1,1]
        end
    end

    # you can't be on land (negative bathymetry)
    prediction .= ifelse.(bathymetry .< 0, 0, prediction)
    prediction
end


"""
Apply the observations at `t` and normalize the joint state.
"""
function filter_observations!(state, t, bathymetry,
                              observations, observation_models, distances)

    for k in eachindex(observations)
        p_obs = observation_models[k]
        signals = observations[k]
        state .*= p_obs.(Ref(signals), Ref(t), bathymetry, view(distances, :,:,k))
    end

    Z = sum(state)
    isfinite(Z) ||
        error("No solution at time point $(t)! Check for data incompatibilities.")
    state ./= Z
    Z
end


# ---
# filter algorithm

function run_filter(pos_init, activity_init, H, transition_matrix,
                    bathymetry, observations, observation_models, distances;
                    hops_per_step, tsave=1:100, show_progressbar)

    tmax = maximum(tsave)

    pos_init ./= sum(pos_init)
    nx, ny = size(bathymetry)[1:2]
    n_activity = length(activity_init)

    pos = similar(first(H), nx, ny, n_activity, length(tsave))
    pos_tmp = similar(first(H), nx, ny, n_activity, 1)
    pos_prediction = similar(pos_tmp)

    for a in 1:n_activity
        pos_tmp[:,:,a,1] .= activity_init[a] .* pos_init
    end
    pos[:,:,:,1] .= pos_tmp[:,:,:,1]

    pmeter = ProgressMeter.Progress(tmax - 1; desc = "Filtering...:",
                                    output = stderr, enabled = show_progressbar)

    log_p = zeros(eltype(first(H)), length(tsave))
    log_p_acc = zero(eltype(first(H)))

    for t in 2:tmax
        predict_state!(pos_prediction, pos_tmp, H, transition_matrix, bathymetry;
                       hops_per_step = hops_per_step)
        pos_tmp .= pos_prediction

        Z = filter_observations!(pos_tmp, t, bathymetry,
                                 observations, observation_models, distances)
        log_p_acc += log(Z)

        if t in tsave
            pos[:,:,:,time2index(t, tsave)] .= pos_tmp[:,:,:,1]
            log_p[time2index(t, tsave)] = log_p_acc
            log_p_acc = zero(eltype(first(H)))
        end

        ProgressMeter.next!(pmeter)
    end

    return pos, log_p
end


# ---
# smoothing algorithm

"""
Define division by zero as equal to zero.
"""
function divzero(a, b)
    if iszero(b)
        zero(a/b)
    else
        # if b < abs(a) / typemax(a) we can get Inf
        min(a/b, floatmax(b))
    end
end


function run_smoother(pos_filter, H, transition_matrix,
                      bathymetry, observations, observation_models, distances;
                      hops_per_step, tsave=1:100, show_progressbar)

    n_tsave = length(tsave)
    nx, ny, n_activity = size(pos_filter)[1:3]

    # hold smoothed results, P(s_t, a_t | y_{1:T})
    pos_smoother = similar(pos_filter)
    pos_smoother_tmp = similar(pos_filter, nx, ny, n_activity, 1)
    pos_smoother_tmp[:,:,:,1] .= pos_smoother[:,:,:,end] .= pos_filter[:,:,:,end]

    # distribution of residence time over all time steps
    residence_dist = similar(pos_filter, nx, ny)
    residence_dist[:,:] .= dropdims(sum(pos_filter[:,:,:,end], dims=3), dims=3)

    ratio = similar(pos_smoother_tmp)
    message = similar(pos_smoother_tmp)
    weighted_ratio = similar(first(H), nx, ny, 1, 1)

    pmeter = ProgressMeter.Progress(n_tsave - 1; desc = "Smoothing...:",
                                    output = stderr, enabled = show_progressbar)

    # jth jump back in time
    for j in (n_tsave-1):-1:1
        tsave_jump = tsave[j]:1:tsave[j+1]

        # recompute all filter steps between j and j+1
        pos_filter_jump = similar(pos_filter, nx, ny, n_activity, length(tsave_jump))
        pos_filter_jump_no_obs = similar(pos_filter, nx, ny, n_activity, length(tsave_jump))
        pos_filter_jump[:,:,:,1] .= pos_filter_jump_no_obs[:,:,:,1] .=
            pos_filter[:,:,:,time2index(tsave_jump[1], tsave)]

        for (i, t) in enumerate(tsave_jump[1:(end-1)])
            state = @view pos_filter_jump[:,:,:,i:i]
            prediction = @view pos_filter_jump_no_obs[:,:,:,i+1:i+1]
            predict_state!(prediction, state, H, transition_matrix, bathymetry;
                           hops_per_step = hops_per_step)

            next_state = @view pos_filter_jump[:,:,:,i+1:i+1]
            next_state .= prediction
            filter_observations!(next_state, t + 1, bathymetry,
                                 observations, observation_models, distances)
        end

        # backward smoothing
        pos_smoother_tmp[:,:,:,1] .= pos_smoother[:,:,:,time2index(tsave_jump[end], tsave)]

        for (i, t) in enumerate(reverse(tsave_jump)[2:end])
            idx = length(tsave_jump) - i + 1
            ratio .= divzero.(pos_smoother_tmp,
                              @view(pos_filter_jump_no_obs[:,:,:,idx:idx]))
            message .= zero(eltype(message))

            for a_prev in 1:n_activity
                weighted_ratio .= zero(eltype(weighted_ratio))
                for a_next in 1:n_activity
                    weighted_ratio[:,:,1,1] .+= transition_matrix[a_prev,a_next] .*
                                                   ratio[:,:,a_next,1]
                end
                for k in 1:hops_per_step[a_prev]
                    weighted_ratio = NNlib.conv(weighted_ratio, H[a_prev], pad=1)
                end
                message[:,:,a_prev,1] .= weighted_ratio[:,:,1,1]
            end

            message .= ifelse.(bathymetry .< 0, 0, message)
            pos_smoother_tmp .= @view(pos_filter_jump[:,:,:,idx-1:idx-1]) .* message
            pos_smoother_tmp ./= sum(pos_smoother_tmp)

            residence_dist .+= dropdims(sum(pos_smoother_tmp[:,:,:,1], dims=3), dims=3)

            if t in tsave
                pos_smoother[:,:,:,time2index(t, tsave)] .= pos_smoother_tmp[:,:,:,1]
            end
        end

        ProgressMeter.next!(pmeter)
    end

    residence_dist ./= sum(residence_dist)
    return pos_smoother, residence_dist
end
