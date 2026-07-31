import StatsBase

"""
If `dist` is an array of probabilities, sample a joint position and activity.
"""
function sample_index(dist::Array)
    sample_idx = StatsBase.sample(1:length(dist), StatsBase.Weights(vec(dist)))
    y, x, a, _ = Tuple(CartesianIndices(dist)[sample_idx])
    return y, x, a
end


"""
Set all values of `A` to zero except the joint state at `(y, x, a)`.
"""
function one_hot!(A::Array, y, x, a)
    A .= zero(eltype(A))
    A[y, x, a, 1] = one(eltype(A))
end


"""
Sample position trajectories from the joint distribution
`Prob(s_{1...T}, a_{1...T} | y_{1...T})`.
"""
function sample_trajectories(pos_filter, H, transition_matrix,
                             bathymetry, observations, observation_models, distances;
                             tsave=1:100, n_trajectories, hops_per_step,
                             show_progressbar)

    tmax = maximum(tsave)
    n_tsave = length(tsave)
    nx, ny, n_activity = size(pos_filter)[1:3]

    trajectories = [zeros(Int, 2, tmax) for _ in 1:n_trajectories]
    state_tmp = similar(pos_filter, nx, ny, n_activity, 1)
    previous_distribution = similar(state_tmp)
    backward_spatial = similar(first(H), nx, ny, 1, 1)

    pmeter = ProgressMeter.Progress(n_trajectories*(n_tsave-1);
                                    desc = "Sample $n_trajectories trajectories...:",
                                    output = stderr, enabled = show_progressbar)

    for n_trj in 1:n_trajectories
        state_tmp[:,:,:,1] .= pos_filter[:,:,:,end]
        y, x, a_next = sample_index(state_tmp)
        trajectories[n_trj][:,tmax] .= (y, x)
        one_hot!(state_tmp, y, x, a_next)

        # jth jump back in time
        for j in (n_tsave-1):-1:1
            tsave_jump = tsave[j]:1:tsave[j+1]

            # recompute all filter steps between j and j+1
            pos_filter_jump = similar(pos_filter, nx, ny, n_activity, length(tsave_jump))
            pos_filter_jump[:,:,:,1] .=
                pos_filter[:,:,:,time2index(tsave_jump[1], tsave)]

            prediction = similar(state_tmp)
            for (i, t) in enumerate(tsave_jump[1:(end-1)])
                state = @view pos_filter_jump[:,:,:,i:i]
                predict_state!(prediction, state, H, transition_matrix, bathymetry;
                               hops_per_step = hops_per_step)

                next_state = @view pos_filter_jump[:,:,:,i+1:i+1]
                next_state .= prediction
                filter_observations!(next_state, t + 1, bathymetry,
                                     observations, observation_models, distances)
            end

            # sample each preceding joint state conditional on the sampled future state
            for (i, t) in enumerate(reverse(tsave_jump)[2:end])
                idx = length(tsave_jump) - i
                previous_distribution .= zero(eltype(previous_distribution))

                for a_prev in 1:n_activity
                    backward_spatial[:,:,1,1] .= state_tmp[:,:,a_next,1]
                    for k in 1:hops_per_step[a_prev]
                        backward_spatial = NNlib.conv(backward_spatial, H[a_prev], pad=1)
                    end
                    previous_distribution[:,:,a_prev,1] .=
                        pos_filter_jump[:,:,a_prev,idx] .* backward_spatial[:,:,1,1] .*
                        transition_matrix[a_prev,a_next]
                end

                previous_distribution .= ifelse.(bathymetry .< 0, 0, previous_distribution)
                previous_distribution ./= sum(previous_distribution)
                y, x, a_next = sample_index(previous_distribution)
                trajectories[n_trj][:,t] .= (y, x)
                one_hot!(state_tmp, y, x, a_next)
            end

            ProgressMeter.next!(pmeter)
        end
    end
    return trajectories
end
