
import StatsBase

"""
If `dist` is a array of probabilities, sample an elemnet and returns the first two indices
"""
function sample_index(dist::Array)
     # Sample a linear index with weights from the flattened matrix
    sample_idx = StatsBase.sample(1:length(dist), StatsBase.Weights(vec(dist)))
    y, x, _ = Tuple(CartesianIndices(dist)[sample_idx])
    return y, x
end


function sample_trajectories(pos_filter, H,
                             bathymetry,
                             observations, observation_models,
                             distances;
                             tsave = tsave,
                             n_trajectories,
                             hops_per_step,
                             show_progressbar)

    ext = Base.get_extension(@__MODULE__, :CUDAExt)

    tmax = maximum(tsave)
    n_tsave = length(tsave)
    nx, ny = size(pos_filter)[1:2]

    trajectories = [zeros(Int, 2, tmax) for _ in 1:n_trajectories]

    pos_distribution_tmp = similar(pos_filter, nx, ny, 1)

    pmeter = ProgressMeter.Progress(n_trajectories; desc = "Sample $n_trajectories trajectories...:",
                                    output = stderr, enabled = show_progressbar)

    for n in 1:n_trajectories
        pos_distribution_tmp[:,:,1] .= pos_filter[:,:,1,end]

        # Sample initial point
        trajectories[n][:, end] .= sample_index(pos_distribution_tmp)


        # jth jump back in time
        for j in (n_tsave-1):-1:1

            tsave_jump = tsave[j]:1:tsave[j+1]    # "internal", reconstructed time-steps

            # -----------
            # 1) recompute all filter steps between j and j+1

            # holds reconstructed filter results
            pos_filter_jump = similar(pos_filter, nx, ny, 1, length(tsave_jump))          # P(s_t | y_{1:t})
            pos_filter_jump_no_obs = similar(pos_filter, nx, ny, 1, length(tsave_jump))   # P(s_{t+1} | y_{1:t})

            pos_filter_jump[:,:,1,1] .= pos_filter_jump_no_obs[:,:,1,1] .= pos_filter[:,:,1,time2index(tsave_jump[1], tsave)]


            for (i,t) in enumerate(tsave_jump[1:(end-1)])

                pos_filter_jump[:,:,1,i+1] .= pos_filter_jump[:,:,1,i]

                # --- solve Fokker-Plank
                for k in 1:hops_per_step
                    pos_filter_jump[:,:,1,(i+1):(i+1)] = NNlib.conv(pos_filter_jump[:,:,1:1,(i+1):(i+1)], H, pad=1)
                end

                # you can't be on land (negative bathymetry)
                pos_filter_jump .= ifelse.(bathymetry .< 0, 0, pos_filter_jump)

                # --- save P(s_{t+1} | y_{1:t})
                pos_filter_jump_no_obs[:,:,1,i+1] .= pos_filter_jump[:,:,1,i+1]

                # --- add observations
                for k  in eachindex(observations)
                    p_obs = observation_models[k]
                    signals = observations[k]
                    pos_filter_jump[:,:,1,1] .*= p_obs.(Ref(signals), Ref(t), bathymetry, view(distances, :,:,k))
                end

                # --- normalize
                Z = sum(pos_filter_jump[:,:,1,i+1])
                isfinite(Z) ||
                    error("No solution at time point $(t)!")
                pos_filter_jump[:,:,1,i+1] .= pos_filter_jump[:,:,1,i+1] ./ Z


            end

            # -----------
            # 2) backward smoothing

            for (i, t) in enumerate(reverse(tsave_jump)[2:end])

                idx = length(tsave_jump) - i + 1 # index of pos_filter_jump

                # --- sample position (i,j) ~ pos_distribution_tmp[j,j,1,1]
                # Sample a linear index with weights from the flattened matrix
                y, x = sample_index(pos_distribution_tmp)

                # inital distribution
                pos_distribution_tmp[:,:,1,1] .= 0
                pos_distribution_tmp[y,x,1,1] = 1

                # treat division by zero as special case
                pos_distribution_tmp[:,:,1,1] .= divzero.(pos_distribution_tmp[:,:,1,1], pos_filter_jump_no_obs[:,:,1,idx])


                # --- solve Fokker-Plank backwards
                # K = rot180(H) = H if no advections
                for k in 1:hops_per_step
                    pos_distribution_tmp[:,:,1,1] = NNlib.conv(pos_distribution_tmp[:,:,1:1,1:1], H, pad=1)
                end

                # you can't be on land (negative bathymetry)
                pos_distribution_tmp .= ifelse.(bathymetry .< 0, 0, pos_distribution_tmp)

                pos_distribution_tmp[:,:,1,1] .=  pos_filter_jump[:,:,1,idx-1] .* pos_distribution_tmp[:,:,1,1] #.+ eps(0f0)
                pos_distribution_tmp[:,:,1,1] ./= sum(pos_distribution_tmp[:,:,1,1])

                # --- save
                trajectories[n][:, t] .= (y, x)

            end

        end
        ProgressMeter.next!(pmeter)

    end
    return  trajectories
end
