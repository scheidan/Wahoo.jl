
import StatsBase

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

    trajectory = zeros(Int, 2, tmax)

    pos_distribution_tmp = similar(pos_filter, nx, ny, 1)
    pos_distribution_tmp[:,:,1] .= pos_filter[:,:,1,end]

    # distribution of resdence time over all time steps
    residence_dist = similar(pos_filter, nx, ny)
    residence_dist[:,:] .= pos_filter[:,:,1,end]

    pmeter = ProgressMeter.Progress(n_tsave - 1; desc = "Sample trajectories...:",
                                    output = stderr, enabled = show_progressbar)

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

        # !!!TODO!!!: sample multible trajectories at once!

        #pos_distribution_tmp[:,:,1,1] .= pos_smoother[:,:,1,time2index(tsave_jump[end], tsave)]

        for (i, t) in enumerate(reverse(tsave_jump)[2:end])

            idx = length(tsave_jump) - i + 1 # index of pos_filter_jump

            # --- sample position (i,j) ~ pos_distribution_tmp[j,j,1,1]
            # Sample a linear index with weights from the flattened matrix
            sample_idx = StatsBase.sample(1:length(pos_distribution_tmp), StatsBase.Weights(vec(pos_distribution_tmp)))
            y, x, _, = Tuple(CartesianIndices(pos_distribution_tmp)[sample_idx])

            # inital ditribution
            pos_distribution_tmp[:,:,1,1] .= 0
            pos_distribution_tmp[y,x,1,1] = 1

            # treat division by zero as special case
            pos_distribution_tmp[:,:,1,1] .= divzero.(pos_distribution_tmp[:,:,1,1], pos_filter_jump_no_obs[:,:,1,idx])


            # --- solve Fokker-Plank backwards
            # K = rot180(H) = H if no advections
            for k in 1:hops_per_step
                pos_distribution_tmp[:,:,1,1] = NNlib.conv(pos_distribution_tmp[:,:,1:1,1:1], H, pad=1)
            end

            pos_distribution_tmp[:,:,1,1] .=  pos_filter_jump[:,:,1,idx-1] .* pos_distribution_tmp[:,:,1,1] #.+ eps(0f0)
            pos_distribution_tmp[:,:,1,1] ./= sum(pos_distribution_tmp[:,:,1,1])

            # --- save
            trajectory[1, t] = y
            trajectory[2, t] = x
        end

        ProgressMeter.next!(pmeter)

    end

    return  trajectory
end
