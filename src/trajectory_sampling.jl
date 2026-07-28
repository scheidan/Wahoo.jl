
import StatsBase

"""
If `dist` is a array of probabilities, sample an element and return the first two indices.
"""
function sample_index(dist::Array)
     # Sample a linear index with weights from the flattened matrix
    sample_idx = StatsBase.sample(1:length(dist), StatsBase.Weights(vec(dist)))
    y, x, _ = Tuple(CartesianIndices(dist)[sample_idx])
    return y, x
end


"""
Set all values of A to zero except A[y,x] to one.
"""
function one_hot!(A::Array, y, x)
    A .= zero(eltype(A))
    A[y, x] = one(eltype(A))
end


trajectory_host_array(A::Array) = A
trajectory_host_array(A) = Array(A)

function make_transition_state(T, hops)
    n = 2*hops + 1
    (current=zeros(T, n, n), next=zeros(T, n, n),
     ys=zeros(Int, n^2), xs=zeros(Int, n^2), weights=zeros(T, n^2))
end

function sample_transition(state, n_weights)
    idx = StatsBase.sample(1:n_weights, StatsBase.Weights(view(state.weights, 1:n_weights)))
    return state.ys[idx], state.xs[idx]
end

function backward_weights!(state, y, x, pos_filter, bathymetry, H, hops)
    current, next = state.current, state.next
    fill!(current, zero(eltype(current)))
    center = hops + 1
    current[center,center] = one(eltype(current))
    ny, nx = size(pos_filter)

    for _ in 1:hops
        fill!(next, zero(eltype(next)))
        for j in axes(current, 2), i in axes(current, 1)
            gy, gx = y + i - center, x + j - center
            if 1 <= gy <= ny && 1 <= gx <= nx
                v = H[2,2,1,1] * current[i,j]
                gy > 1 && i > 1 && (v += H[1,2,1,1] * current[i-1,j])
                gy < ny && i < size(current, 1) && (v += H[3,2,1,1] * current[i+1,j])
                gx > 1 && j > 1 && (v += H[2,1,1,1] * current[i,j-1])
                gx < nx && j < size(current, 2) && (v += H[2,3,1,1] * current[i,j+1])
                next[i,j] = v
            end
        end
        current, next = next, current
    end

    n_weights = 0
    for j in axes(current, 2), i in axes(current, 1)
        gy, gx = y + i - center, x + j - center
        if 1 <= gy <= ny && 1 <= gx <= nx && bathymetry[gy,gx] >= 0
            weight = current[i,j] * pos_filter[gy,gx]
            if weight > 0
                n_weights += 1
                state.ys[n_weights] = gy
                state.xs[n_weights] = gx
                state.weights[n_weights] = weight
            end
        end
    end
    n_weights
end


function initialize_trajectory_sampler(pos_filter, H, bathymetry;
                                       tsave, n_trajectories, hops_per_step,
                                       show_progressbar)
    tmax = maximum(tsave)
    nx, ny = size(pos_filter)[1:2]
    trajectories = [zeros(Int, 2, tmax) for _ in 1:n_trajectories]
    transition_states = [make_transition_state(eltype(H), hops_per_step) for _ in 1:n_trajectories]
    n_transition_weights = zeros(Int, n_trajectories)
    initial_transition = trues(n_trajectories)
    pos_distribution_tmp = similar(pos_filter, nx, ny, 1)
    pos_distribution_tmp[:,:,1] .= pos_filter[:,:,1,end]
    for n_trj in eachindex(trajectories)
        trajectories[n_trj][:, end] .= sample_index(pos_distribution_tmp)
    end
    pmeter = ProgressMeter.Progress(n_trajectories*(length(tsave)-1);
                                    desc = "Sample $n_trajectories trajectories...:",
                                    output = stderr, enabled = show_progressbar)
    (; trajectories, transition_states, n_transition_weights, initial_transition,
     pos_distribution_tmp, H_host=trajectory_host_array(H),
     bathymetry_host=trajectory_host_array(bathymetry), pmeter)
end

@views function sample_trajectory_block!(sampler, pos_filter_jump, tsave_jump,
                                         hops_per_step)
    pos_filter_jump_host = trajectory_host_array(pos_filter_jump)
    for n_trj in eachindex(sampler.trajectories)
        transition_state = sampler.transition_states[n_trj]
        for (i, t) in enumerate(reverse(tsave_jump)[2:end])
            idx = length(tsave_jump) - i + 1
            y, x = if sampler.initial_transition[n_trj]
                sampler.initial_transition[n_trj] = false
                sample_index(sampler.pos_distribution_tmp)
            else
                sample_transition(transition_state, sampler.n_transition_weights[n_trj])
            end
            sampler.n_transition_weights[n_trj] = backward_weights!(
                transition_state, y, x, pos_filter_jump_host[:,:,1,idx-1],
                sampler.bathymetry_host, sampler.H_host, hops_per_step)
            sampler.trajectories[n_trj][:, t] .= (y, x)
        end
        ProgressMeter.next!(sampler.pmeter)
    end
end


"""
Sample random trajectories from the joint distribution Prob(s_{1...T} | y_{1...T}).
"""
@views function sample_trajectories(pos_filter, H,
                                    bathymetry,
                                    observations, observation_models,
                                    distances;
                                    tsave = tsave,
                                    n_trajectories,
                                    hops_per_step,
                                    show_progressbar)

    n_tsave = length(tsave)
    nx, ny = size(pos_filter)[1:2]

    sampler = initialize_trajectory_sampler(pos_filter, H, bathymetry;
                                            tsave, n_trajectories, hops_per_step,
                                            show_progressbar)
    pos_filter_jump_tmp = similar(pos_filter, nx, ny, 1, 1)
    pos_filter_jump_scratch = similar(pos_filter_jump_tmp)

    # jth jump back in time
    for j in (n_tsave-1):-1:1

        tsave_jump = tsave[j]:1:tsave[j+1]    # "internal", reconstructed time-steps

        # -----------
        # 1) recompute all filter steps between j and j+1

        # holds reconstructed filter results
        pos_filter_jump = similar(pos_filter, nx, ny, 1, length(tsave_jump))          # P(s_t | y_{1:t})

        pos_filter_jump[:,:,1,1] .= pos_filter[:,:,1,j]

        for (i,t) in enumerate(tsave_jump[1:(end-1)])

            # --- solve Fokker-Plank
            pos_filter_jump_tmp[:,:,1,1] .= pos_filter_jump[:,:,1,i]
            pos_filter_jump_tmp, pos_filter_jump_scratch = diffuse!(
                pos_filter_jump_tmp, pos_filter_jump_scratch, H, hops_per_step)
            pos_filter_jump[:,:,1,i+1] .= pos_filter_jump_tmp[:,:,1,1]

            # you can't be on land (negative bathymetry)
            pos_filter_jump[:,:,1,i+1] .= ifelse.(bathymetry .< 0, 0, pos_filter_jump[:,:,1,i+1])

            # --- add observations
            for k  in eachindex(observations)
                p_obs = observation_models[k]
                signals = observations[k]
                pos_filter_jump[:,:,1,i+1] .*= p_obs.(Ref(signals), Ref(t), bathymetry, view(distances, :,:,k))
            end

            # --- normalize
            Z = sum(pos_filter_jump[:,:,1,i+1])
            isfinite(Z) ||
                error("No solution at time point $(t)!")
            pos_filter_jump[:,:,1,i+1] .= pos_filter_jump[:,:,1,i+1] ./ Z

        end

        # -----------
        # 2) sample trajectories

        sample_trajectory_block!(sampler, pos_filter_jump, tsave_jump, hops_per_step)
    end
    return sampler.trajectories
end
