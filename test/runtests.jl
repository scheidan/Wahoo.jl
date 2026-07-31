using Wahoo
using Test

import Wahoo.GeoArrays
using DelimitedFiles: readdlm

import Random
Random.seed!(14022025)

using Logging
global_logger(ConsoleLogger(stderr, Logging.Warn)) # disable info logging


@testset "Wahoo.jl" verbose = true begin
    @testset "divzero" begin

        eps1 = eps(0f0)         # smallest nonzero number

        @test Wahoo.divzero(4f0, 3f0) == 4f0 / 3f0
        @test Wahoo.divzero(4f0, 0f0) == zero(Float32)
        @test Wahoo.divzero(0f0, 3f0) == zero(Float32)
        @test Wahoo.divzero(0f0, 0f0) == zero(Float32)

        @test Wahoo.divzero(eps1, 3f0) == zero(Float32)
        @test Wahoo.divzero(eps1, 0f0) == zero(Float32)

        @test isfinite(Wahoo.divzero(4f0, eps1)) # must not be inf32
        @test Wahoo.divzero(0f0, eps1) == zero(Float32)
        @test Wahoo.divzero(eps1, eps1) == one(Float32)

    end


    @testset "Activity model" begin
        p0 = zeros(Float32, 9, 9)
        p0[5,5] = 1
        bathymetry = ones(Float32, 9, 9)
        distances = Array{Float32}(undef, 9, 9, 0)
        kernels_and_hops = [Wahoo.make_kernel(D = sd^2/2, h = 1,
                                              precision = Float32) for sd in (0.1, 0.2)]
        H = first.(kernels_and_hops)
        n_hops = last.(kernels_and_hops)
        transition_matrix = Float32[0.8 0.2; 0.3 0.7]
        activity_init = Float32[0.25, 0.75]

        state_filter, _ = Wahoo.run_filter(p0, activity_init, H, transition_matrix,
                                           bathymetry, Any[], Function[], distances;
                                           hops_per_step = n_hops, tsave = 1:3,
                                           show_progressbar = false)
        activity_filter = dropdims(sum(state_filter, dims=(1, 2)), dims=(1, 2))
        expected = hcat(activity_init, transition_matrix' * activity_init,
                        transition_matrix' * transition_matrix' * activity_init)

        @test size(state_filter) == (9, 9, 2, 3)
        @test activity_filter ≈ expected
        @test all(sum(state_filter, dims=(1, 2, 3)) .≈ 1)

        # Activity at t controls movement to t + 1.
        switch_matrix = Float32[0 1; 1 0]
        switch_filter, _ = Wahoo.run_filter(p0, Float32[1, 0], H, switch_matrix,
                                            bathymetry, Any[], Function[], distances;
                                            hops_per_step = n_hops, tsave = 1:2,
                                            show_progressbar = false)
        expected_slow_move = Wahoo.NNlib.conv(reshape(p0, 9, 9, 1, 1), H[1], pad=1)
        @test switch_filter[:,:,1,2] == zeros(Float32, 9, 9)
        @test switch_filter[:,:,2,2] ≈ expected_slow_move[:,:,1,1]

        # Sparse smoothing must apply the observation at the destination time.
        function p_obs_test(signals, t, depth, dist)
            exp(-abs(signals[t] - dist))
        end
        signals = Float32[2, 3, 5, 6, 4]
        obs_distances = reshape(repeat(Float32.(1:9), inner=9), 9, 9, 1)
        single_H = H[1:1]
        single_hops = n_hops[1:1]
        single_transition = ones(Float32, 1, 1)
        dense_filter, _ = Wahoo.run_filter(p0, Float32[1], single_H, single_transition,
                                           bathymetry, [signals], [p_obs_test], obs_distances;
                                           hops_per_step = single_hops, tsave = 1:5,
                                           show_progressbar = false)
        sparse_filter, _ = Wahoo.run_filter(p0, Float32[1], single_H, single_transition,
                                            bathymetry, [signals], [p_obs_test], obs_distances;
                                            hops_per_step = single_hops, tsave = 1:2:5,
                                            show_progressbar = false)
        dense_smoother, _ = Wahoo.run_smoother(dense_filter, single_H, single_transition,
                                               bathymetry, [signals], [p_obs_test], obs_distances;
                                               hops_per_step = single_hops, tsave = 1:5,
                                               show_progressbar = false)
        sparse_smoother, _ = Wahoo.run_smoother(sparse_filter, single_H, single_transition,
                                                bathymetry, [signals], [p_obs_test], obs_distances;
                                                hops_per_step = single_hops, tsave = 1:2:5,
                                                show_progressbar = false)
        @test sparse_smoother ≈ dense_smoother[:,:,:,[1, 3, 5]]

        # Sampled positions must be written at their own time point.
        trajectory_filter = zeros(Float32, 9, 9, 1, 3)
        trajectory_filter[5,5,1,1] = 1
        trajectory_filter[5,6,1,2] = 1
        trajectory_filter[5,7,1,3] = 1
        trajectory_H = [reshape(Float32[0 0.1 0; 0.1 0.6 0.1; 0 0.1 0], 3, 3, 1, 1)]
        trajectory = Wahoo.sample_trajectories(trajectory_filter, trajectory_H,
                                               single_transition, bathymetry,
                                               Any[], Function[], distances;
                                               tsave = 1:3, n_trajectories = 1,
                                               hops_per_step = [1],
                                               show_progressbar = false)[1]
        @test trajectory == [5 5 5; 5 6 7]
    end


    @testset "Integration tests CPU" begin


        # Read example data that come with Wahoo
        pathdata = joinpath(pkgdir(Wahoo), "example_data")

        bathymetry_map = GeoArrays.read(joinpath(pathdata, "bathymetry_200m.tif"))

        # -- depth observations

        # likelihood function
        function p_obs_depth(signals, t, depth::Number, dist::Number)
            Wahoo.p_depth_exponential(signals[t], depth, dist)
        end

        depth_signals = readdlm(joinpath(pathdata, "depth_observations.csv"), ',', header=true)[1][:,2]

        # -- acoustic observations

        # likelihood function
        function p_obs_acoustic(signals, t::Int, depth::Number, distance::Number)
            Wahoo.p_acoustic_sigmoid(signals[t], depth, distance)
        end

        acoustic_signals = readdlm(joinpath(pathdata, "acoustic_observations.csv"), ',', header=true)[1][:,2:3]
        acoustic_signals = Int.(acoustic_signals')
        acoustic_obs = [ acoustic_signals[1,:], acoustic_signals[2,:]]
        acoustic_obs_models = [p_obs_acoustic, p_obs_acoustic]

        moorings = readdlm(joinpath(pathdata, "acoustic_moorings.csv"), ',', header=true)[1]
        acoustic_pos = tuple.(moorings[:,2], moorings[:,3])


        # initial values: Matrix{Float64}
        p0 = zeros(size(bathymetry_map)[1], size(bathymetry_map)[2])
        idx = GeoArrays.indices(bathymetry_map, (709757.111649658, 6.26772603565296e6)) # last known location of the fish
        bathymetry_map[idx]
        p0[idx] = 1

        h = 200                             # spatial resolution [m]
        movement_std = 200          # std of one time step movement [m]

        tsave = 1:5:720             # time steps to save

        # run tracker
        res = track(pos_init = p0, bathymetry = bathymetry_map,
                    tsave = tsave,
                    spatial_resolution = h,  movement_std = movement_std,
                    observations = [depth_signals, acoustic_obs...],
                    observation_models = [p_obs_depth, acoustic_obs_models...],
                    sensor_positions = [nothing, acoustic_pos...],
                    filter = true,
                    n_trajectories = 2,
                    precision = Float32)

        # check dimensions
        @test res.tsave == tsave
        @test size(res.pos_filter) == size(res.pos_smoother)
        @test size(res.pos_filter, 4) == length(tsave)
        @test size(res.pos_filter)[1:2] == size(bathymetry_map)
        @test length(res.log_p) == length(tsave)

        @test all(isfinite.(res.pos_filter))
        @test all(isfinite.(res.pos_smoother))

        # check normalization
        @test all(isfinite.(res.residence_dist))
        @test sum(res.residence_dist) ≈ 1
        for j in 1:size(res.pos_filter,4)
            @test sum(res.pos_filter[:,:,:,j]) ≈ 1
            @test sum(res.pos_smoother[:,:,:,j]) ≈ 1
        end

        @test length(res.trajectories) == 2
        @test size((res.trajectories[1])) == (2, length(minimum(tsave):maximum(tsave)))
        for k in 1:length(res.trajectories)
            @test all(1 .<= res.trajectories[k][1,:] .<= size(bathymetry_map, 1)) # y-coordinates
            @test all(1 .<= res.trajectories[k][2,:] .<= size(bathymetry_map, 2)) # x-coordinates
        end

        # a crude check to see if results have changed
        @test sum(abs2, res.pos_filter) ≈  7.9684896f0
        @test sum(abs2, res.pos_smoother) ≈ 12.602797f0

        transition_matrix = Float32[0.8 0.2; 0.3 0.7]
        activity_init = Float32[0.25, 0.75]
        hmm_res = track(pos_init = p0, bathymetry = bathymetry_map,
                        tsave = 1:3, spatial_resolution = h,
                        movement_std = (100, 200),
                        transition_matrix = transition_matrix,
                        activity_init = activity_init,
                        observations = Any[], observation_models = Function[],
                        sensor_positions = Any[], filter = true,
                        n_trajectories = 1, precision = Float32)
        @test size(hmm_res.pos_filter) == (size(bathymetry_map)[1:2]..., 1, 3)
        @test size(hmm_res.pos_smoother) == size(hmm_res.pos_filter)
        @test size(hmm_res.activity_filter) == (2, 3)
        @test hmm_res.activity_filter[:,1] == activity_init
        @test all(sum(hmm_res.activity_filter, dims=1) .≈ 1)
        @test size(hmm_res.trajectories[1]) == (2, 3)

        hmm_res_default = track(pos_init = p0, bathymetry = bathymetry_map,
                                tsave = 1:1, spatial_resolution = h,
                                movement_std = (100, 200),
                                transition_matrix = transition_matrix,
                                observations = Any[], observation_models = Function[],
                                sensor_positions = Any[], smoother = false,
                                precision = Float32)
        @test !hasproperty(hmm_res_default, :pos_filter)
        @test hmm_res_default.activity_filter[:,1] == Float32[0.5, 0.5]

        # Test for mismatched inputs
        @test_throws ErrorException track(pos_init = p0, bathymetry = bathymetry_map,
                                          tsave = tsave,
                                          spatial_resolution = h, movement_std = movement_std,
                                          observations = [depth_signals],
                                          observation_models = [p_obs_depth, p_obs_acoustic],
                                          sensor_positions = [nothing])

        @test_throws ErrorException track(pos_init = p0[1:5, 1:20], bathymetry = bathymetry_map,
                                          tsave = tsave,
                                          spatial_resolution = h, movement_std = movement_std,
                                          observations = [depth_signals],
                                          observation_models = [p_obs_depth, p_obs_acoustic],
                                          sensor_positions = [nothing])

    end


end
