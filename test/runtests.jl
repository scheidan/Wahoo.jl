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

        h = 200                     # spatial resolution [m]
        movement_std = 200          # std of one time step movement [m]

        tsave = 1:5:720             # time steps to save

        # run tracker
        res = track(pos_init = p0, bathymetry = bathymetry_map,
                    tsave = tsave,
                    spatial_resolution = h,  movement_std = movement_std,
                    observations = [depth_signals, acoustic_obs...],
                    observation_models = [p_obs_depth, acoustic_obs_models...],
                    sensor_positions = [nothing, acoustic_pos...],
                    save_filter = true,
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
        @test sum(abs2, res.pos_filter) ≈ 5.656419f0
        @test sum(abs2, res.pos_smoother) ≈ 7.3226733f0

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
