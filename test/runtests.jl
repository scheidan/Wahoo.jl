using Wahoo
using Test

import Wahoo.GeoArrays
using DelimitedFiles: readdlm

using Logging
global_logger(ConsoleLogger(stderr, Logging.Warn)) # disable info logging


@testset "Wahoo.jl" verbose = true begin

    @testset "Integration tests CPU" begin


        # Read example data that come with Wahoo
        pathdata = joinpath(pkgdir(Wahoo), "example_data")

        bathymetry_map = GeoArrays.read(joinpath(pathdata, "bathymetry_200m.tif"))

        # -- depth observations
        # define a type and overload the observationsmodel `p_obs`
        struct DepthObservations <: ObservationData
            signals::Vector{Float32}
        end

        function Wahoo.p_obs(o::DepthObservations, t, depth::Number, dist::Number, p)
            Wahoo.p_depth_exponential(o.signals[t], depth, dist)
        end

        depth_signals = readdlm(joinpath(pathdata, "depth_observations.csv"), ',', header=true)[1][:,2]
        depth_obs = DepthObservations(depth_signals)

        # -- acoustic observations
        # define a type and overload the observationsmodel `p_obs`
        struct AcousticObservations <: ObservationData
            signals::Vector{Int}
        end

        function Wahoo.p_obs(o::AcousticObservations, t::Int, depth::Number, distance::Number, p)
            Wahoo.p_acoustic_sigmoid(o.signals[t], depth, distance)
        end

        acoustic_signals = readdlm(joinpath(pathdata, "acoustic_observations.csv"), ',', header=true)[1][:,2:3]
        acoustic_signals = Int.(acoustic_signals')

        acoustic_obs = [AcousticObservations(acoustic_signals[1,:]),
                        AcousticObservations(acoustic_signals[2,:])]

        moorings = readdlm(joinpath(pathdata, "acoustic_moorings.csv"), ',', header=true)[1]
        acoustic_pos = tuple.(moorings[:,2], moorings[:,3])


        # initial values: Matrix{Float64}
        p0 = zeros(size(bathymetry_map)[1], size(bathymetry_map)[2])
        idx = GeoArrays.indices(bathymetry_map, (709757.111649658, 6.26772603565296e6)) # last known location of the fish
        bathymetry_map[idx]
        p0[idx] = 1

        h = 200                     # spatial resolution [m]

        tsave = 105:5:200           # time steps to save
        p = (D = 200^2, )           # tuple with parameters. Diffusion coefficient, i.e. variance of one time step movement [m^2]

        # run tracker
        res = track(p0, bathymetry_map, p; tsave = tsave,
                    h = h,
                    observations = [depth_obs, acoustic_obs...],
                    sensor_positions = [nothing, acoustic_pos...],
                    smoothing = true);

        # check dimensions
        @test res.tsave == tsave
        @test size(res.pos_filter) == size(res.pos_smoother)
        @test size(res.pos_filter, 4) == length(tsave)
        @test size(res.pos_filter)[1:2] == size(bathymetry_map)

        # check normalization
        @test sum(res.residence_dist) ≈ 1
        for j in 1:size(res.pos_filter,4)
            @test sum(res.pos_filter[:,:,:,j]) ≈ 1
            @test sum(res.pos_smoother[:,:,:,j]) ≈ 1
        end
    end


end
