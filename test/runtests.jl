using Wahoo
using Test

import Wahoo.GeoArrays
using DelimitedFiles: readdlm

using Logging
global_logger(ConsoleLogger(stderr, Logging.Warn)) # disable info logging


@testset "Wahoo.jl" begin

    @testset "Integration tests CPU" begin


        # Read example data that come with Wahoo
        pathdata = joinpath(pkgdir(Wahoo), "example_data")

        bathymetry_map = GeoArrays.read(joinpath(pathdata, "bathymetry_200m.tif"))
        depth_obs = readdlm(joinpath(pathdata, "depth_observations.csv"), ',', header=true)[1][:,2]
        moorings = readdlm(joinpath(pathdata, "acoustic_moorings.csv"), ',', header=true)[1]
        acoustic_pos = tuple.(moorings[:,2], moorings[:,3])
        acoustic_obs = readdlm(joinpath(pathdata, "acoustic_observations.csv"), ',', header=true)[1][:,2:3]
        acoustic_obs = Int.(acoustic_obs')

        # initial values: Matrix{Float64}
        p0 = zeros(size(bathymetry_map)[1], size(bathymetry_map)[2])
        idx = GeoArrays.indices(bathymetry_map, (709757.111649658, 6.26772603565296e6)) # last known location of the fish
        bathymetry_map[idx]
        p0[idx] = 1

        h = 200                     # spatial resolution [m]
        D = 200^2                   # Diffusion coefficient, i.e. variance of one time step movement [m^2]

        tsave = 105:5:200           # time steps to save

        # run tracker
        res = track(p0, bathymetry_map; tsave = tsave,
                    D = D, h = h,
                    depth_obs = depth_obs,
                    acoustic_obs = acoustic_obs, acoustic_pos = acoustic_pos,
                    smoothing = true)

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
