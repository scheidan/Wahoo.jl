
import NNlib
import GeoArrays

# ---
# depth likelihood

function p_depth(obs, waterdepth)
    if obs > waterdepth         # water is too shallow
        return zero(waterdepth)
    else
        # # uniform depth
        # one(obs)/waterdepth

        # exponential
        scale = 30f0
        Z = 1 - exp(-waterdepth/scale) # normalisation due to truncation
        exp(-(waterdepth - obs)/scale)/(scale * Z)
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

    return one(T)               # sensor not active
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
