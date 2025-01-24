
import NNlib
import GeoArrays



struct Signal{T, S, V}
    p_obs::T
    observations::S
    location::Union{V, Nothing}
end


function build_distances(signals::Vector{<:Signal},
                         bathymetry::GeoArrays.GeoArray, h)

    ny, nx = size(bathymetry)[1:2]
    distances = zeros(Float32, ny, nx, length(signals))

    for k in 1:length(signals)
        if !isnothing(sig.location)
        idx = GeoArrays.indices(bathymetry, sig.location[k])
        dist_acoustic[:,:,k] .= [sqrt((i-idx[1])^2 + (j-idx[2])^2) * h
                                 for i in 1:ny, j in 1:nx]
    end

    distances
end



# ---
# depth likelihood

function p_depth(obs, waterdepth, dist)
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

function p_acoustic(obs::Int, waterdepth, dist::T; d0 = 400f0, k = 100f0)::T where T
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
