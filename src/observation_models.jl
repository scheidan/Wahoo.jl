
import NNlib
import GeoArrays


# -------

function build_distances(sensor_positions::Vector,
                         bathymetry::GeoArrays.GeoArray, h)

    ny, nx = size(bathymetry)[1:2]
    distances = zeros(Float32, ny, nx, length(sensor_positions))

    for (k, pos) in enumerate(sensor_positions)
        if !isnothing(pos)
            idx = GeoArrays.indices(bathymetry, pos)
            distances[:,:,k] .= [sqrt((i-idx[1])^2 + (j-idx[2])^2) * h
                                     for i in 1:ny, j in 1:nx]
        end
    end

    distances
end



# ---
# depth likelihood

function p_depth_uniform(signal, waterdepth, dist)
    if signal > waterdepth         # water is too shallow
        return zero(waterdepth)
    else
        # uniform depth
        one(waterdepth)/waterdepth
    end
end


function p_depth_exponential(signal, waterdepth, dist; scale=30f0)
    if signal > waterdepth         # water is too shallow
        return zero(waterdepth)
    else
        # exponential
        Z = 1 - exp(-waterdepth/scale) # normalisation due to truncation
        exp(-(waterdepth - signal)/scale)/(scale * Z)
    end
end

# ---
# acoustic likelihood

function p_acoustic_sigmoid(signal, waterdepth, dist; d0 = 1000f0, k = 200f0)

    p = if signal == 0
        # 1/(1 + exp(-(dist - d0)/k))
        NNlib.sigmoid((dist - d0)/k)
    elseif signal == 1
        # 1 - 1/(1 + exp(-(dist - d0)/k))
        1 - NNlib.sigmoid((dist - d0)/k)
    else  # sensor not active
        one(eltype(dist))
    end

    return p

end
