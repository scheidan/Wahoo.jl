## -------------------------------------------------------
## This module is only loaded if CUDA.jl and cuDNN.jl are imported
## -------------------------------------------------------


module CUDAExt

using Wahoo
using CUDA
import cuDNN
import NNlib

function to32bitarray(v, precision)
    T = eltype(v)
    if T <: AbstractFloat
        CuArray{precision}(v)
    elseif T <: Int
        precision == Float32 ? CuArray{Int32}(v) : CuArray{Int64}(v)
    else
        error("Type $T cannot be converted into a CuArray!")
    end
end


function move_to_GPU(H, bathymetry, observations, dist_acoustic, precision)

    H = CuArray{precision}(H)
    bathymetry = CuArray{precision}(bathymetry[:,:,1])
    observations = [to32bitarray(obs, precision) for obs in observations]
    dist_acoustic = CuArray{precision}(dist_acoustic)

    @info "Using GPU: $(CUDA.current_device())"
    return H, bathymetry, observations, dist_acoustic
end

end
