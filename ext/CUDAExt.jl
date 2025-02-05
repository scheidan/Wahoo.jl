## -------------------------------------------------------
## This module is only loaded if CUDA.jl and cuDNN.jl are imported
## -------------------------------------------------------


module CUDAExt

using Wahoo
using CUDA
import cuDNN
import NNlib

function to32bitarray(v)
    T = eltype(v)
    if T <: AbstractFloat
        CuArray{Float32}(v)
    elseif T <: Int
        CuArray{Int32}(v)
    else
        error("Type $T cannot be converted into a CuArray!")
    end
end


function move_to_GPU(H, bathymetry, observations, dist_acoustic)

    H = CuArray{Float32}(H)
    bathymetry = CuArray{Float32}(bathymetry[:,:,1])
    observations = [to32bitarray(obs) for obs in observations]
    dist_acoustic = CuArray{Float32}(dist_acoustic)

    @info "Using GPU: $(CUDA.current_device())"
    return H, bathymetry, observations, dist_acoustic
end

end
