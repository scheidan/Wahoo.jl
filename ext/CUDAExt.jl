## -------------------------------------------------------
## This module is only loaded if CUDA.jl and cuDNN.jl are loaded
## -------------------------------------------------------


module CUDAExt

using Wahoo
using CUDA
import cuDNN
import NNlib

function move_to_GPU(H, bathymetry, dist_acoustic, tsave)

    nx, ny = size(bathymetry)[1:2]
    H = CuArray{Float32}(H)
    bathymetry = CuArray{Float32}(bathymetry[:,:,1])
    dist_acoustic = CuArray{Float32}(dist_acoustic)


    @info "Using GPU: $(CUDA.current_device())"
    return H, bathymetry, dist_acoustic
end

end
