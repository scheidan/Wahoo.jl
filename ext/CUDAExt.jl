## -------------------------------------------------------
## This module is only loaded if CUDA.jl and cuDNN.jl are imported
## -------------------------------------------------------


module CUDAExt

using Wahoo
using CUDA
import cuDNN
import NNlib

function move_to_GPU(H, bathymetry, observations, dist_acoustic)


    H = CuArray{Float32}(H)
    bathymetry = CuArray{Float32}(bathymetry[:,:,1])
    observations = [(obs[1], CuArray{Float32}(obs[2])) for obs in observations]
    dist_acoustic = CuArray{Float32}(dist_acoustic)


    @info "Using GPU: $(CUDA.current_device())"
    return H, bathymetry, observations, dist_acoustic
end

end
