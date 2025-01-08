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
    pos = CuArray{Float32}(undef, (nx, ny, 1, length(tsave)))

    memory_demand = round(nx * ny * length(tsave) * 4 / 1024^2, digits=0) # in Mega bytes
    memory_free = round(CUDA.free_memory() / 1024^2, digits=0)
	@info "Using GPU: $(CUDA.current_device())"
    @info "Filter computation requires at least $memory_demand Mb of GPU memmory ($memory_free Mb free)"

    return H, bathymetry, dist_acoustic, pos
end

end
