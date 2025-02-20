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


# ----

"""
If `dist` is a CuArray of probabilities, sample an element and returns the first two indices.

Uses the Max-Gumbel trick, see: https://arxiv.org/abs/2110.01515
"""
function Wahoo.sample_index(dist::CuArray)
    logw = log.(dist)
    # Generate uniform random numbers on the GPU.
    # N.B. We must avoid 0f0 and 1f0 !!!
    U = clamp.(CUDA.rand(size(dist)...), eps(eltype(dist)), 1 - eps(eltype(dist)))
    # Compute Gumbel noise: G = -log(-log(U))
    G = -log.(-log.(U))
    # The Gumbel trick: add noise to log(weights)...
    scores = logw .+ G
    # ...and find the index of the maximum score.
    _, idxs = findmax(scores)
    return Int32(idxs[1]), Int32(idxs[2])
end


# ----

"""
Set all values of A (of size (m,n,1,1)) to zero except A[y,x] to one.
"""
function Wahoo.one_hot!(A::CuArray, y, x)
    # Convert indices to 32-bit integers for efficiency.
    y32 = Int32(y)
    x32 = Int32(x)
    # Choose thread and block dimensions (adjust as needed).
    threads = (16, 16)
    blocks  = (cld(size(A, 1), threads[1]), cld(size(A, 2), threads[2]))
    CUDA.@sync begin
        @cuda threads=threads blocks=blocks kernel_one_hot!(A, y32, x32)
    end
end

function kernel_one_hot!(A, y, x)
    # Compute global indices for the first and second dimensions.
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    # Update only if within bounds (assumes A has shape (M, N, 1, 1))
    if i <= size(A, 1) && j <= size(A, 2)
        if i == y && j == x
            @inbounds A[i, j, 1, 1] = one(eltype(A))
        else
            @inbounds A[i, j, 1, 1] = zero(eltype(A))
        end
    end
    nothing
end


end
