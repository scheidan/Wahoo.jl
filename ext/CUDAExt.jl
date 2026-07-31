## -------------------------------------------------------
## This module is only loaded if CUDA.jl is imported
## -------------------------------------------------------


module CUDAExt

using Wahoo
using CUDA
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

    H = [CuArray{precision}(kernel) for kernel in H]
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
    return Int32(idxs[1]), Int32(idxs[2]), Int32(idxs[3])
end


# ----

"""
Set all values of `A` to zero except the joint state at `(y, x, a)`.
"""
function Wahoo.one_hot!(A::CuArray, y, x, a)
    # Convert indices to 32-bit integers for efficiency.
    y32 = Int32(y)
    x32 = Int32(x)
    a32 = Int32(a)
    # Choose thread and block dimensions (adjust as needed).
    threads = (16, 16)
    blocks  = (cld(size(A, 1), threads[1]), cld(size(A, 2), threads[2]))
    CUDA.@sync begin
        @cuda threads=threads blocks=blocks kernel_one_hot!(A, y32, x32, a32)
    end
end

function kernel_one_hot!(A, y, x, a)
    # Compute global indices for the first and second dimensions.
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    # Update only if within bounds
    if i <= size(A, 1) && j <= size(A, 2)
        for k in axes(A, 3)
            if i == y && j == x && k == a
                @inbounds A[i, j, k, 1] = one(eltype(A))
            else
                @inbounds A[i, j, k, 1] = zero(eltype(A))
            end
        end
    end
    nothing
end


end
