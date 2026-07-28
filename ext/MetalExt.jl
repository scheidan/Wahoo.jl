## -------------------------------------------------------
## This module is only loaded if Metal.jl is imported
## -------------------------------------------------------


module MetalExt

using Wahoo
using Metal
import NNlib

function to32bitarray(v, precision)
    T = eltype(v)
    if T <: AbstractFloat
        MtlArray{precision}(v)
    elseif T <: Integer
        precision == Float32 ? MtlArray{Int32}(v) : MtlArray{Int64}(v)
    else
        error("Type $T cannot be converted into a MtlArray!")
    end
end


function move_to_GPU(H, bathymetry, observations, dist_acoustic, precision)

    H = MtlArray{precision}(H)
    bathymetry = MtlArray{precision}(bathymetry[:,:,1])
    observations = [to32bitarray(obs, precision) for obs in observations]
    dist_acoustic = MtlArray{precision}(dist_acoustic)

    @info "Using GPU: $(Metal.device().name)"
    return H, bathymetry, observations, dist_acoustic
end


function Wahoo.diffuse!(src::MtlArray, dest::MtlArray, H::MtlArray, hops)
    threads = (16, 16)
    groups = (cld(size(src, 1), threads[1]), cld(size(src, 2), threads[2]))
    for _ in 1:hops
        Metal.@metal threads=threads groups=groups kernel_diffuse!(dest, src, H)
        src, dest = dest, src
    end
    return src, dest
end

function kernel_diffuse!(dest, src, H)
    pos = Metal.thread_position_in_grid()
    i = Int32(pos.x)
    j = Int32(pos.y)
    ny = Int32(size(src, 1))
    nx = Int32(size(src, 2))
    if i <= ny && j <= nx
        @inbounds begin
            v = H[2,2,1,1] * src[i,j,1,1]
            i > 1 && (v += H[1,2,1,1] * src[i-1,j,1,1])
            i < ny && (v += H[3,2,1,1] * src[i+1,j,1,1])
            j > 1 && (v += H[2,1,1,1] * src[i,j-1,1,1])
            j < nx && (v += H[2,3,1,1] * src[i,j+1,1,1])
            dest[i,j,1,1] = v
        end
    end
    nothing
end


# ----

"""
If `dist` is a MtlArray of probabilities, sample an element and returns the first two indices.

Uses the Max-Gumbel trick, see: https://arxiv.org/abs/2110.01515
"""
function Wahoo.sample_index(dist::MtlArray)
    logw = log.(dist)
    # Generate uniform random numbers on the GPU.
    # N.B. We must avoid 0f0 and 1f0 !!!
    U = clamp.(Metal.rand(size(dist)...), eps(eltype(dist)), 1 - eps(eltype(dist)))
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
function Wahoo.one_hot!(A::MtlArray, y, x)
    # Convert indices to 32-bit integers for efficiency.
    y32 = Int32(y)
    x32 = Int32(x)
    # Choose thread and threadgroup dimensions (adjust as needed).
    threads = (16, 16)
    groups = (cld(size(A, 1), threads[1]), cld(size(A, 2), threads[2]))
    Metal.@sync begin
        Metal.@metal threads=threads groups=groups kernel_one_hot!(A, y32, x32)
    end
end

function kernel_one_hot!(A, y, x)
    # Compute global indices for the first and second dimensions.
    pos = Metal.thread_position_in_grid()
    i = Int32(pos.x)
    j = Int32(pos.y)
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
