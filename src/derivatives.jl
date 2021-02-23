"""
represents cumsum(A ./
"""
mutable struct CumsumRatio{T} <: AbstractCachedVector{T}
    A # OPs
    B
    data::Vector{T}
    datasize::Tuple{Int}
end

"""
represents cumprod(A ./ B)
"""
mutable struct CumprodRatio{T} <: AbstractCachedVector{T}
    A # OPs
    B
    data::Vector{T}
    datasize::Tuple{Int}
end

size(::Union{CumsumRatio,CumprodRatio}) = (ℵ₀,)

function LazyArrays.cache_filldata!(K::CumsumRatio, inds)
    A,B,_ = recurrencecoefficients(K.P)
    @inbounds for k in inds
        K.data[k] =  K.data[k-1] + K.A[k]/K.B[k]
    end
end

function LazyArrays.cache_filldata!(K::CumprodRatio, inds)
    @inbounds for k in inds
        K.data[k] = K.A[k]/K.B[k] * K.data[k-1]
    end
end

