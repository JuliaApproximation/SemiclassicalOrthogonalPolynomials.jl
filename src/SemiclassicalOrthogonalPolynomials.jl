module SemiclassicalOrthogonalPolynomials

using ApproxFun
import ApproxFun: evaluate, domain,
                    domainspace, rangespace, bandwidths, prectype, canonicaldomain, tocanonical,
                    spacescompatible, points, transform, itransform, AbstractProductSpace,
                    checkpoints, plan_transform, clenshaw
import ApproxFunOrthogonalPolynomials: PolynomialSpace, recα, recβ, recγ, recA, recB, recC
import ApproxFunBase: tensorizer, columnspace
import Base: in, *
using StaticArrays
using FastGaussQuadrature
using LinearAlgebra
using SparseArrays
using BlockBandedMatrices
using BlockArrays
using GenericLinearAlgebra
using Test

export OrthogonalPolynomialFamily, HalfDisk, HalfDiskSpace, HalfDiskFamily


abstract type SpaceFamily{D,R} end

struct OrthogonalPolynomialSpace{FA,WW,F,D,B,R,N} <: PolynomialSpace{D,R}
    family::FA # Pointer back to the family
    weight::Vector{WW} # The full product weight
    params::NTuple{N,B} # The powers of the weight factors (i.e. key to this
                        # space in the dict of the family) (could be BigFloats)
    ops::Vector{F}  # Cache the ops we get for free from lanczos
    a::Vector{B} # Diagonal recurrence coefficients
    b::Vector{B} # Off diagonal recurrence coefficients
    opnorm::Vector{B} # The norm of the OPs (all OPs of an OPSpace have the same norm).
                      # NOTE this is the value of the norm squared
    opptseval::Vector{Vector{B}}
    derivopptseval::Vector{Vector{B}}
end

# Finds the OPs and recurrence for weight w, having already found N₀ OPs
function lanczos!(w, P, β, γ; N₀=0)

    # x * P[n](x) == (γ[n] * P[n+1](x) + β[n] * P[n](x) + γ[n-1] * P[n-1](x))

    N = length(β)
    # x = Fun(identity, space(w)) # NOTE: space(w) does not "work" sometimes
    x = Fun(identity, domain(w))

    if N₀ <= 0
        N₀ = 1
        f1 = Fun(1/sqrt(sum(w)),space(x))
        P[1] = f1
        v = x*P[1]
        β[1] = sum(w*v*P[1])
        v = v - β[1]*P[1]
        γ[1] = sqrt(sum(w*v^2))
        P[2] = v/γ[1]
    end

    for k = N₀+1:N
        @show "lanczos", N, k
        v = x*P[2] - γ[k-1]*P[1]
        β[k] = sum(P[2]*w*v)
        v = v - β[k]*P[2]
        γ[k] = sqrt(sum(w*v*v))
        P[1] = P[2]
        P[2] = v/γ[k]
    end

    P, β, γ
end

# Finds the OPs and recurrence for weight w
function lanczos(w, N)
    # x * P[n](x) == (γ[n] * P[n+1](x) + β[n] * P[n](x) + γ[n-1] * P[n-1](x))
    P = Array{Fun}(undef, N + 1)
    β = Array{eltype(w)}(undef, N)
    γ = Array{eltype(w)}(undef, N)
    lanczos!(w, P, β, γ)
end

# TODO/NOTE to prevent building full weight for large param values just to get
# domain, we calculate the product of the weights and use that as the domain
domain(S::OrthogonalPolynomialSpace) = domain(prod(S.family.factors))
canonicaldomain(S::OrthogonalPolynomialSpace) = domain(S)
tocanonical(S::OrthogonalPolynomialSpace, x) = x

function OrthogonalPolynomialSpace(fam::SpaceFamily{D,R}, w::Fun, α::NTuple{N,B}) where {D,R,B,N}
    W = Vector{typeof(w)}(); resize!(W, 1); W[1] = w
    OrthogonalPolynomialSpace{typeof(fam),typeof(w),Fun,D,B,R,N}(
        fam, W, α, Vector{Fun}(), Vector{B}(), Vector{B}(), Vector{B}(),
        Vector{Vector{B}}(), Vector{Vector{B}}())
end

OrthogonalPolynomialSpace(fam::SpaceFamily{D,R}, α::NTuple{N,B}) where {D,R,B,N} =
    OrthogonalPolynomialSpace{typeof(fam),Fun,Fun,D,B,R,N}(
        fam, Vector{Fun}(), α, Vector{Fun}(), Vector{B}(), Vector{B}(),
        Vector{B}(), Vector{Vector{B}}(), Vector{Vector{B}}())

# Creates and returns the Fun() representing the weight function for the OPSpace
function getweightfun(S::OrthogonalPolynomialSpace)
    if length(S.weight) == 0
        # @show "getweightfun() for OPSpace", S.params
        resize!(S.weight, 1)
        if length(S.params) == 1
            S.weight[1] = (S.family.factors.^(S.params))[1]
        else
            S.weight[1] = prod(S.family.factors.^(S.params))
        end
    end
    S.weight[1]
end

# Calls lanczos!() to get the recurrence coeffs for the OPSpace up to deg n
function resizedata!(S::OrthogonalPolynomialSpace, n)
    N₀ = length(S.a)
    n ≤ N₀ && return S
    resize!(S.a, n)
    resize!(S.b, n)
    resize!(S.ops, 2)
    # We set the weight here when this is called
    @show "resizedata! for OPSpace", Float64.(S.params)
    lanczos!(getweightfun(S), S.ops, S.a, S.b, N₀=N₀)
    S
end

# R is range-type, which should be Float64. B is the r-type of the weight Funs,
# which could be BigFloats
struct OrthogonalPolynomialFamily{OPS,FF,D,R,B,N} <: SpaceFamily{D,R}
    factors::FF
    spaces::Dict{NTuple{N,B}, OPS}
end
function OrthogonalPolynomialFamily(w::Vararg{Fun{<:Space{D,B}},N}) where {D,B,N}
    all(domain.(w) .== Ref(domain(first(w)))) || throw(ArgumentError("domains incompatible"))
    R = Float64 # TODO - is there a way to not hardcode this? (see below)
    spaces = Dict{NTuple{N,R}, OrthogonalPolynomialSpace}()
    OrthogonalPolynomialFamily{OrthogonalPolynomialSpace,typeof(w),D,R,B,N}(w, spaces)
end
function OrthogonalPolynomialFamily(::Type{R}, w::Vararg{Fun{<:Space{D,B}},N}) where {D,R,B,N}
    all(domain.(w) .== Ref(domain(first(w)))) || throw(ArgumentError("domains incompatible"))
    spaces = Dict{NTuple{N,R}, OrthogonalPolynomialSpace}()
    OrthogonalPolynomialFamily{OrthogonalPolynomialSpace,typeof(w),D,R,B,N}(w, spaces)
end

function (P::OrthogonalPolynomialFamily{<:Any,<:Any,<:Any,R,B,N})(α::Vararg{B,N}) where {R,B,N}
    haskey(P.spaces,α) && return P.spaces[α]
    # We set the weight when we call resizedata!()
    P.spaces[α] = OrthogonalPolynomialSpace(P, α)
end

#======#
# Methods to return the recurrence coeffs

#####
# recα/β/γ are given by
#       x p_{n-1} = γ_n p_{n-2} + α_n p_{n-1} +  p_n β_n
#####
recα(::Type{T}, S::OrthogonalPolynomialSpace, n) where T =
    T(resizedata!(S, n).a[n])
recβ(::Type{T}, S::OrthogonalPolynomialSpace, n) where T =
    T(resizedata!(S, n).b[n])
recγ(::Type{T}, S::OrthogonalPolynomialSpace, n) where T =
    T(resizedata!(S, n).b[n-1])

#####
# recA/B/C are given by
#       p_{n+1} = (A_n x + B_n)p_n - C_n p_{n-1}
#####
recA(::Type{T}, S::OrthogonalPolynomialSpace, n) where T =
    1 / recβ(T, S, n+1)
recB(::Type{T}, S::OrthogonalPolynomialSpace, n) where T =
    -recα(T, S, n+1) / recβ(T, S, n+1)
recC(::Type{T}, S::OrthogonalPolynomialSpace, n) where T =
    recγ(T, S, n+1) / recβ(T, S, n+1)

#======#
# points() and associanted methods

# Returns weights and nodes for N-point quad rule for given weight
function golubwelsch(S::OrthogonalPolynomialSpace{<:Any,<:Any,<:Any,<:Any,B,T,<:Any},
                        N::Integer) where {B,T}
    resizedata!(S, N)                # 3-term recurrence
    J = SymTridiagonal(T.(S.a[1:N]), T.(S.b[1:N-1]))   # Jacobi matrix
    D, V = eigen(J)                  # Eigenvalue decomposition using BigFloats
    indx = sortperm(D)               # Hermite points
    μ = getopnorm(S)                 # Integral of weight function
    w = μ * V[1, indx].^2            # quad rule weights to output
    x = D[indx]                      # quad rule nodes to output
    return T.(x), T.(w)
end
# Returns, as type B, weights and nodes for N-point quad rule for given weight
function golubwelsch(::Type{B}, S::OrthogonalPolynomialSpace{<:Any,<:Any,<:Any,<:Any,B,T,<:Any},
                        N::Integer) where {B,T}
    resizedata!(S, N)                # 3-term recurrence
    J = SymTridiagonal(S.a[1:N], S.b[1:N-1])   # Jacobi matrix
    D, V = eigen(J)                  # Eigenvalue decomposition using BigFloats
    indx = sortperm(D)               # Hermite points
    μ = getopnorm(S)                 # Integral of weight function
    w = μ * V[1, indx].^2            # quad rule weights to output
    x = D[indx]                      # quad rule nodes to output
    return B.(x), B.(w)
end
points(S::OrthogonalPolynomialSpace, n) = golubwelsch(S, n)[1]
pointswithweights(S::OrthogonalPolynomialSpace, n) = golubwelsch(S, n)
pointswithweights(::Type{B}, S::OrthogonalPolynomialSpace, n) where B =
    golubwelsch(B, S, n)

spacescompatible(A::OrthogonalPolynomialSpace, B::OrthogonalPolynomialSpace) =
    A.weight ≈ B.weight

#=====#
# transforms

# Inputs: OP space, f(pts) for desired f
# Output: Coeffs of the function f for its expansion in the OPSpace OPs
function transform(S::OrthogonalPolynomialSpace, vals::Vector{T}) where T
    n = length(vals)
    pts, w = pointswithweights(S, n)
    getopptseval(S, n-1, pts)
    cfs = zeros(T, n)
    for k = 0:n-1
        cfs[k+1] = T(inner2(S, opevalatpts(S, k+1, pts), vals, w) / getopnorm(S))
    end
    cfs
    # # Vandermonde matrix transposed, including weights and normalisations
    # Ṽ = Array{T}(undef, n, n)
    # for k = 0:n-1
    #     pk = Fun(S, [zeros(k); 1])
    #     nrm = sum([pk(pts[j])^2 * w[j] for j = 1:n])
    #     Ṽ[k+1, :] = pk.(pts) .* w / nrm
    # end
    # Ṽ * vals
end

# Inputs: OP space, coeffs of a function f for its expansion in the OPSpace OPs
# Output: vals = {f(x_j)} where x_j are are the points(S,n)
function itransform(S::OrthogonalPolynomialSpace, cfs::Vector{T}) where T
    n = length(cfs)
    pts, w = pointswithweights(S, n)
    vals = zeros(T, n)
    getopptseval(S, n-1, pts)
    for k = 1:n
        vals[k] = T(sum([cfs[j] * opevalatpts(S, j, pts)[k] for j = 1:n]))
    end
    # # Vandermonde matrix
    # V = Array{T}(undef, n, n)
    # for k = 0:n-1
    #     pk = Fun(S, [zeros(k); 1])
    #     V[:, k+1] = pk.(pts)
    # end
    # V * cfs
end
end