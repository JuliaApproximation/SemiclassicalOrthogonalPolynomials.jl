"""
MulAddAccumulate(μ, A, B)

represents the vector satisfying v[k+1] == A[k]*v[k]+B[k] with v[1] == μ
"""
mutable struct MulAddAccumulate{T} <: AbstractCachedVector{T}
    data::Vector{T}
    A::AbstractVector
    B::AbstractVector
    datasize::Tuple{Int}
    function MulAddAccumulate{T}(data::Vector{T}, A::AbstractVector, B::AbstractVector, datasize::Tuple{Int}) where T
        length(A) == length(B) || throw(ArgumentError("lengths must match"))
        new{T}(data, A, B, datasize)
    end
end

size(M::MulAddAccumulate) = size(M.A)

MulAddAccumulate(data::Vector{T}, A::AbstractVector, B::AbstractVector, datasize::Tuple{Int}) where T =
    MulAddAccumulate{T}(data, A, B, datasize)
MulAddAccumulate(μ, A, B) = MulAddAccumulate([μ], A, B, (1,))
MulAddAccumulate(A, B) = MulAddAccumulate(A[1]+B[1], A, B)

function LazyArrays.cache_filldata!(K::MulAddAccumulate, inds)
    A,B = K.A,K.B
    @inbounds for k in inds
        K.data[k] = muladd(A[k], K.data[k-1], B[k])
    end
end

@simplify function *(D::Derivative, P::SemiclassicalJacobi)
    Q = SemiclassicalJacobi(P.t, P.a+1,P.b+1,P.c+1,P)
    A,B,_ = recurrencecoefficients(P)
    α,β,_ = recurrencecoefficients(Q)

    d = AccumulateAbstractVector(*, A ./ Vcat(1,α))
    v1 = AccumulateAbstractVector(+, B ./ A)
    v2 = MulAddAccumulate(Vcat(0,0,α[2:∞]) ./ α, Vcat(0,β ./ α) ./ α);
    v3 = AccumulateAbstractVector(*, Vcat(A[1]A[2], A[3:∞] ./ α))
    Q * (_BandedMatrix(Vcat(((1:∞) .* d)', (((1:∞) .* (v1 .+ B[2:end]./A[2:end]) .- (2:∞) .* (α .* v2 .+ β ./ α)) .* v3)'), ∞, 2,-1))'
end

@simplify function *(D::Derivative, wQ::Weighted{<:Any,<:SemiclassicalJacobi})
    Q = wQ.P
    P = SemiclassicalJacobi(Q.t, Q.a-1,Q.b-1,Q.c-1)
    Weighted(P) * ((-sum(orthogonalityweight(Q))/sum(orthogonalityweight(P))) * (Q \ (D * P))')
end

@simplify function *(D::Derivative, HP::HalfWeighted{:a,<:Any,<:SemiclassicalJacobi})
    P = HP.P
    Q = SemiclassicalJacobi(P.t, P.a-1, P.b+1, P.c+1)
    a = Q.a
    A,B,C = recurrencecoefficients(P)
    α,β,γ = recurrencecoefficients(Q)
    d = AccumulateAbstractVector(*, A ./ α)
    v1 = MulAddAccumulate(Vcat(0,0,α[2:∞] ./ α), Vcat(0,β))
    v2 = MulAddAccumulate(Vcat(0,0,A[2:∞] ./ α), Vcat(0,B[1], B[2:end] .* d))

    HalfWeighted{:a}(Q) * _BandedMatrix(
        Vcat(
        ((a:∞) .* v2 .- ((a+1):∞) .* Vcat(1,v1[2:end] .* d))',
        (((a+1):∞) .* Vcat(1,d))'), ℵ₀, 0,1)
end

@simplify function *(D::Derivative, HP::HalfWeighted{:b,<:Any,<:SemiclassicalJacobi})
    P = HP.P
    Q = SemiclassicalJacobi(P.t, P.a+1, P.b-1, P.c+1)
    b = Q.b
    A,B,C = recurrencecoefficients(P)
    α,β,γ = recurrencecoefficients(Q)
    d = AccumulateAbstractVector(*, A ./ α)
    d2 = AccumulateAbstractVector(*, A ./ Vcat(1,α))
    v1 = MulAddAccumulate(Vcat(0,0,α[2:∞] ./ α), Vcat(0,β))
    v2 = MulAddAccumulate(Vcat(0,0,A[2:∞] ./ α), Vcat(0,B[1], B[2:end] .* d))

    HalfWeighted{:b}(Q) * _BandedMatrix(
        Vcat(
        (-(b:∞) .* v2 .+ ((b+1):∞) .* Vcat(1,v1[2:end] .* d) .+ Vcat(0,(1:∞) .* d2))',
        (-((b+1):∞) .* Vcat(1,d))'), ℵ₀, 0,1)
end

@simplify function *(D::Derivative, HP::HalfWeighted{:c,<:Any,<:SemiclassicalJacobi})
    P = HP.P
    t = P.t
    Q = SemiclassicalJacobi(t, P.a+1, P.b+1, P.c-1)
    c = Q.c
    A,B,C = recurrencecoefficients(P)
    α,β,γ = recurrencecoefficients(Q)
    d = AccumulateAbstractVector(*, A ./ α)
    d2 = AccumulateAbstractVector(*, A ./ Vcat(1,α))
    v1 = MulAddAccumulate(Vcat(0,0,α[2:∞] ./ α), Vcat(0,β))
    v2 = MulAddAccumulate(Vcat(0,0,A[2:∞] ./ α), Vcat(0,B[1], B[2:end] .* d))

    HalfWeighted{:c}(Q) * _BandedMatrix(
        Vcat(
        (-(c:∞) .* v2 .+ ((c+1):∞) .* Vcat(1,v1[2:end] .* d) .+ Vcat(0,(t:t:∞) .* d2))',
        (-((c+1):∞) .* Vcat(1,d))'), ℵ₀, 0,1)
end

@simplify function *(D::Derivative, HP::HalfWeighted{:ab,<:Any,<:SemiclassicalJacobi})
    P = HP.P
    t = P.t
    Q = SemiclassicalJacobi(t, P.a,P.b,P.c-1,P)
    A,B,_ = recurrencecoefficients(P)
    α,β,_ = recurrencecoefficients(Q)

    d = AccumulateAbstractVector(*, A ./ Vcat(1,α))
    v1 = AccumulateAbstractVector(+, B ./ A)
    v2 = MulAddAccumulate(Vcat(0,0,α[2:∞]) ./ α, Vcat(0,β ./ α) ./ α);
    v3 = AccumulateAbstractVector(*, Vcat(A[1]A[2], A[3:∞] ./ α))
    # HalfWeighted{:ab}(Q) * _BandedMatrix(Vcat((((1:∞) .* (v1 .+ B[2:end]./A[2:end]) .- (2:∞) .* (α .* v2 .+ β ./ α)) .* v3)', ((P.b-P.a):∞) ./ d)'), ∞, 2,-1)
end