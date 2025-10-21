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

########## Structs for representing specific broadcast operators to get around inefficient broadcast machinery 
## TODO: Implement something like a CacheStyle for BroadcastStyle that can do all this more generically
mutable struct CachedBroadcastVector{T,F,A,B} <: AbstractCachedVector{T}
    data::Vector{T}
    const f::F
    const A::A
    const B::B
    datasize::Tuple{Int}
    function CachedBroadcastVector{T}(data::Vector{T}, f::F, a::A, b::B, datasize::Tuple{Int}) where {T,F,A,B}
        length(a) == length(b) || throw(ArgumentError("lengths must match"))
        new{T,F,A,B}(data,f,a,b,datasize)
    end
end
const CBV = CachedBroadcastVector

size(K::CachedBroadcastVector) = size(K.A)

CachedBroadcastVector(data::Vector{T}, f, a::A, b::B, datasize) where {T, A, B} = CachedBroadcastVector{T}(data, f, a, b, datasize)
CachedBroadcastVector(f, a, b) = CachedBroadcastVector([f(first(a), first(b))], f, a, b, (1,))

function LazyArrays.cache_filldata!(K::CachedBroadcastVector{T, F, A, B}, inds) where {T, F, A, B}
    @show inds
    f, a, b = K.f, K.A, K.B
    resizedata!(a, maximum(inds)) 
    resizedata!(b, maximum(inds))
    @inbounds for k in inds
        K.data[k] = f(a[k], b[k])
    end
end


########## Derivatives

"""
    divdiff(A, C)

is equivalent to A \\ diff(C)
"""
function divdiff(Q::SemiclassicalJacobi, P::SemiclassicalJacobi)
    A,B,_ = recurrencecoefficients(P)
    α,β,_ = recurrencecoefficients(Q)

    d = AccumulateAbstractVector(*, CBV(/, A, Vcat(1,α)))
    v1 = AccumulateAbstractVector(+, CBV(/, B, A))
    v2 = MulAddAccumulate(CBV(/, Vcat(0,0,α[2:∞]), α), CBV(/, Vcat(0,CBV(/, β,  α)),  α))
    v3 = AccumulateAbstractVector(*, Vcat(A[1]A[2], CBV(/, A[3:∞], α)))
    op1 = CBV(*, 1:∞, d)
    op2 = CBV(/, B[2:end], A[2:end])
    op3 = CBV(+, v1, op2)
    op4 = CBV(*, 1:∞, op3)
    op5 = CBV(/, β, α)
    op6 = CBV(*, α, v2)
    op7 = CBV(+, op5, op6)
    op8 = CBV(*, 2:∞, op7)
    op9 = CBV(-, op4, op8)
    op10 = CBV(*, op9, v3) 
    return _BandedMatrix(Vcat(op1', op10'), ∞, 2,-1)'
    # return _BandedMatrix(Vcat(((1:∞) .* d)', (((1:∞) .* (v1 .+ B[2:end]./A[2:end]) .- (2:∞) .* (α .* v2 .+ β ./ α)) .* v3)'), ∞, 2,-1)'
end

function diff(P::SemiclassicalJacobi{T}; dims=1) where {T}
    if P.b ≠ -1
        Q = SemiclassicalJacobi(P.t, P.a+1,P.b+1,P.c+1,P)
        Q * divdiff(Q, P)
    elseif P.b == -1
        P1 = SemiclassicalJacobi(P.t, P.a, one(P.b), P.c, P)
        WP1 = HalfWeighted{:b}(P1)
        D = diff(WP1)
        Pᵗᵃ⁺¹⁰ᶜ⁺¹ = D.args[1].P
        Dmat = D.args[2]
        b2 = Vcat(zero(T), zero(T), Dmat[band(1)])
        b1 = Vcat(zero(T), Dmat[band(0)])
        data = Hcat(b2, b1)'
        D = _BandedMatrix(data, ∞, -1, 2)
        return Pᵗᵃ⁺¹⁰ᶜ⁺¹ * D
    end
end

function divdiff(wP::Weighted{<:Any,<:SemiclassicalJacobi}, wQ::Weighted{<:Any,<:SemiclassicalJacobi})
    Q,P = wQ.P,wP.P
    ((-sum(orthogonalityweight(Q))/sum(orthogonalityweight(P))) * (Q \ diff(P))')
end

function diff(wQ::Weighted{<:Any,<:SemiclassicalJacobi}; dims=1)
    wP = Weighted(SemiclassicalJacobi(wQ.P.t, wQ.P.a-1,wQ.P.b-1,wQ.P.c-1))
    wP * divdiff(wP, wQ)
end


##
# One-Weighted
##

function divdiff(HQ::HalfWeighted{:a,<:Any,<:SemiclassicalJacobi}, HP::HalfWeighted{:a,<:Any,<:SemiclassicalJacobi})
    Q = HQ.P
    P = HP.P
    t = P.t
    a = Q.a
    A,B,C = recurrencecoefficients(P)
    α,β,γ = recurrencecoefficients(Q)
    d = AccumulateAbstractVector(*, A ./ α)
    v1 = MulAddAccumulate(Vcat(0,0,α[2:∞] ./ α), Vcat(0,β))
    v2 = MulAddAccumulate(Vcat(0,0,A[2:∞] ./ α), Vcat(0,B[1], B[2:end] .* d))

    _BandedMatrix(
        Vcat(
            ((a:∞) .* v2 .- ((a+1):∞) .* Vcat(1,v1[2:end] .* d))',
            (((a+1):∞) .* Vcat(1,d))'), ℵ₀, 0,1)
end

function divdiff(HQ::HalfWeighted{:b,<:Any,<:SemiclassicalJacobi}, HP::HalfWeighted{:b,<:Any,<:SemiclassicalJacobi})
    Q = HQ.P
    P = HP.P
    t = P.t
    b = Q.b
    A,B,C = recurrencecoefficients(P)
    α,β,γ = recurrencecoefficients(Q)
    d = AccumulateAbstractVector(*, A ./ α)
    d2 = AccumulateAbstractVector(*, A ./ Vcat(1,α))
    v1 = MulAddAccumulate(Vcat(0,0,α[2:∞] ./ α), Vcat(0,β))
    v2 = MulAddAccumulate(Vcat(0,0,A[2:∞] ./ α), Vcat(0,B[1], B[2:end] .* d))

    _BandedMatrix(
        Vcat(
            (-(b:∞) .* v2 .+ ((b+1):∞) .* Vcat(1,v1[2:end] .* d) .+ Vcat(0,(1:∞) .* d2))',
            (-((b+1):∞) .* Vcat(1,d))'), ℵ₀, 0,1)
end

function divdiff(HQ::HalfWeighted{:c,<:Any,<:SemiclassicalJacobi}, HP::HalfWeighted{:c,<:Any,<:SemiclassicalJacobi})
    Q = HQ.P
    P = HP.P
    t = P.t
    c = Q.c
    A,B,C = recurrencecoefficients(P)
    α,β,γ = recurrencecoefficients(Q)
    d = AccumulateAbstractVector(*, A ./ α)
    d2 = AccumulateAbstractVector(*, A ./ Vcat(1,α))
    v1 = MulAddAccumulate(Vcat(0,0,α[2:∞] ./ α), Vcat(0,β))
    v2 = MulAddAccumulate(Vcat(0,0,A[2:∞] ./ α), Vcat(0,B[1], B[2:end] .* d))
    _BandedMatrix(
        Vcat(
            (-(c:∞) .* v2 .+ ((c+1):∞) .* Vcat(1,v1[2:end] .* d) .+ Vcat(0,(t:t:∞) .* d2))',
            (-((c+1):∞) .* Vcat(1,d))'), ℵ₀, 0,1)
end

function diff(HP::HalfWeighted{:a,<:Any,<:SemiclassicalJacobi}; dims=1)
    P = HP.P
    t = P.t
    HQ = HalfWeighted{:a}(SemiclassicalJacobi(t, P.a-1, P.b+1, P.c+1))
    HQ * divdiff(HQ, HP)
end

function diff(HP::HalfWeighted{:b,<:Any,<:SemiclassicalJacobi}; dims=1)
    P = HP.P
    t = P.t
    HQ = HalfWeighted{:b}(SemiclassicalJacobi(t, P.a+1, P.b-1, P.c+1))
    HQ * divdiff(HQ, HP)
end

function diff(HP::HalfWeighted{:c,<:Any,<:SemiclassicalJacobi}; dims=1)
    P = HP.P
    t = P.t
    HQ = HalfWeighted{:c}(SemiclassicalJacobi(t, P.a+1, P.b+1, P.c-1))
    HQ * divdiff(HQ, HP)
end

##
# Double-Weighted
##

function divdiff(HQ::HalfWeighted{:ab}, HP::HalfWeighted{:ab})
    Q = HQ.P
    P = HP.P
    A,B,_ = recurrencecoefficients(P)
    α,β,_ = recurrencecoefficients(Q)
    a,b = Q.a,Q.b

    d = AccumulateAbstractVector(*, Vcat(1,A) ./ α)
    e = AccumulateAbstractVector(*, Vcat(1,A ./ α))
    f = MulAddAccumulate(Vcat(0,0,A[2:end] ./ α[2:end]), Vcat(0, (B./ α) .* e))
    g = cumsum(β ./ α)
    _BandedMatrix(Vcat((((a+1):∞) .* e .- ((b+a+1):∞).*f .+ ((a+b+2):∞) .* e .* g )',
            (-((a+b+2):∞)  .* d)'),ℵ₀,1,0)
end


function divdiff(HQ::HalfWeighted{:bc}, HP::HalfWeighted{:bc})
    Q = HQ.P
    P = HP.P
    t = P.t
    A,B,_ = recurrencecoefficients(P)
    α,β,_ = recurrencecoefficients(Q)
    b,c = Q.b,Q.c

    d = AccumulateAbstractVector(*, Vcat(1,A) ./ α)
    e = AccumulateAbstractVector(*, Vcat(1,A ./ α))
    f = MulAddAccumulate(Vcat(0,0,A[2:end] ./ α[2:end]), Vcat(0, (B./ α) .* e))
    g = cumsum(β ./ α)
    _BandedMatrix(Vcat((-((t+1)* (0:∞) .+ (t*(b+1) + c+1)) .* e .+ ((c+b+1):∞).*f .- ((b+c+2):∞) .* e .* g )',
            (((b+c+2):∞)  .* d)'),ℵ₀,1,0)
end

function divdiff(HQ::HalfWeighted{:ac}, HP::HalfWeighted{:ac})
    Q = HQ.P
    P = HP.P
    t = P.t
    A,B,_ = recurrencecoefficients(P)
    α,β,_ = recurrencecoefficients(Q)
    a,c = Q.a,Q.c

    d = AccumulateAbstractVector(*, Vcat(1,A) ./ α)
    e = AccumulateAbstractVector(*, Vcat(1,A ./ α))
    f = MulAddAccumulate(Vcat(0,0,A[2:end] ./ α[2:end]), Vcat(0, (B./ α) .* e))
    g = cumsum(β ./ α)
    _BandedMatrix(Vcat((t* ((a+1):∞) .* e .- ((c+a+1):∞).*f .+ ((a+c+2):∞) .* e .* g )',
            (-((a+c+2):∞)  .* d)'),ℵ₀,1,0)
end



function diff(HP::HalfWeighted{:ab,<:Any,<:SemiclassicalJacobi}; dims=1)
    P = HP.P
    t = P.t
    HQ = HalfWeighted{:ab}(SemiclassicalJacobi(t, P.a-1,P.b-1,P.c+1))
    HQ * divdiff(HQ, HP)
end


function diff(HP::HalfWeighted{:ac,<:Any,<:SemiclassicalJacobi}; dims=1)
    P = HP.P
    t = P.t
    HQ = HalfWeighted{:ac}(SemiclassicalJacobi(t, P.a-1,P.b+1,P.c-1))
    HQ * divdiff(HQ, HP)
end


function diff(HP::HalfWeighted{:bc,<:Any,<:SemiclassicalJacobi}; dims=1)
    P = HP.P
    t = P.t
    HQ = HalfWeighted{:bc}(SemiclassicalJacobi(t, P.a+1,P.b-1,P.c-1))
    HQ * divdiff(HQ, HP)
end