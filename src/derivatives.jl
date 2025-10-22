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

########## Derivatives
mutable struct DivDiffData{T,V1,V2,V3,AA,BB,CC,DD} <: AbstractCachedVector{T}
    data::Vector{T}
    const v1::V1 
    const v2::V2
    const v3::V3
    const A::AA 
    const B::BB
    const α::CC
    const β::DD
    datasize::Tuple{Int}
end
function LazyArrays.cache_filldata!(K::DivDiffData, inds)
    v1, v2, v3 = K.v1, K.v2, K.v3
    A, B, α, β = K.A, K.B, K.α, K.β
    @inbounds for n in inds
        K.data[n] = (n * (v1[n] + B[n+1]/A[n+1]) - (n+1) * (α[n]*v2[n] + β[n]/α[n])) * v3[n]
    end
end
size(K::DivDiffData) = size(K.v1)
function DivDiffData(v1, v2, v3, A, B, α, β)
    T = Base.promote_type(eltype(v1), eltype(v2), eltype(v3), eltype(A), eltype(B), eltype(α), eltype(β)) # This is just Base.promote_eltype, but promote_eltype is an internal function
    data = zeros(T, 0)
    return DivDiffData(data, v1, v2, v3, A, B, α, β, size(data))
end

"""
    divdiff(A, C)

is equivalent to A \\ diff(C)
"""
function divdiff(Q::SemiclassicalJacobi, P::SemiclassicalJacobi)
    A,B,_ = recurrencecoefficients(P)
    α,β,_ = recurrencecoefficients(Q)

    d = AccumulateAbstractVector(*, A ./ Vcat(1,α))
    v1 = AccumulateAbstractVector(+, B ./ A)
    v2 = MulAddAccumulate(Vcat(0,0,α[2:∞]) ./ α, Vcat(0,β ./ α) ./ α)
    v3 = AccumulateAbstractVector(*, Vcat(A[1]A[2], A[3:∞] ./ α))
    data = DivDiffData(v1, v2, v3, A, B, α, β)
    return _BandedMatrix(Vcat(((1:∞) .* d)', data'), ∞, 2,-1)'
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
        bdata = bandeddata(Dmat)
        b2 = Vcat(zero(T),  bdata[1,:])
        b1 = Vcat(zero(T),  bdata[2,:])
        data = Vcat(b2', b1')
        D = _BandedMatrix(data, ∞, -1, 2)
        return Pᵗᵃ⁺¹⁰ᶜ⁺¹ * D
    end
end

function divdiff(wP::Weighted{<:Any,<:SemiclassicalJacobi}, wQ::Weighted{<:Any,<:SemiclassicalJacobi})
    Q,P = wQ.P,wP.P
    (-sum(orthogonalityweight(Q))/sum(orthogonalityweight(P))) * (Q \ diff(P))'
end

function diff(wQ::Weighted{<:Any,<:SemiclassicalJacobi}; dims=1)
    wP = Weighted(SemiclassicalJacobi(wQ.P.t, wQ.P.a-1,wQ.P.b-1,wQ.P.c-1))
    wP * divdiff(wP, wQ)
end


##
# One-Weighted
##
mutable struct WeightedDivDiffDataA1{T,A,D,V1,V2} <: AbstractCachedVector{T}
    data::Vector{T}
    const a::A 
    const d::D
    const v1::V1
    const v2::V2 
    datasize::Tuple{Int}
end
mutable struct WeightedDivDiffDataA2{T,A,D} <: AbstractCachedVector{T}
    data::Vector{T}
    const a::A 
    const d::D
    datasize::Tuple{Int}
end
function LazyArrays.cache_filldata!(K::WeightedDivDiffDataA1{T}, inds) where {T}
    a, d, v1, v2 = K.a, K.d, K.v1, K.v2
    @inbounds for n in inds
        K.data[n] = (a + n - 1) * v2[n] - (a + n) * (n == 1 ? one(T) : v1[n] * d[n-1])
    end
end
function LazyArrays.cache_filldata!(K::WeightedDivDiffDataA2{T}, inds) where {T}
    a, d = K.a, K.d
    @inbounds for n in inds
        K.data[n] = (a + n) * (n == 1 ? one(T) : d[n-1])
    end
end
size(K::WeightedDivDiffDataA1) = size(K.v1)
size(K::WeightedDivDiffDataA2) = size(K.d)
function WeightedDivDiffDataA1(a, d, v1, v2)
    T = Base.promote_type(eltype(a), eltype(d), eltype(v1), eltype(v2))
    data = zeros(T, 0)
    return WeightedDivDiffDataA1(data, a, d, v1, v2, size(data))
end
function WeightedDivDiffDataA2(a, d)
    T = Base.promote_type(eltype(a), eltype(d))
    data = zeros(T, 0)
    return WeightedDivDiffDataA2(data, a, d, size(data))
end

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
    p = WeightedDivDiffDataA1(a, d, v1, v2)
    q = WeightedDivDiffDataA2(a, d)
    _BandedMatrix(Vcat(p', q'), ℵ₀, 0,1)
end

mutable struct WeightedDivDiffDataB1{T,B,D,D2,V1,V2} <: AbstractCachedVector{T}
    data::Vector{T}
    const b::B 
    const d::D
    const d2::D2 
    const v1::V1
    const v2::V2 
    datasize::Tuple{Int}
end
mutable struct WeightedDivDiffDataB2{T,B,D} <: AbstractCachedVector{T}
    data::Vector{T}
    const b::B 
    const d::D
    datasize::Tuple{Int}
end
function LazyArrays.cache_filldata!(K::WeightedDivDiffDataB1{T}, inds) where {T}
    b, d, d2, v1, v2 = K.b, K.d, K.d2, K.v1, K.v2
    @inbounds for n in inds
        K.data[n] = -(b + n - 1) * v2[n] + (b + n) * (n == 1 ? one(T) : v1[n] * d[n-1]) + (n == 1 ? zero(T) : (n-1) * d2[n-1])
    end
end
function LazyArrays.cache_filldata!(K::WeightedDivDiffDataB2{T}, inds) where {T}
    b, d = K.b, K.d
    @inbounds for n in inds
        K.data[n] = -(b + n) * (n == 1 ? one(T) : d[n-1])
    end
end
size(K::WeightedDivDiffDataB1) = size(K.v1)
size(K::WeightedDivDiffDataB2) = size(K.d)
function WeightedDivDiffDataB1(b, d, d2, v1, v2)
    T = Base.promote_type(eltype(b), eltype(d), eltype(d2), eltype(v1), eltype(v2))
    data = zeros(T, 0)
    return WeightedDivDiffDataB1(data, b, d, d2, v1, v2, size(data))
end
function WeightedDivDiffDataB2(b, d)
    T = Base.promote_type(eltype(b), eltype(d))
    data = zeros(T, 0)
    return WeightedDivDiffDataB2(data, b, d, size(data))
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
    p = WeightedDivDiffDataB1(b, d, d2, v1, v2)
    q = WeightedDivDiffDataB2(b, d)
    _BandedMatrix(Vcat(p', q'), ℵ₀, 0,1)
end

mutable struct WeightedDivDiffDataC1{T,TT,C,D,D2,V1,V2} <: AbstractCachedVector{T}
    data::Vector{T}
    const t::TT
    const c::C
    const d::D
    const d2::D2 
    const v1::V1
    const v2::V2 
    datasize::Tuple{Int}
end
mutable struct WeightedDivDiffDataC2{T,C,D} <: AbstractCachedVector{T}
    data::Vector{T}
    const c::C
    const d::D
    datasize::Tuple{Int}
end
function LazyArrays.cache_filldata!(K::WeightedDivDiffDataC1{T}, inds) where {T}
    t, c, d, d2, v1, v2 = K.t, K.c, K.d, K.d2, K.v1, K.v2
    @inbounds for n in inds
        K.data[n] = -(c + n - 1) * v2[n] + (c + n) * (n == 1 ? one(T) : v1[n] * d[n-1]) + (n == 1 ? zero(T) : t*(n-1)* d2[n-1])
    end
end
function LazyArrays.cache_filldata!(K::WeightedDivDiffDataC2{T}, inds) where {T}
    c, d = K.c, K.d
    @inbounds for n in inds
        K.data[n] = -(c + n) * (n == 1 ? one(T) : d[n-1])
    end
end
size(K::WeightedDivDiffDataC1) = size(K.v1)
size(K::WeightedDivDiffDataC2) = size(K.d)
function WeightedDivDiffDataC1(t, c, d, d2, v1, v2)
    T = Base.promote_type(eltype(t), eltype(c), eltype(d), eltype(d2), eltype(v1), eltype(v2))
    data = zeros(T, 0)
    return WeightedDivDiffDataC1(data, t, c, d, d2, v1, v2, size(data))
end
function WeightedDivDiffDataC2(c, d)
    T = Base.promote_type(eltype(c), eltype(d))
    data = zeros(T, 0)
    return WeightedDivDiffDataC2(data, c, d, size(data))
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
    p = WeightedDivDiffDataC1(t, c, d, d2, v1, v2)
    q = WeightedDivDiffDataC2(c, d)

    _BandedMatrix(Vcat(p', q'), ℵ₀, 0,1)
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

mutable struct WeightedDivDiffDataAB1{T,A,B,E,F,G} <: AbstractCachedVector{T}
    data::Vector{T}
    const a::A 
    const b::B 
    const e::E 
    const f::F 
    const g::G
    datasize::Tuple{Int}
end
mutable struct WeightedDivDiffDataAB2{T,A,B,D} <: AbstractCachedVector{T}
    data::Vector{T}
    const a::A 
    const b::B 
    const d::D
    datasize::Tuple{Int}
end
function LazyArrays.cache_filldata!(K::WeightedDivDiffDataAB1{T}, inds) where {T}
    a, b, e, f, g = K.a, K.b, K.e, K.f, K.g
    @inbounds for n in inds
        K.data[n] = (a + n) * e[n] - (b + a + n) * f[n] + (b + a + n + 1) * e[n] * g[n] 
    end
end
function LazyArrays.cache_filldata!(K::WeightedDivDiffDataAB2{T}, inds) where {T}
    a, b, d = K.a, K.b, K.d
    @inbounds for n in inds
        K.data[n] = -(a + b + n + 1) * d[n]
    end
end
size(K::WeightedDivDiffDataAB1) = size(K.e)
size(K::WeightedDivDiffDataAB2) = size(K.d)
function WeightedDivDiffDataAB1(a, b, e, f, g)
    T = Base.promote_type(eltype(a), eltype(b), eltype(e), eltype(f), eltype(g))
    data = zeros(T, 0)
    return WeightedDivDiffDataAB1(data, a, b, e, f, g, size(data))
end
function WeightedDivDiffDataAB2(a, b, d)
    T = Base.promote_type(eltype(a), eltype(b), eltype(d))
    data = zeros(T, 0)
    return WeightedDivDiffDataAB2(data, a, b, d, size(data))
end

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

    p = WeightedDivDiffDataAB1(a, b, e, f, g)
    q = WeightedDivDiffDataAB2(a, b, d)

    return _BandedMatrix(Vcat(p', q'), ℵ₀, 1, 0)
end

mutable struct WeightedDivDiffDataBC1{T,TT,B,C,E,F,G} <: AbstractCachedVector{T}
    data::Vector{T}
    const t::TT 
    const b::B 
    const c::C 
    const e::E 
    const f::F 
    const g::G 
    datasize::Tuple{Int}
end
mutable struct WeightedDivDiffDataBC2{T,B,C,D} <: AbstractCachedVector{T}
    data::Vector{T}
    const b::B 
    const c::C 
    const d::D
    datasize::Tuple{Int}
end
function LazyArrays.cache_filldata!(K::WeightedDivDiffDataBC1{T}, inds) where {T}
    t, b, c, e, f, g = K.t, K.b, K.c, K.e, K.f, K.g
    @inbounds for n in inds
        K.data[n] = -((t+1) * (n - 1) + (t * (b + 1) + c + 1)) * e[n] + (c + b + n) * f[n] - (b + c + n + 1) * e[n] * g[n]
    end
end
function LazyArrays.cache_filldata!(K::WeightedDivDiffDataBC2{T}, inds) where {T}
    b, c, d = K.b, K.c, K.d
    @inbounds for n in inds
        K.data[n] = (b + c + n + 1) * d[n]
    end
end
size(K::WeightedDivDiffDataBC1) = size(K.e)
size(K::WeightedDivDiffDataBC2) = size(K.d)
function WeightedDivDiffDataBC1(t, b, c, e, f, g)
    T = Base.promote_type(eltype(t), eltype(b), eltype(c), eltype(e), eltype(f), eltype(g))
    data = zeros(T, 0)
    return WeightedDivDiffDataBC1(data, t, b, c, e, f, g, size(data))
end
function WeightedDivDiffDataBC2(b, c, d)
    T = Base.promote_type(eltype(b), eltype(c), eltype(d))
    data = zeros(T, 0)
    return WeightedDivDiffDataBC2(data, b, c, d, size(data))
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

    p = WeightedDivDiffDataBC1(t, b, c, e, f, g)
    q = WeightedDivDiffDataBC2(b, c, d)

    return _BandedMatrix(Vcat(p', q'), ℵ₀, 1, 0)
end

mutable struct WeightedDivDiffDataAC1{T,TT,A,C,E,F,G} <: AbstractCachedVector{T}
    data::Vector{T}
    const t::TT 
    const a::A 
    const c::C 
    const e::E 
    const f::F 
    const g::G
    datasize::Tuple{Int}
end
mutable struct WeightedDivDiffDataAC2{T,A,C,D} <: AbstractCachedVector{T}
    data::Vector{T}
    const a::A 
    const c::C 
    const d::D
    datasize::Tuple{Int}
end
function LazyArrays.cache_filldata!(K::WeightedDivDiffDataAC1{T}, inds) where {T}
    t, a, c, e, f, g = K.t, K.a, K.c, K.e, K.f, K.g
    @inbounds for n in inds
        K.data[n] = t * (a + n) * e[n] - (c + a + n) * f[n] + (a + c + n + 1) * e[n] * g[n]
    end
end
function LazyArrays.cache_filldata!(K::WeightedDivDiffDataAC2{T}, inds) where {T}
    a, c, d = K.a, K.c, K.d
    @inbounds for n in inds
        K.data[n] = -(a + c + n + 1) * d[n]
    end
end
size(K::WeightedDivDiffDataAC1) = size(K.e)
size(K::WeightedDivDiffDataAC2) = size(K.d)
function WeightedDivDiffDataAC1(t, a, c, e, f, g)
    T = Base.promote_type(eltype(t), eltype(a), eltype(c), eltype(e), eltype(f), eltype(g))
    data = zeros(T, 0)
    return WeightedDivDiffDataAC1(data, t, a, c, e, f, g, size(data))
end
function WeightedDivDiffDataAC2(a, c, d)
    T = Base.promote_type(eltype(a), eltype(c), eltype(d))
    data = zeros(T, 0)
    return WeightedDivDiffDataAC2(data, a, c, d, size(data))
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

    p = WeightedDivDiffDataAC1(t, a, c, e, f, g)
    q = WeightedDivDiffDataAC2(a, c, d)

    return _BandedMatrix(Vcat(p', q'), ℵ₀, 1, 0)
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