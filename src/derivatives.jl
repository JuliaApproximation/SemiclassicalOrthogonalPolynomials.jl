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

"""
    divmul(A, B, C)

is equivalent to A \\ (B*C)
"""
function divmul(Q::SemiclassicalJacobi, D::Derivative, P::SemiclassicalJacobi)
    A,B,_ = recurrencecoefficients(P)
    α,β,_ = recurrencecoefficients(Q)

    d = AccumulateAbstractVector(*, A ./ Vcat(1,α))
    v1 = AccumulateAbstractVector(+, B ./ A)
    v2 = MulAddAccumulate(Vcat(0,0,α[2:∞]) ./ α, Vcat(0,β ./ α) ./ α);
    v3 = AccumulateAbstractVector(*, Vcat(A[1]A[2], A[3:∞] ./ α))
    _BandedMatrix(Vcat(((1:∞) .* d)', (((1:∞) .* (v1 .+ B[2:end]./A[2:end]) .- (2:∞) .* (α .* v2 .+ β ./ α)) .* v3)'), ∞, 2,-1)'
end

@simplify function *(D::Derivative, P::SemiclassicalJacobi)
    Q = SemiclassicalJacobi(P.t, P.a+1,P.b+1,P.c+1,P)
    Q * divmul(Q, D, P)
end




function divmul(wP::Weighted{<:Any,<:SemiclassicalJacobi}, D::Derivative, wQ::Weighted{<:Any,<:SemiclassicalJacobi})
    Q,P = wQ.P,wP.P
    ((-sum(orthogonalityweight(Q))/sum(orthogonalityweight(P))) * (Q \ (D * P))')
end

@simplify function *(D::Derivative, wQ::Weighted{<:Any,<:SemiclassicalJacobi})
    wP = Weighted(SemiclassicalJacobi(wQ.P.t, wQ.P.a-1,wQ.P.b-1,wQ.P.c-1))
    wP * divmul(wP, D, wQ)
end


##
# One-Weighted
##

function divmul(HQ::HalfWeighted{:a,<:Any,<:SemiclassicalJacobi}, D::Derivative, HP::HalfWeighted{:a,<:Any,<:SemiclassicalJacobi})
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

function divmul(HQ::HalfWeighted{:b,<:Any,<:SemiclassicalJacobi}, D::Derivative, HP::HalfWeighted{:b,<:Any,<:SemiclassicalJacobi})
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

function divmul(HQ::HalfWeighted{:c,<:Any,<:SemiclassicalJacobi}, D::Derivative, HP::HalfWeighted{:c,<:Any,<:SemiclassicalJacobi})
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

@simplify function *(D::Derivative, HP::HalfWeighted{:a,<:Any,<:SemiclassicalJacobi})
    P = HP.P
    t = P.t
    HQ = HalfWeighted{:a}(SemiclassicalJacobi(t, P.a-1, P.b+1, P.c+1))
    HQ * divmul(HQ, D, HP)
end

@simplify function *(D::Derivative, HP::HalfWeighted{:b,<:Any,<:SemiclassicalJacobi})
    P = HP.P
    t = P.t
    HQ = HalfWeighted{:b}(SemiclassicalJacobi(t, P.a+1, P.b-1, P.c+1))
    HQ * divmul(HQ, D, HP)
end

@simplify function *(D::Derivative, HP::HalfWeighted{:c,<:Any,<:SemiclassicalJacobi})
    P = HP.P
    t = P.t
    HQ = HalfWeighted{:c}(SemiclassicalJacobi(t, P.a+1, P.b+1, P.c-1))
    HQ * divmul(HQ, D, HP)
end

##
# Double-Weighted
##

function divmul(HQ::HalfWeighted{:ab}, D::Derivative, HP::HalfWeighted{:ab})
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


function divmul(HQ::HalfWeighted{:bc}, D::Derivative, HP::HalfWeighted{:bc})
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

function divmul(HQ::HalfWeighted{:ac}, D::Derivative, HP::HalfWeighted{:ac})
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



@simplify function *(D::Derivative, HP::HalfWeighted{:ab,<:Any,<:SemiclassicalJacobi})
    P = HP.P
    t = P.t
    HQ = HalfWeighted{:ab}(SemiclassicalJacobi(t, P.a-1,P.b-1,P.c+1))
    HQ * divmul(HQ, D, HP)
end


@simplify function *(D::Derivative, HP::HalfWeighted{:ac,<:Any,<:SemiclassicalJacobi})
    P = HP.P
    t = P.t
    HQ = HalfWeighted{:ac}(SemiclassicalJacobi(t, P.a-1,P.b+1,P.c-1))
    HQ * divmul(HQ, D, HP)
end


@simplify function *(D::Derivative, HP::HalfWeighted{:bc,<:Any,<:SemiclassicalJacobi})
    P = HP.P
    t = P.t
    HQ = HalfWeighted{:bc}(SemiclassicalJacobi(t, P.a+1,P.b-1,P.c-1))
    HQ * divmul(HQ, D, HP)
end