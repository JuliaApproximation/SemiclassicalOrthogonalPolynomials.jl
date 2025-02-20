module SemiclassicalOrthogonalPolynomials
using ClassicalOrthogonalPolynomials: WeightedOPLayout
using ClassicalOrthogonalPolynomials, FillArrays, LazyArrays, ArrayLayouts, QuasiArrays, InfiniteArrays, ContinuumArrays, LinearAlgebra, BandedMatrices,
        SpecialFunctions, HypergeometricFunctions, InfiniteLinearAlgebra

import Base: getindex, axes, size, \, /, *, +, -, summary, show, ==, copy, sum, unsafe_getindex, convert, OneTo, diff

import ArrayLayouts: MemoryLayout, ldiv, diagonaldata, subdiagonaldata, supdiagonaldata
import BandedMatrices: bandwidths, AbstractBandedMatrix, BandedLayout, _BandedMatrix
import LazyArrays: resizedata!, paddeddata, CachedVector, CachedMatrix, CachedAbstractVector, LazyMatrix, LazyVector, arguments, ApplyLayout, colsupport, AbstractCachedVector, ApplyArray,
                    AccumulateAbstractVector, LazyVector, AbstractCachedMatrix, BroadcastLayout, simplifiable
import ClassicalOrthogonalPolynomials: OrthogonalPolynomial, recurrencecoefficients, jacobimatrix, normalize, _p0, UnitInterval, orthogonalityweight, NormalizedOPLayout, MappedOPLayout,
                                        Bidiagonal, Tridiagonal, SymTridiagonal, symtridiagonalize, normalizationconstant, LanczosPolynomial,
                                        OrthogonalPolynomialRatio, Weighted, AbstractWeightLayout, UnionDomain, oneto, WeightedBasis, HalfWeighted,
                                        golubwelsch, AbstractOPLayout, weight, cholesky_jacobimatrix, qr_jacobimatrix, isnormalized, ConvertedOrthogonalPolynomial, AbstractNormalizedOPLayout

import InfiniteArrays: OneToInf, InfUnitRange
import ContinuumArrays: basis, Weight, @simplify, AbstractBasisLayout, BasisLayout, MappedBasisLayout, grid, plotgrid, equals_layout, ExpansionLayout
import FillArrays: SquareEye
import HypergeometricFunctions: _₂F₁general2, _₂F₁
import SpecialFunctions: beta

export Legendre, Normalized, normalize, SemiclassicalJacobi, SemiclassicalJacobiWeight, WeightedSemiclassicalJacobi, OrthogonalPolynomialRatio

""""
    SemiclassicalJacobiWeight(t, a, b, c)

is a quasi-vector corresponding to the weight `x^a * (1-x)^b * (t-x)^c` on `x in 0..1`
"""
struct SemiclassicalJacobiWeight{T} <: Weight{T}
    t::T
    a::T
    b::T
    c::T
end

function SemiclassicalJacobiWeight(t, a, b, c)
    T = promote_type(eltype(t), eltype(a), eltype(b), eltype(c))
    SemiclassicalJacobiWeight{T}(t, a, b, c)
end

copy(w::SemiclassicalJacobiWeight) = w

axes(P::SemiclassicalJacobiWeight{T}) where T = (Inclusion(UnitInterval{T}()),)
function getindex(P::SemiclassicalJacobiWeight, x::Real)
    t,a,b,c = P.t,P.a,P.b,P.c
    @boundscheck checkbounds(P, x)
    x^a * (1-x)^b * (t-x)^c
end

function sum(P::SemiclassicalJacobiWeight{T}) where T
    (t,a,b,c) = (P.t, P.a, P.b, P.c)
    t,a,b,c = convert(BigFloat,t),convert(BigFloat,a),convert(BigFloat,b),convert(BigFloat,c) # This is needed at high parameter values
    return abs(convert(T, t^c*exp(loggamma(a+1)+loggamma(b+1)-loggamma(a+b+2)) * pFq((a+1,-c),(a+b+2, ), 1/t)))
end

show(io::IO, P::SemiclassicalJacobiWeight) = summary(io, P)
function summary(io::IO, P::SemiclassicalJacobiWeight)
    t,a,b,c = P.t,P.a,P.b,P.c
    print(io, "x^$a * (1-x)^$b * ($t-x)^$c on 0..1")
end

function ==(A::SemiclassicalJacobiWeight, B::SemiclassicalJacobiWeight)
    A.a == B.a && A.b == B.b &&
        ((iszero(A.c) && iszero(B.c)) ||
         (A.t == B.t && A.c == B.c))
end

function jacobiexpansion(w::SemiclassicalJacobiWeight{T}) where T
    t,a,b,c = w.t,w.a,w.b,w.c
    P = jacobi(b, a, UnitInterval{T}())
    x = axes(P,1)
    # TODO: simplify
    LanczosPolynomial(@.(x^a * (1-x)^b * (t-x)^c), P).w
end

equals_layout(::AbstractWeightLayout, ::ExpansionLayout, A::SemiclassicalJacobiWeight, B) = jacobiexpansion(A) == B
equals_layout(::ExpansionLayout, ::AbstractWeightLayout, A, B::SemiclassicalJacobiWeight) = A == jacobiexpansion(B)


"""
   RaisedOP(P, y)

Gives the OPs w.r.t. (y - x) .* w based on lowering to Q.
"""
struct RaisedOP{T, QQ, LL<:OrthogonalPolynomialRatio} <: OrthogonalPolynomial{T}
    Q::QQ
    ℓ::LL
end

RaisedOP(Q, ℓ::OrthogonalPolynomialRatio) = RaisedOP{eltype(Q),typeof(Q),typeof(ℓ)}(Q, ℓ)
RaisedOP(Q, y::Number) = RaisedOP(Q, OrthogonalPolynomialRatio(Q,y))


function jacobimatrix(P::RaisedOP{T}) where T
    ℓ = P.ℓ
    X = jacobimatrix(P.Q)
    a,b = diagonaldata(X), supdiagonaldata(X)
    # non-normalized lower diag of Jacobi
    v = Vcat(zero(T),b .* ℓ)
    c = BroadcastVector((ℓ,a,b,sa,v) -> ℓ*a + b - b*ℓ^2 - sa*ℓ + ℓ*v, ℓ, a, b, a[2:∞], v)
    Tridiagonal(c, BroadcastVector((ℓ,a,b,v) -> a - b * ℓ + v, ℓ,a,b,v), b)
end

"""
   SemiclassicalJacobi(t, a, b, c)

is a quasi-matrix for the  orthogonal polynomials w.r.t. x^a * (1-x)^b * (t-x)^c on [0,1]
"""
struct SemiclassicalJacobi{T} <: OrthogonalPolynomial{T}
    t::T
    a::T
    b::T
    c::T
    X::AbstractMatrix{T}
    SemiclassicalJacobi{T}(t::T,a::T,b::T,c::T,X::AbstractMatrix{T}) where T = new{T}(t,a,b,c,X)
end

const WeightedSemiclassicalJacobi{T} = WeightedBasis{T,<:SemiclassicalJacobiWeight,<:SemiclassicalJacobi}

function SemiclassicalJacobi(t, a, b, c, X::AbstractMatrix)
    T = float(promote_type(typeof(t), typeof(a), typeof(b), typeof(c), eltype(X)))
    SemiclassicalJacobi{T}(T(t), T(a), T(b), T(c), convert(AbstractMatrix{T},X))
end

SemiclassicalJacobi(t, a, b, c, P::SemiclassicalJacobi) = SemiclassicalJacobi(t, a, b, c, semiclassical_jacobimatrix(P, a, b, c))
SemiclassicalJacobi(t, a, b, c) = SemiclassicalJacobi(t, a, b, c, semiclassical_jacobimatrix(t, a, b, c))
SemiclassicalJacobi{T}(t, a, b, c) where T = SemiclassicalJacobi(convert(T,t), convert(T,a), convert(T,b), convert(T,c))

WeightedSemiclassicalJacobi{T}(t, a, b, c, P...) where T = SemiclassicalJacobiWeight{T}(convert(T,t), convert(T,a), convert(T,b), convert(T,c)) .* SemiclassicalJacobi{T}(convert(T,t), convert(T,a), convert(T,b), convert(T,c), P...)

# Returns α, β such that P1(x) = β(x-α)
function _linear_coefficients(t, a, b, c)
    # beta(a + 1, b + 1) * t^c * _₂F₁(a + 1, -c, a + b + 2, 1/t) is the integral ∫₀¹ wᵗ⁽ᵃᵇᶜ⁾(x) dx
    Γᵃ = beta(a + 1, b + 1) * _₂F₁(a + 1, -c, a + b + 2, 1/t)
    Γᵃ⁺¹ = beta(a + 2, b + 1) * _₂F₁(a + 2, -c, a + b + 3, 1/t)
    Γᵃ⁺² = beta(a + 3, b + 1) * _₂F₁(a + 3, -c, a + b + 4, 1/t)
    α = Γᵃ⁺¹/Γᵃ
    β = sqrt(Γᵃ / (Γᵃ⁺² - 2α*Γᵃ⁺¹ + α^2*Γᵃ))
    return α, β
end

function semiclassical_jacobimatrix(t, a, b, c)
    T = float(promote_type(typeof(t), typeof(a), typeof(b), typeof(c)))
    if iszero(a) && iszero(b) && c == -one(T)
        # for this special case we can generate the Jacobi operator explicitly
        N = (1:∞)
        α = neg1c_αcfs(one(T)*t)
        A = Vcat((α[1]+1)/2 , -N./(N.*4 .- 2).*α .+ (N.+1)./(N.*4 .+ 2).*α[2:end].+1/2)
        C = -(N)./(N.*4 .- 2)
        B = Vcat((α[1]^2*3-α[1]*α[2]*2-1)/6 , -(N)./(N.*4 .+ 2).*α[2:end]./α)
        return SymTridiagonal(A, sqrt.(B.*C)) # if J is Tridiagonal(c,a,b) then for norm. OPs it becomes SymTridiagonal(a, sqrt.( b.* c))
    elseif b == -one(T)
        J′ = semiclassical_jacobimatrix(t, a, one(b), c)
        J′a, J′b = diagonaldata(J′), supdiagonaldata(J′)
        A = Vcat(one(T), J′a[1:end])
        B = Vcat(-one(T), J′b[1:end])
        C = Vcat(zero(T), J′b[1:end])
        return Tridiagonal(B, A, C)
    else
        P = Normalized(jacobi(b, a, UnitInterval{T}()))
        X = jacobimatrix(P)
        iszero(c) && return X
        if isone(c)
            return cholesky_jacobimatrix(Symmetric(P \ ((t.-axes(P,1)).*P)), X)[1]
        elseif isone(c/2)
            return qr_jacobimatrix(Symmetric(P \ ((t.-axes(P,1)).*P)), X)[1]
        elseif isinteger(c) && c ≥ 0 # reduce other integer c cases to hierarchy
            return SemiclassicalJacobi.(t, a, b, 0:Int(c))[end].X
        else # if c is not an integer, use Lanczos
            x = axes(P,1)
            return cholesky_jacobimatrix(@.(x^a * (1-x)^b * (t-x)^c), P)[1]
        end
    end
end

function semiclassical_jacobimatrix(Q::SemiclassicalJacobi, a, b, c)
    Δa = a-Q.a
    Δb = b-Q.b
    Δc = c-Q.c

    # special cases
    if iszero(a) && iszero(b) && c == -one(eltype(Q.t)) # (a,b,c) = (0,0,-1) special case
        return semiclassical_jacobimatrix(Q.t, zero(Q.t), zero(Q.t), c)
    elseif iszero(Δa) && iszero(Δc) && Δb == 2 && b == 1
        # When going from P[t, a, -1, c] to P[t, a, 1, c], you can just take
        return SymTridiagonal(Q.X.d[2:end], Q.X.du[2:end])
    elseif iszero(c) # classical Jacobi polynomial special case
        return jacobimatrix(Normalized(jacobi(b, a, UnitInterval{eltype(Q.t)}())))
    elseif iszero(Δa) && iszero(Δb) && iszero(Δc) # same basis
        return Q.X
    elseif b == -one(eltype(Q.t))
        return semiclassical_jacobimatrix(Q.t, a, b, c)
    elseif Q.b == -one(eltype(Q.t))
        newQ = SemiclassicalJacobi(Q.t, Q.a, one(Q.b), Q.c, Q) # Constructing first b = 1 from b = -1 is efficient since b = -1 is defined directly in terms of b = 1.
        return semiclassical_jacobimatrix(newQ, a, b, c)
    end

    X = jacobimatrix(Q)
    if isone(Δa/2) && iszero(Δb) && iszero(Δc)  # raising by 2
        qr_jacobimatrix(X,X)[1]
    elseif iszero(Δa) && isone(Δb/2) && iszero(Δc)
        qr_jacobimatrix(I-X,X)[1]
    elseif iszero(Δa) && iszero(Δb) && isone(Δc/2)
        qr_jacobimatrix(Q.t*I-X,X)[1]
    elseif isone(Δa) && iszero(Δb) && iszero(Δc)  # raising by 1
        cholesky_jacobimatrix(X,X)[1]
    elseif iszero(Δa) && isone(Δb) && iszero(Δc)
        cholesky_jacobimatrix(I-X,X)[1]
    elseif iszero(Δa) && iszero(Δb) && isone(Δc)
        cholesky_jacobimatrix(Q.t*I-X,X)[1]
    elseif isone(-Δa) && iszero(Δb) && iszero(Δc) # in these cases we currently have to reconstruct
        # TODO: This is re-constructing. It should instead use reverse Cholesky (or an alternative)!
        semiclassical_jacobimatrix(Q.t,a,b,c)
    elseif iszero(Δa) && isone(-Δb) && iszero(Δc)
        # TODO: This is re-constructing. It should instead use reverse Cholesky (or an alternative)!
        semiclassical_jacobimatrix(Q.t,a,b,c)
    elseif iszero(Δa) && iszero(Δb) && isone(-Δc)
        # TODO: This is re-constructing. It should instead use reverse Cholesky (or an alternative)!
        semiclassical_jacobimatrix(Q.t,a,b,c)
    elseif a > Q.a  # iterative raising by 1
        semiclassical_jacobimatrix(SemiclassicalJacobi(Q.t, Q.a+1, Q.b, Q.c, Q), a, b, c)
    elseif b > Q.b
        semiclassical_jacobimatrix(SemiclassicalJacobi(Q.t, Q.a, Q.b+1, Q.c, Q), a, b, c)
    elseif c > Q.c
        semiclassical_jacobimatrix(SemiclassicalJacobi(Q.t, Q.a, Q.b, Q.c+1, Q), a, b, c)
    elseif a < Q.a  # iterative lowering by 1
        semiclassical_jacobimatrix(SemiclassicalJacobi(Q.t, Q.a-1, Q.b, Q.c, Q), a, b, c)
    elseif b < Q.b
        semiclassical_jacobimatrix(SemiclassicalJacobi(Q.t, Q.a, Q.b-1, Q.c, Q), a, b, c)
    elseif c < Q.c
        semiclassical_jacobimatrix(SemiclassicalJacobi(Q.t, Q.a, Q.b, Q.c-1, Q), a, b, c)
    else
        error("Not Implemented")
    end
end

resizedata!(P::SemiclassicalJacobi, ::Colon, n::Int) = resizedata!(diagonaldata(P.X), n)

ConvertedOrthogonalPolynomial(P::SemiclassicalJacobi{T}) where T = ConvertedOrthogonalPolynomial(orthogonalityweight(P), P.X, P.X.dv.data.U, Normalized(jacobi(P.b, P.a, 0..1)))

"""
    toclassical(P::SemiclassicalJacobi)

gives either a mapped `Jacobi` or `CholeskyPolynomial` version of `P`.
"""
toclassical(P::SemiclassicalJacobi{T}) where T = iszero(P.c) ? Normalized(jacobi(P.b, P.a, UnitInterval{T}())) : ConvertedOrthogonalPolynomial(P)

copy(P::SemiclassicalJacobi) = P
axes(P::SemiclassicalJacobi{T}) where T = (Inclusion(UnitInterval{T}()),OneToInf())

==(A::SemiclassicalJacobi, B::SemiclassicalJacobi) = A.t == B.t && A.a == B.a && A.b == B.b && A.c == B.c
==(::AbstractQuasiMatrix, ::SemiclassicalJacobi) = false
==(::SemiclassicalJacobi, ::AbstractQuasiMatrix) = false
==(::SemiclassicalJacobi, ::SubQuasiArray{<:Any,2}) = false
==(::SubQuasiArray{<:Any,2}, ::SemiclassicalJacobi) = false

orthogonalityweight(P::SemiclassicalJacobi) = SemiclassicalJacobiWeight(P.t, P.a, P.b, P.c)

show(io::IO, P::SemiclassicalJacobi) = summary(io, P)
function summary(io::IO, P::SemiclassicalJacobi)
    t,a,b,c = P.t,P.a,P.b,P.c
    print(io, "SemiclassicalJacobi with weight x^$a * (1-x)^$b * ($t-x)^$c on 0..1")
end


jacobimatrix(P::SemiclassicalJacobi) = P.X

"""
    op_lowering(Q, y)

Gives the Lowering operator from OPs w.r.t. (x-y)*w(x) to Q
as constructed from Chistoffel–Darboux
"""
function op_lowering(Q, y)
    # we first use Christoff-Darboux with d = 1 but the first OP should be 1 so we rescale
    P = RaisedOP(Q, y)
    A,_,_ = recurrencecoefficients(Q)
    d = -inv(A[1]*_p0(Q)*P.ℓ[1])
    κ = d * normalizationconstant(1, P)
    T = eltype(κ)
    # hide array type for compilation
    Bidiagonal(κ, -(κ .* P.ℓ), :L)
end

function semijacobi_ldiv(Q, P::SemiclassicalJacobi)
    if P.a ≤ 0 && P.b ≤ 0 && P.c ≤ 0
        P̃ = toclassical(P)
        (Q \ P̃)/_p0(P̃)
    else
        error("Implement")
    end
end

function semijacobi_ldiv(P::SemiclassicalJacobi, Q)
    R = SemiclassicalJacobi(P.t, mod(P.a,-1), mod(P.b,-1), mod(P.c,-1))
    R̃ = toclassical(R)
    (P \ R) * _p0(R̃) * (R̃ \ Q)
end

"""
    semijacobi_ldiv_direct(Q::SemiclassicalJacobi, P::SemiclassicalJacobi)

Returns conversion operator from SemiclassicalJacobi `P` to SemiclassicalJacobi `Q` in a single step via decomposition.
Numerically unstable if the parameter modification is large. Typically one should instead use `P \\ Q` which is equivalent to `semijacobi_ldiv(P,Q)` and proceeds step by step.
"""
function semijacobi_ldiv_direct(Q::SemiclassicalJacobi, P::SemiclassicalJacobi)
    (Q.t ≈ P.t) && (Q.a ≈ P.a) && (Q.b ≈ P.b) && (Q.c ≈ P.c) && return SquareEye{eltype(Q.t)}((axes(P,2),))
    Δa = Q.a-P.a
    Δb = Q.b-P.b
    Δc = Q.c-P.c
    # special case (Δa,Δb,Δc) = (2,0,0)
    if (Δa == 2) && iszero(Δb) && iszero(Δc)
        M = qr(P.X).R
        return ApplyArray(*, Diagonal(sign.(view(M,band(0)))/abs(M[1])), M)
    # special case (Δa,Δb,Δc) = (0,2,0)
    elseif iszero(Δa) && (Δb == 2) && iszero(Δc)
        M = qr(I-P.X).R
        return ApplyArray(*, Diagonal(sign.(view(M,band(0)))/abs(M[1])), M)
    # special case (Δa,Δb,Δc) = (0,0,2)
    elseif iszero(Δa) && iszero(Δb) && (Δc == 2)
        M = qr(Q.t*I-P.X).R
        return ApplyArray(*, Diagonal(sign.(view(M,band(0)))/abs(M[1])), M)
    # special case (Δa,Δb,Δc) = (-2,0,0)
    elseif (Δa == -2) && iszero(Δb) && iszero(Δc)
        M = qr(Q.X).R
        return ApplyArray(\, M, Diagonal(sign.(view(M,band(0)))*abs(M[1])))
    # special case (Δa,Δb,Δc) = (0,-2,0)
    elseif iszero(Δa) && (Δb == -2) && iszero(Δc)
        M = qr(I-Q.X).R
        return ApplyArray(\, M, Diagonal(sign.(view(M,band(0)))*abs(M[1])))
    # special case (Δa,Δb,Δc) = (0,0,-2)
    elseif iszero(Δa) && iszero(Δb) && (Δc == -2)
        M = qr(Q.t*I-Q.X).R
        return ApplyArray(\, M, Diagonal(sign.(view(M,band(0)))*abs(M[1])))
    # special case (Δa,Δb,Δc) = (1,0,0)
    elseif isone(Δa) && iszero(Δb) && iszero(Δc)
        M = cholesky(P.X).U
        return M/M[1]
    # special case (Δa,Δb,Δc) = (0,1,0)
    elseif iszero(Δa) && isone(Δb) && iszero(Δc)
        M = cholesky(I-P.X).U
        return M/M[1]
    # special case (Δa,Δb,Δc) = (0,0,1)
    elseif iszero(Δa) && iszero(Δb) && isone(Δc)
        M = cholesky(Q.t*I-P.X).U
        return M/M[1]
    # special case (Δa,Δb,Δc) = (-1,0,0)
    elseif isone(-Δa) && iszero(Δb) && iszero(Δc)
        M = cholesky(Q.X).U
        return UpperTriangular(ApplyArray(inv,M/M[1]))
    # special case (Δa,Δb,Δc) = (0,-1,0)
    elseif iszero(Δa) && isone(-Δb) && iszero(Δc)
        M = cholesky(I-Q.X).U
        return UpperTriangular(ApplyArray(inv,M/M[1]))
    # special case (Δa,Δb,Δc) = (0,0,-1)
    elseif iszero(Δa) && iszero(Δb) && isone(-Δc)
        M = cholesky(Q.t*I-Q.X).U
        return UpperTriangular(ApplyArray(inv,M/M[1]))
    elseif isinteger(Δa) && isinteger(Δb) && isinteger(Δc) && (Δa ≥ 0) && (Δb ≥ 0) && (Δc ≥ 0)
        M = cholesky(Symmetric(P.X^(Δa)*(I-P.X)^(Δb)*(Q.t*I-P.X)^(Δc))).U
        return ApplyArray(*, Diagonal(Fill(1/M[1],∞)), M)
    else
        error("Implement modification by ($Δa,$Δb,$Δc)")
    end
end

"""
    semijacobi_ldiv_direct(Q::SemiclassicalJacobi, P::SemiclassicalJacobi)

Returns conversion operator from SemiclassicalJacobi `P` to SemiclassicalJacobi `Q`. Integer distances are covered by decomposition methods.
"""
function semijacobi_ldiv(Q::SemiclassicalJacobi, P::SemiclassicalJacobi)
    @assert Q.t ≈ P.t
    T = promote_type(eltype(Q), eltype(P))
    (Q.t ≈ P.t) && (Q.a ≈ P.a) && (Q.b ≈ P.b) && (Q.c ≈ P.c) && return return SquareEye{eltype(Q.t)}((axes(P,2),))
    Δa = Q.a-P.a
    Δb = Q.b-P.b
    Δc = Q.c-P.c
    if isinteger(Δa) && isinteger(Δb) && isinteger(Δc) # (Δa,Δb,Δc) are integers -> use QR/Cholesky iteratively
        if ((isone(abs(Δa))||(Δa == 2)) && iszero(Δb) && iszero(Δc)) || (iszero(Δa) && (isone(abs(Δb))||(Δb == 2)) && iszero(Δc))  || (iszero(Δa) && iszero(Δb) && (isone(abs(Δc))||(Δc == 2)))
            return semijacobi_ldiv_direct(Q, P)
        elseif Δa > 0  # iterative raising by 1
            QQ = SemiclassicalJacobi(Q.t, Q.a-1-iseven(Δa), Q.b, Q.c, P)
            return ApplyArray(*,semijacobi_ldiv_direct(Q, QQ),semijacobi_ldiv(QQ, P))
        elseif Δb > 0
            QQ = SemiclassicalJacobi(Q.t, Q.a, Q.b-1-iseven(Δb), Q.c, P)
            return ApplyArray(*,semijacobi_ldiv_direct(Q, QQ),semijacobi_ldiv(QQ, P))
        elseif Δc > 0
            QQ = SemiclassicalJacobi(Q.t, Q.a, Q.b, Q.c-1-iseven(Δc), P)
            return ApplyArray(*,semijacobi_ldiv_direct(Q, QQ),semijacobi_ldiv(QQ, P))
        elseif Δa < 0  # iterative lowering by 1
            QQ = SemiclassicalJacobi(Q.t, Q.a+1+iseven(Δa), Q.b, Q.c, P)
            return ApplyArray(*,semijacobi_ldiv_direct(Q, QQ),semijacobi_ldiv(QQ, P))
        elseif Δb < 0
            QQ = SemiclassicalJacobi(Q.t, Q.a, Q.b+1+iseven(Δb), Q.c, P)
            return ApplyArray(*,semijacobi_ldiv_direct(Q, QQ),semijacobi_ldiv(QQ, P))
        elseif Δc < 0
            QQ = SemiclassicalJacobi(Q.t, Q.a, Q.b, Q.c+1+iseven(Δc), P)
            return ApplyArray(*,semijacobi_ldiv_direct(Q, QQ),semijacobi_ldiv(QQ, P))
        end
    else # fallback
        R = SemiclassicalJacobi(P.t, mod(P.a,-1), mod(P.b,-1), mod(P.c,-1))
        R̃ = toclassical(R)
        return (P \ R) * _p0(R̃) * (R̃ \ Q)
    end
end

struct SemiclassicalJacobiLayout <: AbstractOPLayout end
MemoryLayout(::Type{<:SemiclassicalJacobi}) = SemiclassicalJacobiLayout()

copy(L::Ldiv{<:AbstractNormalizedOPLayout,SemiclassicalJacobiLayout}) = copy(Ldiv{ApplyLayout{typeof(*)},SemiclassicalJacobiLayout}(L.A, L.B))
copy(L::Ldiv{SemiclassicalJacobiLayout,<:AbstractNormalizedOPLayout}) = copy(Ldiv{SemiclassicalJacobiLayout,ApplyLayout{typeof(*)}}(L.A, L.B))

copy(L::Ldiv{ApplyLayout{typeof(*)},SemiclassicalJacobiLayout}) = copy(Ldiv{ApplyLayout{typeof(*)},BasisLayout}(L.A, L.B))
copy(L::Ldiv{SemiclassicalJacobiLayout,ApplyLayout{typeof(*)}}) = copy(Ldiv{BasisLayout,ApplyLayout{typeof(*)}}(L.A, L.B))

copy(L::Ldiv{MappedBasisLayout,SemiclassicalJacobiLayout}) = semijacobi_ldiv(L.A, L.B)
copy(L::Ldiv{SemiclassicalJacobiLayout,MappedBasisLayout}) = semijacobi_ldiv(L.A, L.B)

copy(L::Ldiv{WeightedOPLayout,SemiclassicalJacobiLayout}) = copy(Ldiv{WeightedOPLayout,BasisLayout}(L.A, L.B))
copy(L::Ldiv{SemiclassicalJacobiLayout,WeightedOPLayout}) = copy(Ldiv{BasisLayout,WeightedOPLayout}(L.A, L.B))


simplifiable(L::Ldiv{SemiclassicalJacobiLayout,<:AbstractBasisLayout}) = Val(true)
copy(L::Ldiv{SemiclassicalJacobiLayout}) = semijacobi_ldiv(L.A, L.B)
copy(L::Ldiv{SemiclassicalJacobiLayout,<:AbstractBasisLayout}) = semijacobi_ldiv(L.A, L.B)
copy(L::Ldiv{SemiclassicalJacobiLayout,BroadcastLayout{typeof(*)}}) = semijacobi_ldiv(L.A, L.B)
copy(L::Ldiv{<:Any,SemiclassicalJacobiLayout}) = semijacobi_ldiv(L.A, L.B)
copy(L::Ldiv{MappedOPLayout,SemiclassicalJacobiLayout}) = semijacobi_ldiv(L.A, L.B)
copy(L::Ldiv{<:AbstractBasisLayout,SemiclassicalJacobiLayout}) = semijacobi_ldiv(L.A, L.B)
function copy(L::Ldiv{SemiclassicalJacobiLayout,SemiclassicalJacobiLayout})
    Q,P = L.A,L.B
    @assert Q.t == P.t
    Q == P && return SquareEye{eltype(L)}((axes(P,2),))
    M_Q = massmatrix(Q)
    M_P = massmatrix(P)
    L = P \ (SemiclassicalJacobiWeight(Q.t, Q.a-P.a, Q.b-P.b, Q.c-P.c) .* Q)
    (inv(M_Q) * L') * M_P
end

function \(A::SemiclassicalJacobi, B::SemiclassicalJacobi{T}) where {T}
    if A.b == -1 && B.b ≠ -1
        return UpperTriangular(ApplyArray(inv, B \ A))
    elseif B.b == -1 && A.b ≠ -1
        # First convert Bᵗᵃ⁻¹ᶜ into Bᵗᵃ⁰ᶜ
        Bᵗᵃ⁰ᶜ = SemiclassicalJacobi(B.t, B.a, zero(B.b), B.c, A)
        Bᵗᵃ¹ᶜ = SemiclassicalJacobi(B.t, B.a, one(B.a), B.c, A)
        Rᵦₐ₁ᵪᵗᵃ⁰ᶜ = Weighted(Bᵗᵃ⁰ᶜ) \ Weighted(Bᵗᵃ¹ᶜ)
        b1 = Rᵦₐ₁ᵪᵗᵃ⁰ᶜ[band(0)]
        b0 = Vcat(one(T), Rᵦₐ₁ᵪᵗᵃ⁰ᶜ[band(-1)])
        Rᵦₐ₋₁ᵪᵗᵃ⁰ᶜ = Bidiagonal(b0, b1, :U)
        # Then convert Bᵗᵃ⁰ᶜ into A and complete
        Rₐ₀ᵪᴬ = UpperTriangular(A \ Bᵗᵃ⁰ᶜ)
        return ApplyArray(*, Rₐ₀ᵪᴬ, Rᵦₐ₋₁ᵪᵗᵃ⁰ᶜ)
    elseif A.b == B.b == -1
        Bᵗᵃ¹ᶜ = SemiclassicalJacobi(B.t, B.a, one(B.b), B.c, B)
        Aᵗᵃ¹ᶜ = SemiclassicalJacobi(A.t, A.a, one(A.b), A.c, A)
        Rₐ₁ᵪᵗᵘ¹ᵛ = Aᵗᵃ¹ᶜ \ Bᵗᵃ¹ᶜ
        # Make 1 ⊕ Rₐ₁ᵪᵗᵘ¹ᵛ
        V = eltype(Rₐ₁ᵪᵗᵘ¹ᵛ)
        Rₐ₋₁ᵪᵗᵘ⁻¹ᵛ = Vcat(
            Hcat(one(V), Zeros{V}(1, ∞)),
            Hcat(Zeros{V}(∞), Rₐ₁ᵪᵗᵘ¹ᵛ)
        )
        return Rₐ₋₁ᵪᵗᵘ⁻¹ᵛ
    else
        return semijacobi_ldiv(A, B)
    end
end
\(A::ConvertedOrthogonalPolynomial, B::SemiclassicalJacobi) = semijacobi_ldiv(A, B)
\(A::SemiclassicalJacobi, B::ConvertedOrthogonalPolynomial) = semijacobi_ldiv(A, B)
function \(w_A::WeightedSemiclassicalJacobi{T}, w_B::WeightedSemiclassicalJacobi{T}) where T
    wA,A = w_A.args
    wB,B = w_B.args
    @assert wA.t == wB.t == A.t == B.t
    Δa = B.a-A.a
    Δb = B.b-A.b
    Δc = B.c-A.c

    # k = (A \ SemiclassicalJacobiWeight(A.t,Δa,Δb,Δc))[1]
    if isone(-wA.b) && isone(-wB.b)
        @assert A.a + 1 == B.a && A.c + 1 == B.c
        Q = SemiclassicalJacobi(B.t, B.a, one(B.b), B.c, B)
        P = SemiclassicalJacobi(A.t, A.a, one(A.b), A.c, A)
        wP = Weighted(P)
        wQ = Weighted(Q)
        R22 = wP \ wQ
        α, β = _linear_coefficients(P.t, P.a, P.b, P.c)
        ℓ₁ = A.t - 1
        ℓ₂ = 1 + α - A.t
        ℓ₃ = inv(β)
        d0 = Vcat(ℓ₁, R22[band(0)])
        d1 = Vcat(ℓ₂, R22[band(-1)])
        d2 = Vcat(ℓ₃, R22[band(-2)])
        data = Hcat(d0, d1, d2)
        return _BandedMatrix(data', 1:∞, 2, 0)
    elseif (wA.a == A.a) && (wA.b == A.b) && (wA.c == A.c) && (wB.a == B.a) && (wB.b == B.b) && (wB.c == B.c) && isinteger(A.a) && isinteger(A.b) && isinteger(A.c) && isinteger(B.a) && isinteger(B.b) && isinteger(B.c)
            k = sumquotient(SemiclassicalJacobiWeight(B.t,B.a,B.b,B.c),SemiclassicalJacobiWeight(A.t,A.a,A.b,A.c))
            return (ApplyArray(*,Diagonal(Fill(k,∞)),(B \ A)))'
    elseif wA.a == wB.a && wA.b == wB.b && wA.c == wB.c # fallback to Christoffel–Darboux
        A \ B
    elseif wA.a+1 == wB.a && wA.b == wB.b && wA.c == wB.c
        @assert A.a+1 == B.a && A.b == B.b && A.c == B.c
        op_lowering(A,0)
    elseif wA.a == wB.a && wA.b+1 == wB.b && wA.c == wB.c
        @assert A.a == B.a && A.b+1 == B.b && A.c == B.c
        -op_lowering(A,1)
    elseif wA.a == wB.a && wA.b == wB.b && wA.c+1 == wB.c
        @assert A.a == B.a && A.b == B.b && A.c+1 == B.c
        -op_lowering(A,A.t)
    elseif wA.a+1 ≤ wB.a
        C = SemiclassicalJacobi(A.t, A.a+1, A.b, A.c, A)
        w_C = SemiclassicalJacobiWeight(wA.t, wA.a+1, wA.b, wA.c) .* C
        L_2 = w_C \ w_B
        L_1 = w_A \ w_C
        L_1 * L_2
    elseif wA.b+1 ≤ wB.b
        C = SemiclassicalJacobi(A.t, A.a, A.b+1, A.c, A)
        w_C = SemiclassicalJacobiWeight(wA.t, wA.a, wA.b+1, wA.c) .* C
        L_2 = w_C \ w_B
        L_1 = w_A \ w_C
        L_1 * L_2
    else
        error("Implement")
    end
end

\(P::SemiclassicalJacobi, Q::Weighted{<:Any,<:SemiclassicalJacobi}) = P \ convert(WeightedBasis, Q)

\(A::SemiclassicalJacobi, w_B::WeightedSemiclassicalJacobi) = (SemiclassicalJacobiWeight(A.t,0,0,0) .* A) \ w_B

\(w_A::Weighted{<:Any,<:SemiclassicalJacobi}, w_B::Weighted{<:Any,<:SemiclassicalJacobi}) = convert(WeightedBasis, w_A) \ convert(WeightedBasis, w_B)

function \(w_A::HalfWeighted{lr, T, <:SemiclassicalJacobi}, B::AbstractQuasiArray{V}) where {lr, T, V}
    WP = convert(WeightedBasis, w_A)
    w_A.P.b  ≠ -1 && return WP \ B # no need to special case here
    !iszero(WP.args[1].b) && throw(ArgumentError("Cannot expand in a weighted basis including 1/(1-x)."))
    # To expand f(x) = w(x)P(x)𝐟, note that P = [1 (1-x)Q] so
    #   f(x) = w(x)[1 (1-x)Q(x)][f₀; 𝐟₁] = w(x)f₀ + w(x)(1-x)Q(x)𝐟₁. Thus,
    #   f(1) = w(1)f₀ ⟹ f₀ = f(1) / w(1)
    #   Then, f(x) - w(x)f₀ = w(x)(1-x)Q(x)𝐟₁, so that 𝐟₁ is just the expansion of
    #   f(x) - w(x)f₀ in the w(x)(1-x)Q(x) basis.
    w, P = WP.args
    f₀ = B[end] / w[end]
    C = B - w * f₀
    Q = SemiclassicalJacobiWeight(w.t, w.a, one(w.b), w.c) .* SemiclassicalJacobi(P.t, P.a, one(P.b), P.c, P)
    f = Q \ C
    return Vcat(f₀, f)
end

weightedgrammatrix(P::SemiclassicalJacobi) = Diagonal(Fill(sum(orthogonalityweight(P)),∞))

@simplify function *(Ac::QuasiAdjoint{<:Any,<:SemiclassicalJacobi}, wB::WeightedBasis{<:Any,<:SemiclassicalJacobiWeight,<:SemiclassicalJacobi})
    A = parent(Ac)
    w,B = arguments(wB)
    P = SemiclassicalJacobi(w.t, w.a, w.b, w.c)
    (P\A)' * weightedgrammatrix(P) * (P \ B)
end

function ldiv(Q::SemiclassicalJacobi, f::AbstractQuasiVector)
    T = typeof(Q.t)
    if iszero(Q.a) && iszero(Q.b) && isone(-Q.c) # (0,0,-1) special case
        R = legendre(zero(T)..one(T))
        B = neg1c_tolegendre(Q.t)
        return (B \ (R \ f))
    elseif isinteger(Q.a) && (isinteger(Q.b) && Q.b ≥ 0) && isinteger(Q.c) # (a,b,c) are integers -> use QR/Cholesky
        R̃ = Normalized(jacobi(Q.b, Q.a, UnitInterval{T}()))
        return (Q \ SemiclassicalJacobi(Q.t, Q.a, Q.b, 0)) *  _p0(R̃) * (R̃ \ f)
    elseif isinteger(Q.a) && isone(-Q.b) && isinteger(Q.c)
        return semijacobi_ldiv(Q, f) # jacobi(< 0, Q.a) fails in the method above. jacobi(-1, 0) also leads to NaNs in coefficients
    else # fallback to Lanzcos
        R̃ = toclassical(SemiclassicalJacobi(Q.t, mod(Q.a,-1), mod(Q.b,-1), mod(Q.c,-1)))
        return (Q \ R̃) * (R̃ \ f)
    end
end
function ldiv(Qn::SubQuasiArray{<:Any,2,<:SemiclassicalJacobi,<:Tuple{<:Inclusion,<:Any}}, C::AbstractQuasiArray)
    _,jr = parentindices(Qn)
    Q = parent(Qn)
    if iszero(Q.a) && iszero(Q.b) && Q.c == -one(eltype(Q.t))
        T = typeof(Q.t)
        R = legendre(zero(T)..one(T))
        B = neg1c_tolegendre(Q.t)
        return (B[jr,jr] \ (R[:,jr] \ C))
    end

    R̃ = toclassical(SemiclassicalJacobi(Q.t, mod(Q.a,-1), mod(Q.b,-1), mod(Q.c,-1)))
    (Q \ R̃)[jr,jr] * (R̃[:,jr] \ C)
end

# sqrt(1-(1-x)^2) == sqrt(2x-x^2) == sqrt(x)*sqrt(2-x)
# sqrt(1-(1-x)^2) == sqrt(2x-x^2) == sqrt(x)*sqrt(2-x)

weight(W::HalfWeighted{:a,T,<:SemiclassicalJacobi}) where T = SemiclassicalJacobiWeight(W.P.t, W.P.a,zero(T),zero(T))
weight(W::HalfWeighted{:b,T,<:SemiclassicalJacobi}) where T = SemiclassicalJacobiWeight(W.P.t, zero(T),W.P.b,zero(T))
weight(W::HalfWeighted{:c,T,<:SemiclassicalJacobi}) where T = SemiclassicalJacobiWeight(W.P.t, zero(T),zero(T),W.P.c)
weight(W::HalfWeighted{:ab,T,<:SemiclassicalJacobi}) where T = SemiclassicalJacobiWeight(W.P.t, W.P.a,W.P.b,zero(T))
weight(W::HalfWeighted{:ac,T,<:SemiclassicalJacobi}) where T = SemiclassicalJacobiWeight(W.P.t, W.P.a,zero(T),W.P.c)
weight(W::HalfWeighted{:bc,T,<:SemiclassicalJacobi}) where T = SemiclassicalJacobiWeight(W.P.t, zero(T),W.P.b,W.P.c)

convert(::Type{WeightedBasis}, Q::HalfWeighted{:a,T,<:SemiclassicalJacobi}) where T = SemiclassicalJacobiWeight(Q.P.t, Q.P.a,zero(T),zero(T)) .* Q.P
convert(::Type{WeightedBasis}, Q::HalfWeighted{:b,T,<:SemiclassicalJacobi}) where T = SemiclassicalJacobiWeight(Q.P.t, zero(T),Q.P.b,zero(T)) .* Q.P
convert(::Type{WeightedBasis}, Q::HalfWeighted{:c,T,<:SemiclassicalJacobi}) where T = SemiclassicalJacobiWeight(Q.P.t, zero(T),zero(T),Q.P.c) .* Q.P

convert(::Type{WeightedBasis}, Q::HalfWeighted{:ab,T,<:SemiclassicalJacobi}) where T = SemiclassicalJacobiWeight(Q.P.t, Q.P.a,Q.P.b,zero(T)) .* Q.P
convert(::Type{WeightedBasis}, Q::HalfWeighted{:bc,T,<:SemiclassicalJacobi}) where T = SemiclassicalJacobiWeight(Q.P.t, zero(T),Q.P.b,Q.P.c) .* Q.P
convert(::Type{WeightedBasis}, Q::HalfWeighted{:ac,T,<:SemiclassicalJacobi}) where T = SemiclassicalJacobiWeight(Q.P.t, Q.P.a,zero(T),Q.P.c) .* Q.P

include("derivatives.jl")
include("family.jl")
include("neg1c.jl")
include("deprecated.jl")

end
