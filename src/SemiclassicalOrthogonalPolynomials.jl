module SemiclassicalOrthogonalPolynomials
using ClassicalOrthogonalPolynomials, FillArrays, LazyArrays, ArrayLayouts, QuasiArrays, InfiniteArrays, ContinuumArrays, LinearAlgebra, BandedMatrices, 
        SpecialFunctions, HypergeometricFunctions

import Base: getindex, axes, size, \, /, *, +, -, summary, ==, copy, sum, unsafe_getindex

import ArrayLayouts: MemoryLayout, ldiv, diagonaldata, subdiagonaldata, supdiagonaldata
import BandedMatrices: bandwidths, AbstractBandedMatrix, BandedLayout, _BandedMatrix
import LazyArrays: resizedata!, paddeddata, CachedVector, CachedMatrix, CachedAbstractVector, LazyMatrix, LazyVector, arguments, ApplyLayout, colsupport, AbstractCachedVector, AccumulateAbstractVector
import ClassicalOrthogonalPolynomials: OrthogonalPolynomial, recurrencecoefficients, jacobimatrix, normalize, _p0, UnitInterval, orthogonalityweight, NormalizedBasisLayout,
                                        Bidiagonal, Tridiagonal, SymTridiagonal, symtridiagonalize, normalizationconstant,
                                        OrthogonalPolynomialRatio, normalized_recurrencecoefficients
import InfiniteArrays: OneToInf, InfUnitRange
import ContinuumArrays: basis, Weight, @simplify, AbstractBasisLayout, BasisLayout, MappedBasisLayout
import FillArrays: SquareEye

export LanczosPolynomial, Legendre, Normalized, normalize, SemiclassicalJacobi, SemiclassicalJacobiWeight, WeightedSemiclassicalJacobi, OrthogonalPolynomialRatio



include("ratios.jl")
include("negativeintc.jl")


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

function sum(P::SemiclassicalJacobiWeight)
    t,a,b,c = P.t,P.a,P.b,P.c
    t^c * gamma(1+a)gamma(1+b)/gamma(2+a+b) * _₂F₁(1+a,-c,2+a+b,1/t)
end

function summary(io::IO, P::SemiclassicalJacobiWeight)
    t,a,b,c = P.t,P.a,P.b,P.c
    print(io, "x^$a * (1-x)^$b * ($t-x)^$c")
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
    P # We need to store the basic case where ã,b̃,c̃ = mod(a,-1),mod(b,-1),mod(c,-1)
          # in order to compute lowering operators, etc.
    SemiclassicalJacobi{T}(t::T,a::T,b::T,c::T,P) where T = new{T}(t,a,b,c,P)
end

const WeightedSemiclassicalJacobi{T} = WeightedBasis{T,<:SemiclassicalJacobiWeight,<:SemiclassicalJacobi}

function SemiclassicalJacobi(t, a, b, c, P::OrthogonalPolynomial)
    T = float(promote_type(typeof(t), typeof(a), typeof(b), typeof(c), eltype(P)))
    SemiclassicalJacobi{T}(T(t), T(a), T(b), T(c), P)
end

SemiclassicalJacobi(t, a, b, c, P::SemiclassicalJacobi) = SemiclassicalJacobi(t, a, b, c, P.P)

WeightedSemiclassicalJacobi(t, a, b, c, P...) = SemiclassicalJacobiWeight(t, a, b, c) .* SemiclassicalJacobi(t, a, b, c, P...)


function SemiclassicalJacobi(t, a, b, c)
    ã,b̃,c̃ = mod(a,-1),mod(b,-1),min(c, mod(c,-1))
    T = promote_type(typeof(t), typeof(a), typeof(b), typeof(c))
    P = jacobi(b̃, ã, UnitInterval{T}())
    x = axes(P,1)
    SemiclassicalJacobi(t, a, b, c, LanczosPolynomial(@.(x^ã * (1-x)^b̃ * (t-x)^c̃), P))
end


function SemiclassicalJacobi(t, a::Int, b::Int, c::Int)
    T = promote_type(typeof(t), typeof(a), typeof(b), typeof(c))
    P = Normalized(legendre(UnitInterval{T}()))
    x = axes(P,1)
    SemiclassicalJacobi(t, a, b, c, P)
end

copy(P::SemiclassicalJacobi) = P
axes(P::SemiclassicalJacobi) = axes(P.P)

==(A::SemiclassicalJacobi, B::SemiclassicalJacobi) = A.t == B.t && A.a == B.a && A.b == B.b && A.c == B.c
==(::AbstractQuasiMatrix, ::SemiclassicalJacobi) = false
==(::SemiclassicalJacobi, ::AbstractQuasiMatrix) = false

orthogonalityweight(P::SemiclassicalJacobi) = SemiclassicalJacobiWeight(P.t, P.a, P.b, P.c)

function summary(io::IO, P::SemiclassicalJacobi)
    t,a,b,c = P.t,P.a,P.b,P.c
    print(io, "SemiclassicalJacobi with weight x^$a * (1-x)^$b * ($t-x)^$c")
end

"""
   RaisedOP(P, y)

gives the OPs w.r.t. (y - x) .* w based on lowering to Q.
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
   cache_abstract(A)

caches a `A` without storing the type of `A` as a template.
This prevents exceessive compilation.
"""
cache_abstract(v::AbstractVector{T}) where T = CachedAbstractVector(v)
cache_abstract(S::SymTridiagonal) = SymTridiagonal(cache_abstract(S.dv), cache_abstract(S.ev))

function jacobimatrix(P::SemiclassicalJacobi{T}) where T
    if P.a ≤ 0 && P.b ≤ 0 && P.c ≤ 0
        jacobimatrix(P.P)
    elseif P.a > 0
        Q = SemiclassicalJacobi(P.t, P.a-1, P.b, P.c, P.P)
        cache_abstract(symtridiagonalize(jacobimatrix(RaisedOP(Q, 0))))
    elseif P.b > 0
        Q = SemiclassicalJacobi(P.t, P.a, P.b-1, P.c, P.P)
        cache_abstract(symtridiagonalize(jacobimatrix(RaisedOP(Q, 1))))
    elseif P.c > 0
        Q = SemiclassicalJacobi(P.t, P.a, P.b, P.c-1, P.P)
        cache_abstract(symtridiagonalize(jacobimatrix(RaisedOP(Q, P.t))))
    else
        error("Implement")
    end
end

recurrencecoefficients(P::SemiclassicalJacobi) = normalized_recurrencecoefficients(P)

"""
    op_lowering(Q, y)
Gives the Lowering operator from OPs w.r.t. (x-y)*w(x) to Q
as constructed from Chistoffel–Darboux
"""
function op_lowering(Q, y)
    # we first use Christoff-Darboux with d = 1
    # But we want the first OP to be 1 so we rescale
    P = RaisedOP(Q, y)
    A,_,_ = recurrencecoefficients(Q)
    d = -inv(A[1]*_p0(Q)*P.ℓ[1])
    κ = d * normalizationconstant(1, P)
    T = eltype(κ)
    # hide array type for compilation
    Bidiagonal(κ, -(κ .* P.ℓ), :L)
end

function semijacobi_ldiv(Q, P::SemiclassicalJacobi)
    if P.a ≤ 0 && P.b ≤ 0 && P.c ≤ 0
        (Q \ P.P)/_p0(P.P)
    else
        error("Implement")
    end
end

function semijacobi_ldiv(P::SemiclassicalJacobi, Q)
    R = SemiclassicalJacobi(P.t, mod(P.a,-1), mod(P.b,-1), mod(P.c,-1), P)
    (P \ R) * _p0(P.P) * (P.P \ Q)
end

struct SemiclassicalJacobiLayout <: AbstractBasisLayout end
MemoryLayout(::Type{<:SemiclassicalJacobi}) = SemiclassicalJacobiLayout()

copy(L::Ldiv{<:NormalizedBasisLayout,SemiclassicalJacobiLayout}) = copy(Ldiv{ApplyLayout{typeof(*)},SemiclassicalJacobiLayout}(L.A, L.B))
copy(L::Ldiv{SemiclassicalJacobiLayout,<:NormalizedBasisLayout}) = copy(Ldiv{SemiclassicalJacobiLayout,ApplyLayout{typeof(*)}}(L.A, L.B))

copy(L::Ldiv{ApplyLayout{typeof(*)},SemiclassicalJacobiLayout}) = copy(Ldiv{ApplyLayout{typeof(*)},BasisLayout}(L.A, L.B))
copy(L::Ldiv{SemiclassicalJacobiLayout,ApplyLayout{typeof(*)}}) = copy(Ldiv{BasisLayout,ApplyLayout{typeof(*)}}(L.A, L.B))


copy(L::Ldiv{MappedBasisLayout,SemiclassicalJacobiLayout}) = semijacobi_ldiv(L.A, L.B)
copy(L::Ldiv{SemiclassicalJacobiLayout,MappedBasisLayout}) = semijacobi_ldiv(L.A, L.B)


copy(L::Ldiv{SemiclassicalJacobiLayout}) = semijacobi_ldiv(L.A, L.B)
copy(L::Ldiv{<:Any,SemiclassicalJacobiLayout}) = semijacobi_ldiv(L.A, L.B)
function copy(L::Ldiv{SemiclassicalJacobiLayout,SemiclassicalJacobiLayout})
    Q,P = L.A,L.B
    @assert Q.t == P.t
    Q == P && return SquareEye{eltype(L)}((axes(P,2),))
    M_Q = massmatrix(Q)
    M_P = massmatrix(P)
    L = P \ (SemiclassicalJacobiWeight(Q.t, Q.a-P.a, Q.b-P.b, Q.c-P.c) .* Q)
    inv(M_Q) * L' * M_P
end

\(A::LanczosPolynomial, B::SemiclassicalJacobi) = semijacobi_ldiv(A, B)
\(A::SemiclassicalJacobi, B::LanczosPolynomial) = semijacobi_ldiv(A, B)
function \(w_A::WeightedSemiclassicalJacobi, w_B::WeightedSemiclassicalJacobi)
    wA,A = w_A.args
    wB,B = w_B.args
    @assert wA.t == wB.t == A.t == B.t

    if wA.a == wB.a && wA.b == wB.b && wA.c == wB.c
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
    elseif wA.a+1 ≤ wB.a
        C = SemiclassicalJacobi(B.t, B.a-1, B.b, B.c, B)
        w_C = SemiclassicalJacobiWeight(B.t, wB.a-1, wB.b, wB.c) .* C
        L_2 = Base.inferencebarrier(w_C) \ Base.inferencebarrier(w_B)
        L_1 = Base.inferencebarrier(w_A) \ Base.inferencebarrier(w_C)
        L_1 * L_2
    elseif wA.b+1 ≤ wB.b
        C = SemiclassicalJacobi(B.t, B.a, B.b-1, B.c, B)
        w_C = SemiclassicalJacobiWeight(B.t, wB.a, wB.b-1, wB.c) .* C
        L_2 = Base.inferencebarrier(w_C) \ Base.inferencebarrier(w_B)
        L_1 = Base.inferencebarrier(w_A) \ Base.inferencebarrier(w_C)
        L_1 * L_2
    else
        error("Implement")
    end
end

\(A::SemiclassicalJacobi, w_B::WeightedSemiclassicalJacobi) = (SemiclassicalJacobiWeight(A.t,0,0,0) .* A) \ w_B


massmatrix(P::SemiclassicalJacobi) = Diagonal(Fill(sum(orthogonalityweight(P)),∞))

@simplify function *(Ac::QuasiAdjoint{<:Any,<:SemiclassicalJacobi}, wB::WeightedBasis{<:Any,<:SemiclassicalJacobiWeight,<:SemiclassicalJacobi})
    A = parent(Ac)
    w,B = arguments(wB)
    P = SemiclassicalJacobi(w.t, w.a, w.b, w.c, B.P)
    (P\A)' * massmatrix(P) * (P \ B)
end


ldiv(Q::SemiclassicalJacobi, f::AbstractQuasiVector) = (Q \ Q.P) * (Q.P \ f)
function ldiv(Qn::SubQuasiArray{<:Any,2,<:SemiclassicalJacobi,<:Tuple{<:Inclusion,<:Any}}, C::AbstractQuasiArray)
    _,jr = parentindices(Qn)
    Q = parent(Qn)
    (Q \ Q.P)[jr,jr] * (Q.P[:,jr] \ C)
end

# sqrt(1-(1-x)^2) == sqrt(2x-x^2) == sqrt(x)*sqrt(2-x)
# sqrt(1-(1-x)^2) == sqrt(2x-x^2) == sqrt(x)*sqrt(2-x)

include("derivatives.jl")


end