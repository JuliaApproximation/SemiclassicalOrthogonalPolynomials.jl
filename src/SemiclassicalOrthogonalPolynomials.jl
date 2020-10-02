module SemiclassicalOrthogonalPolynomials
using OrthogonalPolynomialsQuasi, FillArrays, LazyArrays, ArrayLayouts, QuasiArrays, InfiniteArrays, ContinuumArrays, LinearAlgebra, BandedMatrices

import Base: getindex, axes, size, \, /, *, +, -, summary, ==

import ArrayLayouts: MemoryLayout
import BandedMatrices: bandwidths, _BandedMatrix
import LazyArrays: resizedata!, paddeddata, CachedVector, CachedMatrix, LazyMatrix, LazyVector, arguments, ApplyLayout, colsupport
import OrthogonalPolynomialsQuasi: OrthogonalPolynomial, recurrencecoefficients, jacobimatrix, normalize, recurrencecoefficients, _p0, UnitInterval, bands
import InfiniteArrays: OneToInf, InfUnitRange
import ContinuumArrays: basis, Weight

export LanczosPolynomial, Legendre, Normalized, normalize, SemiclassicalJacobi, SemiclassicalJacobiWeight

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

axes(P::SemiclassicalJacobiWeight{T}) where T = (Inclusion(UnitInterval{T}()),)
function getindex(P::SemiclassicalJacobiWeight, x::Real)
    t,a,b,c = P.t,P.a,P.b,P.c
    @boundscheck checkbounds(P, x)
    (1-x)^a * x^b * (t-x)^c
end

# orthogonal w.r.t. (1-x)^a * x^b * (t-x)^c on [0,1]
# We need to store the basic case where ã,b̃,c̃ = mod(a,-1),mod(b,-1),mod(c,-1)
# in order to compute lowering operators, etc.
struct SemiclassicalJacobi{T,PP<:LanczosPolynomial} <: OrthogonalPolynomial{T}
    t::T
    a::T
    b::T
    c::T
    P::PP
end

const WeightedSemiclassicalJacobi{T} = WeightedBasis{T,<:SemiclassicalJacobiWeight,<:SemiclassicalJacobi}

function SemiclassicalJacobi(t, a, b, c, P::LanczosPolynomial)
    T = promote_type(typeof(t), typeof(a), typeof(b), typeof(c), eltype(P))
    SemiclassicalJacobi(T(t), T(a), T(b), T(c), P)
end

SemiclassicalJacobi(t, a, b, c, P::SemiclassicalJacobi) = SemiclassicalJacobi(t, a, b, c, P.P)


function SemiclassicalJacobi(t, a, b, c)
    ã,b̃,c̃ = mod(a,-1),mod(b,-1),mod(c,-1)
    P = jacobi(ã, b̃, UnitInterval())
    x = axes(P,1)
    SemiclassicalJacobi(t, a, b, c, LanczosPolynomial(@.((1-x)^ã * x^b̃ * (t-x)^c̃), P))
end


axes(P::SemiclassicalJacobi) = axes(P.P)

==(A::SemiclassicalJacobi, B::SemiclassicalJacobi) = A.t == B.t && A.a == B.a && A.b == B.b && A.c == B.c
==(::AbstractQuasiMatrix, ::SemiclassicalJacobi) = false
==(::SemiclassicalJacobi, ::AbstractQuasiMatrix) = false


function summary(io::IO, P::SemiclassicalJacobi)
    t,a,b,c = P.t,P.a,P.b,P.c
    print(io, "SemiclassicalJacobi with weight (1-x)^$a*x^$b*($t-x)^$c")
end

function recurrencecoefficients(P::SemiclassicalJacobi)
    if P.a ≤ 0 && P.b ≤ 0 && P.c ≤ 0
        recurrencecoefficients(P.P)
    else
        error("Implement")
    end
end
function jacobimatrix(P::SemiclassicalJacobi)
    if P.a ≤ 0 && P.b ≤ 0 && P.c ≤ 0
        jacobimatrix(P.P)
    else
        error("Implement")
    end
end
"""
    normalized_op_lowering(Q, y)
Gives the Lowering operator from OPs w.r.t. |y-x|*w(x) to Q
as constructed from Chistoffel–Darboux
"""
function normalized_op_lowering(Q, y)
    X = jacobimatrix(Q)
    _,_,b = bands(X)
    _BandedMatrix(Vcat((-b .* Q[y,2:end])', (b .* Q[y,1:end])'), ∞, 1, 0)
end

function semijacobi_ldiv(Q, P::SemiclassicalJacobi)
    if P.a ≤ 0 && P.b ≤ 0 && P.c ≤ 0
        (Q \ P.P)/_p0(P.P)
    else
        error("Implement")
    end
end

function semijacobi_ldiv(P::SemiclassicalJacobi, Q)
    if P.a ≤ 0 && P.b ≤ 0 && P.c ≤ 0
        _p0(P.P)*(P.P \ Q)
    else
        error("Implement")
    end
end

\(Q::OrthogonalPolynomial, P::SemiclassicalJacobi) = semijacobi_ldiv(Q, P)
\(Q::LanczosPolynomial, P::SemiclassicalJacobi) = semijacobi_ldiv(Q, P)
\(Q::SemiclassicalJacobi, P::OrthogonalPolynomial) = semijacobi_ldiv(Q, P)
\(Q::SemiclassicalJacobi, P::LanczosPolynomial) = semijacobi_ldiv(Q, P)

function \(w_A::WeightedSemiclassicalJacobi, w_B::WeightedSemiclassicalJacobi)
    wA,A = w_A.args
    wB,B = w_B.args
    @assert wA.t == wB.t == A.t == B.t

    if wA.a == wB.a && wA.b == wB.b && wA.c == wB.c
        A \ B
    elseif wA.a+1 == wB.a && wA.b == wB.b && wA.c == wB.c
        @assert A.a+1 == B.a && A.b == B.b && A.c == B.c
        normalized_op_lowering(A,1)
    elseif wA.a == wB.a && wA.b+1 == wB.b && wA.c == wB.c
        @assert A.a == B.a && A.b+1 == B.b && A.c == B.c
        normalized_op_lowering(A,0)
    elseif wA.a == wB.a && wA.b == wB.b && wA.c+1 == wB.c
        @assert A.a == B.a && A.b == B.b && A.c+1 == B.c
        normalized_op_lowering(A,A.t)
    else
        error("Implement")
    end
end

\(A::SemiclassicalJacobi, w_B::WeightedSemiclassicalJacobi) = (SemiclassicalJacobiWeight(A.t,0,0,0) .* A) \ w_B

# sqrt(1-(1-x)^2) == sqrt(2x-x^2) == sqrt(x)*sqrt(2-x)
# sqrt(1-(1-x)^2) == sqrt(2x-x^2) == sqrt(x)*sqrt(2-x)


end