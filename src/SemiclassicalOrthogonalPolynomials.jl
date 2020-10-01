module SemiclassicalOrthogonalPolynomials
using OrthogonalPolynomialsQuasi, FillArrays, LazyArrays, ArrayLayouts, QuasiArrays, InfiniteArrays, ContinuumArrays, LinearAlgebra, BandedMatrices

import Base: getindex, axes, size, \, /, *, +, -, summary, ==

import ArrayLayouts: MemoryLayout
import BandedMatrices: bandwidths, _BandedMatrix
import LazyArrays: resizedata!, paddeddata, CachedVector, CachedMatrix, LazyMatrix, LazyVector, arguments, ApplyLayout, colsupport
import OrthogonalPolynomialsQuasi: OrthogonalPolynomial, recurrencecoefficients, jacobimatrix, normalize, recurrencecoefficients, _p0, UnitInterval
import InfiniteArrays: OneToInf, InfUnitRange
import ContinuumArrays: basis

export LanczosPolynomial, Legendre, Normalized, normalize, SemiclassicalJacobi, SemiclassicalJacobiWeight

struct SemiclassicalJacobiWeight{T} <: Weight{T}
    t::T
    a::T
    b::T
    c::T
end

axes(P::SemiclassicalJacobiWeight{T}) where T = (Inclusion(UnitInterval{T}()),)
function getindex(P::SemiclassicalJacobiWeight, x::Real)
    t,a,b,c = P.t,P.a,P.b,P.c
    checkbounds(P, x)
    (1-x)^a * x^b * (t-x)^c
end

# orthogonal w.r.t. (1-x)^a * x^b * (t-x)^c on [0,1]
# represented as P * L. If a,b,c ≤ 0 then L == Eye
struct SemiclassicalJacobi{T,PP} <: OrthogonalPolynomial{T}
    t::T
    a::T
    b::T
    c::T
    P::PP
end

function SemiclassicalJacobi(t, a, b, c)
    T = promote_type(eltype(t), eltype(a), eltype(b), eltype(c))
    ã,b̃,c̃ = mod(a,-1),mod(b,-1),mod(c,-1)
    P = jacobi(ã, b̃, UnitInterval())
    x = axes(P,1)
    SemiclassicalJacobi(T(t), T(a), T(b), T(c), LanczosPolynomial(@.((1-x)^ã * x^b̃ * (t-x)^c̃), P))
end


axes(P::SemiclassicalJacobi) = axes(P.P)

==(A::SemiclassicalJacobi, B::SemiclassicalJacobi) = A.t == B.t && A.a == B.a && A.b == B.b && A.c == B.c

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

function semijacobi_lower(P::SemiclassicalJacobi, y)
    X = jacobimatrix(P.P)
    _BandedMatrix(Vcat((-X.ev .* P.P[y,2:end])', (X.ev .* P.P[y,1:end])'), ∞, 1, 0)
end

function semijacobi_ldiv(Q, P::SemiclassicalJacobi)
    if P.a ≤ 0 && P.b ≤ 0 && P.c ≤ 0
        (Q \ P.P)/_p0(P.P)
    elseif P.a ≤ 0 && P.c ≤ 0
        L = semijacobi_lower(P, 0)
        (Q \ P.P) * L
    else
        error("Implement")
    end
end

\(Q::OrthogonalPolynomial, P::SemiclassicalJacobi) = semijacobi_ldiv(Q, P)
\(Q::LanczosPolynomial, P::SemiclassicalJacobi) = semijacobi_ldiv(Q, P)

# sqrt(1-(1-x)^2) == sqrt(2x-x^2) == sqrt(x)*sqrt(2-x)
# sqrt(1-(1-x)^2) == sqrt(2x-x^2) == sqrt(x)*sqrt(2-x)


end