module SemiclassicalOrthogonalPolynomials
using OrthogonalPolynomialsQuasi, FillArrays, LazyArrays, ArrayLayouts, QuasiArrays, InfiniteArrays, ContinuumArrays, LinearAlgebra, BandedMatrices, 
        SpecialFunctions, HypergeometricFunctions

import Base: getindex, axes, size, \, /, *, +, -, summary, ==, copy, sum, unsafe_getindex

import ArrayLayouts: MemoryLayout
import BandedMatrices: bandwidths, _BandedMatrix, AbstractBandedMatrix, BandedLayout
import LazyArrays: resizedata!, paddeddata, CachedVector, CachedMatrix, LazyMatrix, LazyVector, arguments, ApplyLayout, colsupport
import OrthogonalPolynomialsQuasi: OrthogonalPolynomial, recurrencecoefficients, jacobimatrix, normalize, recurrencecoefficients, _p0, UnitInterval, orthogonalityweight
import InfiniteArrays: OneToInf, InfUnitRange
import ContinuumArrays: basis, Weight, @simplify
import FillArrays: SquareEye

export LanczosPolynomial, Legendre, Normalized, normalize, SemiclassicalJacobi, SemiclassicalJacobiWeight, ConjugateJacobiMatrix

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

function sum(P::SemiclassicalJacobiWeight)
    t,a,b,c = P.t,P.a,P.b,P.c
    t^c * gamma(1+a)gamma(1+b)/gamma(2+a+b) * _₂F₁(1+b,-c,2+a+b,1/t)
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

orthogonalityweight(P::SemiclassicalJacobi) = SemiclassicalJacobiWeight(P.t, P.a, P.b, P.c)

function summary(io::IO, P::SemiclassicalJacobi)
    t,a,b,c = P.t,P.a,P.b,P.c
    print(io, "SemiclassicalJacobi with weight (1-x)^$a*x^$b*($t-x)^$c")
end

# Jacobi matrix formed as inv(L)*X*L
struct ConjugateJacobiMatrix{T,XX<:AbstractMatrix{T},LL<:AbstractMatrix{T}} <: AbstractBandedMatrix{T}
    X::XX
    L::LL
end

MemoryLayout(::Type{<:ConjugateJacobiMatrix}) = BandedLayout()
bandwidths(::ConjugateJacobiMatrix) = (1,1)
size(::ConjugateJacobiMatrix) = (∞,∞)

copy(A::ConjugateJacobiMatrix) = A # immutable entries

function getindex(J::ConjugateJacobiMatrix{T}, k::Int, j::Int) where T
    X,L = J.X,J.L
    @boundscheck checkbounds(X, k, j)
    abs(k-j) ≤ 1 || return zero(T)
    if j == 1
        kr = 1:j+1
        col = L[kr,kr] \ X[kr,j:j+1]*L[j:j+1,j]
        col[k-j+1]
    else
        kr = j-1:j+1
        col = L[kr,kr] \ X[kr,j:j+1]*L[j:j+1,j]
        col[k-j+2]
    end
end

struct JacobiMatrix2Recurrence{k,T,XX<:AbstractMatrix{T}} <: LazyVector{T}
    X::XX
end

JacobiMatrix2Recurrence{k}(X) where k = JacobiMatrix2Recurrence{k,eltype(X),typeof(X)}(X)
size(::JacobiMatrix2Recurrence) = (∞,)
getindex(A::JacobiMatrix2Recurrence{:A}, k::Int) = 1/A.X[k+1,k]
getindex(A::JacobiMatrix2Recurrence{:B}, k::Int) = -A.X[k,k]/A.X[k+1,k]
getindex(A::JacobiMatrix2Recurrence{:C}, k::Int) = A.X[k-1,k]/A.X[k+1,k]

summary(io::IO, P::ConjugateJacobiMatrix{T}) where T = print(io, "ConjugateJacobiMatrix{$T}")
summary(io::IO, P::JacobiMatrix2Recurrence{k,T}) where {k,T} = print(io, "JacobiMatrix2Recurrence{$k,$T}")


function recurrencecoefficients(P::SemiclassicalJacobi)
    if P.a ≤ 0 && P.b ≤ 0 && P.c ≤ 0
        recurrencecoefficients(P.P)
    else
        X = jacobimatrix(P)
        JacobiMatrix2Recurrence{:A}(X),JacobiMatrix2Recurrence{:B}(X),JacobiMatrix2Recurrence{:C}(X)
    end
end
function jacobimatrix(P::SemiclassicalJacobi)
    if P.a ≤ 0 && P.b ≤ 0 && P.c ≤ 0
        jacobimatrix(P.P)
    elseif P.b > 0
        Q = SemiclassicalJacobi(P.t, P.a, P.b-1, P.c, P.P)
        L = Q \ (SemiclassicalJacobiWeight(P.t,0,1,0) .* P)
        X = jacobimatrix(Q)
        ConjugateJacobiMatrix(X,L)
    elseif P.c > 0
        Q = SemiclassicalJacobi(P.t, P.a, P.b, P.c-1, P.P)
        L = Q \ (SemiclassicalJacobiWeight(P.t,0,0,1) .* P)
        X = jacobimatrix(Q)
        ConjugateJacobiMatrix(X,L)
    else
        error("Implement")
    end
end
"""
    normalized_op_lowering(Q, y)
Gives the Lowering operator from OPs w.r.t. (x-y)*w(x) to Q
as constructed from Chistoffel–Darboux
"""
function normalized_op_lowering(Q, y)
    X = jacobimatrix(Q)
    b = X[band(1)]
    _BandedMatrix(Vcat((-b .* unsafe_getindex(Q,y,2:∞))', (b .* unsafe_getindex(Q,y,1:∞))'), ∞, 1, 0)
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
function \(Q::SemiclassicalJacobi{T}, P::SemiclassicalJacobi{V}) where {T,V}
    @assert Q.t == P.t
    Q == P && return SquareEye{promote_type(T,V)}(∞)
    M_Q = semiclassicaljacobi_massmatrix(Q)
    M_P = semiclassicaljacobi_massmatrix(P)
    L = P \ (SemiclassicalJacobiWeight(Q.t, Q.a-P.a, Q.b-P.b, Q.c-P.c) .* Q)
    inv(M_Q) * L' * M_P
end

function \(w_A::WeightedSemiclassicalJacobi, w_B::WeightedSemiclassicalJacobi)
    wA,A = w_A.args
    wB,B = w_B.args
    @assert wA.t == wB.t == A.t == B.t

    if wA.a == wB.a && wA.b == wB.b && wA.c == wB.c
        A \ B
    elseif wA.a+1 == wB.a && wA.b == wB.b && wA.c == wB.c
        @assert A.a+1 == B.a && A.b == B.b && A.c == B.c
        -normalized_op_lowering(A,1)
    elseif wA.a == wB.a && wA.b+1 == wB.b && wA.c == wB.c
        @assert A.a == B.a && A.b+1 == B.b && A.c == B.c
        normalized_op_lowering(A,0)
    elseif wA.a == wB.a && wA.b == wB.b && wA.c+1 == wB.c
        @assert A.a == B.a && A.b == B.b && A.c+1 == B.c
        -normalized_op_lowering(A,A.t)
    else
        error("Implement")
    end
end

\(A::SemiclassicalJacobi, w_B::WeightedSemiclassicalJacobi) = (SemiclassicalJacobiWeight(A.t,0,0,0) .* A) \ w_B

semiclassicaljacobi_massmatrix(P::SemiclassicalJacobi) = Diagonal(Normalized(P).scaling .^ (-2))

@simplify function *(Ac::QuasiAdjoint{<:Any,<:SemiclassicalJacobi}, wB::WeightedBasis{<:Any,<:SemiclassicalJacobiWeight,<:SemiclassicalJacobi})
    A = parent(Ac)
    w,B = arguments(wB)
    P = SemiclassicalJacobi(w.t, w.a, w.b, w.c, B.P)
    (P\A)' * semiclassicaljacobi_massmatrix(P) * (P \ B)
end


# sqrt(1-(1-x)^2) == sqrt(2x-x^2) == sqrt(x)*sqrt(2-x)
# sqrt(1-(1-x)^2) == sqrt(2x-x^2) == sqrt(x)*sqrt(2-x)


end