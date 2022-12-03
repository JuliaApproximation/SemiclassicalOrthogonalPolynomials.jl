module SemiclassicalOrthogonalPolynomials
using ClassicalOrthogonalPolynomials: WeightedOPLayout
using ClassicalOrthogonalPolynomials, FillArrays, LazyArrays, ArrayLayouts, QuasiArrays, InfiniteArrays, ContinuumArrays, LinearAlgebra, BandedMatrices,
        SpecialFunctions, HypergeometricFunctions

import Base: getindex, axes, size, \, /, *, +, -, summary, ==, copy, sum, unsafe_getindex, convert, OneTo

import ArrayLayouts: MemoryLayout, ldiv, diagonaldata, subdiagonaldata, supdiagonaldata
import BandedMatrices: bandwidths, AbstractBandedMatrix, BandedLayout, _BandedMatrix
import LazyArrays: resizedata!, paddeddata, CachedVector, CachedMatrix, CachedAbstractVector, LazyMatrix, LazyVector, arguments, ApplyLayout, colsupport, AbstractCachedVector,
                    AccumulateAbstractVector, LazyVector, AbstractCachedMatrix, BroadcastLayout
import ClassicalOrthogonalPolynomials: OrthogonalPolynomial, recurrencecoefficients, jacobimatrix, normalize, _p0, UnitInterval, orthogonalityweight, NormalizedOPLayout,
                                        Bidiagonal, Tridiagonal, SymTridiagonal, symtridiagonalize, normalizationconstant, LanczosPolynomial,
                                        OrthogonalPolynomialRatio, Weighted, WeightLayout, UnionDomain, oneto, Hilbert, WeightedBasis, HalfWeighted,
                                        Associated, golubwelsch, associated, AbstractOPLayout, weight
import InfiniteArrays: OneToInf, InfUnitRange
import ContinuumArrays: basis, Weight, @simplify, AbstractBasisLayout, BasisLayout, MappedBasisLayout, grid, plotgrid, _equals, ExpansionLayout
import FillArrays: SquareEye
import HypergeometricFunctions: _₂F₁general2

export LanczosPolynomial, Legendre, Normalized, normalize, SemiclassicalJacobi, SemiclassicalJacobiWeight, WeightedSemiclassicalJacobi, OrthogonalPolynomialRatio, TwoBandJacobi, TwoBandWeight

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
    # (t,a,b,c) = map(big, map(float, (P.t,P.a,P.b,P.c)))
    (t,a,b,c) = P.t, P.a, P.b, P.c
    return convert(T, t^c * beta(1+a,1+b) * _₂F₁general2(1+a,-c,2+a+b,1/t))
end

function summary(io::IO, P::SemiclassicalJacobiWeight)
    t,a,b,c = P.t,P.a,P.b,P.c
    print(io, "x^$a * (1-x)^$b * ($t-x)^$c")
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

_equals(::WeightLayout, ::ExpansionLayout, A::SemiclassicalJacobiWeight, B) = jacobiexpansion(A) == B
_equals(::ExpansionLayout, ::WeightLayout, A, B::SemiclassicalJacobiWeight) = A == jacobiexpansion(B)


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
  the bands of the Jacobi matrix
"""
mutable struct SemiclassicalJacobiBand{dv,T} <: AbstractCachedVector{T}
    data::Vector{T}
    a::AbstractVector{T}
    b::AbstractVector{T}
    ℓ::AbstractVector{T}
    datasize::Tuple{Int}
end

function LazyArrays.cache_filldata!(r::SemiclassicalJacobiBand{:dv}, inds::AbstractUnitRange)
    rℓ = r.ℓ[inds[1]-1:inds[end]]; ℓ,sℓ = rℓ[2:end], rℓ[1:end-1]
    ra = r.a[inds[1]:(inds[end]+1)]; a = ra[1:end-1]; sa = ra[2:end]
    rb = r.b[inds[1]-1:inds[end]]; b = rb[2:end]; sb = rb[1:end-1]
    r.data[inds] .= @.(a - b * ℓ + sb*sℓ)
end

function LazyArrays.cache_filldata!(r::SemiclassicalJacobiBand{:ev}, inds::AbstractUnitRange)
    rℓ = r.ℓ[inds[1]-1:inds[end]]; ℓ,sℓ = rℓ[2:end], rℓ[1:end-1]
    ra = r.a[inds[1]:(inds[end]+1)]; a = ra[1:end-1]; sa = ra[2:end]
    rb = r.b[inds[1]-1:inds[end]]; b = rb[2:end]; sb = rb[1:end-1]
    r.data[inds] .= @.(sqrt((ℓ*a + b - b*ℓ^2 - sa*ℓ + ℓ*sb*sℓ)*b))
end

size(::SemiclassicalJacobiBand) = (ℵ₀,)

SemiclassicalJacobiBand{:dv,T}(a,b,ℓ) where T = SemiclassicalJacobiBand{:dv,T}(T[a[1] - b[1]ℓ[1]], a, b, ℓ, (1,))
SemiclassicalJacobiBand{:ev,T}(a,b,ℓ) where T = SemiclassicalJacobiBand{:ev,T}(T[sqrt((ℓ[1]*a[1] + b[1] - b[1]*ℓ[1]^2 - a[2]*ℓ[1])b[1])], a, b, ℓ, (1,))
SemiclassicalJacobiBand{dv}(a,b,ℓ) where dv = SemiclassicalJacobiBand{dv,promote_type(eltype(a),eltype(b),eltype(ℓ))}(a,b,ℓ)

copy(r::SemiclassicalJacobiBand) = r # immutable



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



function semiclassical_jacobimatrix(t, a, b, c)
    T = float(promote_type(typeof(t), typeof(a), typeof(b), typeof(c)))
    if a == 0 && b == 0 && c == -1
        # for this special case we can generate the Jacobi operator explicitly
        N = (1:∞)
        α = T.(neg1c_αcfs(t)) # cached coefficients
        A = Vcat((α[1]+1)/2 , -N./(N.*4 .- 2).*α .+ (N.+1)./(N.*4 .+ 2).*α[2:end].+1/2)
        C = -(N)./(N.*4 .- 2)
        B = Vcat((α[1]^2*3-α[1]*α[2]*2-1)/6 , -(N)./(N.*4 .+ 2).*α[2:end]./α)
        # if J is Tridiagonal(c,a,b) then for norm. OPs it becomes SymTridiagonal(a, sqrt.( b.* c))
        return SymTridiagonal(A, sqrt.(B.*C))
    end
    P = jacobi(b, a, UnitInterval{T}())
    iszero(c) && return symtridiagonalize(jacobimatrix(P))
    x = axes(P,1)
    jacobimatrix(LanczosPolynomial(@.(x^a * (1-x)^b * (t-x)^c), P))
end


function symraised_jacobimatrix(Q, y)
    ℓ = OrthogonalPolynomialRatio(Q,y)
    X = jacobimatrix(Q)
    a,b = diagonaldata(X), supdiagonaldata(X)
    SymTridiagonal(SemiclassicalJacobiBand{:dv}(a,b,ℓ), SemiclassicalJacobiBand{:ev}(a,b,ℓ))
end

function semiclassical_jacobimatrix(Q::SemiclassicalJacobi, a, b, c)
    if  iszero(a) && iszero(b) && c == -1 # (a,b,c) = (0,0,-1) special case
        semiclassical_jacobimatrix(Q.t, a, b, c)
    elseif a == Q.a+1 && b == Q.b && c == Q.c  # raising by 1
        symraised_jacobimatrix(Q, 0)
    elseif a == Q.a && b == Q.b+1 && c == Q.c
        symraised_jacobimatrix(Q, 1)
    elseif a == Q.a && b == Q.b && c == Q.c+1
        symraised_jacobimatrix(Q, Q.t)
    elseif a == Q.a && b == Q.b && c == Q.c-1 # lowering by 1
        symlowered_jacobimatrix(Q, :c)
    elseif a == Q.a-1 && b == Q.b && c == Q.c
        symlowered_jacobimatrix(Q, :a)
    elseif a == Q.a && b == Q.b-1 && c == Q.c
        symlowered_jacobimatrix(Q, :b)
    elseif a > Q.a  # iterative raising
        semiclassical_jacobimatrix(SemiclassicalJacobi(Q.t, Q.a+1, Q.b, Q.c, Q), a, b, c)
    elseif b > Q.b
        semiclassical_jacobimatrix(SemiclassicalJacobi(Q.t, Q.a, Q.b+1, Q.c, Q), a, b, c)
    elseif c > Q.c
        semiclassical_jacobimatrix(SemiclassicalJacobi(Q.t, Q.a, Q.b, Q.c+1, Q), a, b, c)
    elseif b < Q.b  # iterative lowering
        semiclassical_jacobimatrix(SemiclassicalJacobi(Q.t, Q.a, Q.b-1, Q.c, Q), a, b, c)
    elseif c < Q.c
        semiclassical_jacobimatrix(SemiclassicalJacobi(Q.t, Q.a, Q.b, Q.c-1, Q), a, b, c)
    elseif a < Q.a 
        semiclassical_jacobimatrix(SemiclassicalJacobi(Q.t, Q.a-1, Q.b, Q.c, Q), a, b, c)
    elseif a == Q.a && b == Q.b && c == Q.c # same basis
        jacobimatrix(Q)
    else
        error("Not Implemented")
    end
end

LanczosPolynomial(P::SemiclassicalJacobi{T}) where T =
    LanczosPolynomial(orthogonalityweight(P), Normalized(jacobi(P.b, P.a, UnitInterval{T}())), P.X.dv.data)

"""
    toclassical(P::SemiclassicalJacobi)

gives either a mapped `Jacobi` or `LanczosPolynomial` version of `P`.
"""
toclassical(P::SemiclassicalJacobi{T}) where T = iszero(P.c) ? Normalized(jacobi(P.b, P.a, UnitInterval{T}())) : LanczosPolynomial(P)

copy(P::SemiclassicalJacobi) = P
axes(P::SemiclassicalJacobi{T}) where T = (Inclusion(UnitInterval{T}()),OneToInf())

==(A::SemiclassicalJacobi, B::SemiclassicalJacobi) = A.t == B.t && A.a == B.a && A.b == B.b && A.c == B.c
==(::AbstractQuasiMatrix, ::SemiclassicalJacobi) = false
==(::SemiclassicalJacobi, ::AbstractQuasiMatrix) = false

orthogonalityweight(P::SemiclassicalJacobi) = SemiclassicalJacobiWeight(P.t, P.a, P.b, P.c)

function summary(io::IO, P::SemiclassicalJacobi)
    t,a,b,c = P.t,P.a,P.b,P.c
    print(io, "SemiclassicalJacobi with weight x^$a * (1-x)^$b * ($t-x)^$c")
end



jacobimatrix(P::SemiclassicalJacobi) = P.X

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

struct SemiclassicalJacobiLayout <: AbstractOPLayout end
MemoryLayout(::Type{<:SemiclassicalJacobi}) = SemiclassicalJacobiLayout()

copy(L::Ldiv{<:NormalizedOPLayout,SemiclassicalJacobiLayout}) = copy(Ldiv{ApplyLayout{typeof(*)},SemiclassicalJacobiLayout}(L.A, L.B))
copy(L::Ldiv{SemiclassicalJacobiLayout,<:NormalizedOPLayout}) = copy(Ldiv{SemiclassicalJacobiLayout,ApplyLayout{typeof(*)}}(L.A, L.B))

copy(L::Ldiv{ApplyLayout{typeof(*)},SemiclassicalJacobiLayout}) = copy(Ldiv{ApplyLayout{typeof(*)},BasisLayout}(L.A, L.B))
copy(L::Ldiv{SemiclassicalJacobiLayout,ApplyLayout{typeof(*)}}) = copy(Ldiv{BasisLayout,ApplyLayout{typeof(*)}}(L.A, L.B))

copy(L::Ldiv{MappedBasisLayout,SemiclassicalJacobiLayout}) = semijacobi_ldiv(L.A, L.B)
copy(L::Ldiv{SemiclassicalJacobiLayout,MappedBasisLayout}) = semijacobi_ldiv(L.A, L.B)

copy(L::Ldiv{WeightedOPLayout,SemiclassicalJacobiLayout}) = copy(Ldiv{WeightedOPLayout,BasisLayout}(L.A, L.B))
copy(L::Ldiv{SemiclassicalJacobiLayout,WeightedOPLayout}) = copy(Ldiv{BasisLayout,WeightedOPLayout}(L.A, L.B))


copy(L::Ldiv{SemiclassicalJacobiLayout}) = semijacobi_ldiv(L.A, L.B)
copy(L::Ldiv{SemiclassicalJacobiLayout,<:AbstractBasisLayout}) = semijacobi_ldiv(L.A, L.B)
copy(L::Ldiv{SemiclassicalJacobiLayout,BroadcastLayout{typeof(*)}}) = semijacobi_ldiv(L.A, L.B)
copy(L::Ldiv{<:Any,SemiclassicalJacobiLayout}) = semijacobi_ldiv(L.A, L.B)
copy(L::Ldiv{<:AbstractBasisLayout,SemiclassicalJacobiLayout}) = semijacobi_ldiv(L.A, L.B)
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
        C = SemiclassicalJacobi(A.t, A.a+1, A.b, A.c, A)
        w_C = SemiclassicalJacobiWeight(wA.t, wA.a+1, wA.b, wA.c) .* C
        L_2 = w_C \ w_B
        L_1 = w_A \ w_C
        L_1 * L_2
    elseif wA.b+1 ≤ wB.b
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

massmatrix(P::SemiclassicalJacobi) = Diagonal(Fill(sum(orthogonalityweight(P)),∞))

@simplify function *(Ac::QuasiAdjoint{<:Any,<:SemiclassicalJacobi}, wB::WeightedBasis{<:Any,<:SemiclassicalJacobiWeight,<:SemiclassicalJacobi})
    A = parent(Ac)
    w,B = arguments(wB)
    P = SemiclassicalJacobi(w.t, w.a, w.b, w.c)
    (P\A)' * massmatrix(P) * (P \ B)
end

function ldiv(Q::SemiclassicalJacobi, f::AbstractQuasiVector)
    if iszero(Q.a) && iszero(Q.b) && Q.c == -1
        # todo: due to a stdlib error this won't work with bigfloat as is
        T = typeof(Q.t)
        R = Legendre{T}()[affine(Inclusion(zero(T)..one(T)), axes(Legendre{T}(),1)), :]
        B = neg1c_tolegendre(Q.t)
        return (B \ (R \ f))
    end
    R = SemiclassicalJacobi(Q.t, mod(Q.a,-1), mod(Q.b,-1), mod(Q.c,-1))
    R̃ = toclassical(R)
    return (Q \ R̃) * (R̃ \ f)
end
function ldiv(Qn::SubQuasiArray{<:Any,2,<:SemiclassicalJacobi,<:Tuple{<:Inclusion,<:Any}}, C::AbstractQuasiArray)
    _,jr = parentindices(Qn)
    Q = parent(Qn)
    if iszero(Q.a) && iszero(Q.b) && Q.c == -1
        # todo: due to a stdlib error this won't work with bigfloat as is
        T = typeof(Q.t)
        R = Legendre{T}()[affine(Inclusion(zero(T)..one(T)), axes(Legendre{T}(),1)), :]
        B = neg1c_tolegendre(Q.t)
        return (B[jr,jr] \ (R[:,jr] \ C))
    end
    R = SemiclassicalJacobi(Q.t, mod(Q.a,-1), mod(Q.b,-1), mod(Q.c,-1))
    R̃ = toclassical(R)
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


include("twoband.jl")
include("derivatives.jl")


###
# Hierarchy
#
# here we build the operators lazily
###

mutable struct SemiclassicalJacobiFamily{T, A, B, C} <: AbstractCachedVector{SemiclassicalJacobi{T}}
    data::Vector{SemiclassicalJacobi{T}}
    t::T
    a::A
    b::B
    c::C
    datasize::Tuple{Int}
end

size(P::SemiclassicalJacobiFamily) = (max(length(P.a), length(P.b), length(P.c)),)

_checkrangesizes() = ()
_checkrangesizes(a::Number, b...) = _checkrangesizes(b...)
_checkrangesizes(a, b...) = (length(a), _checkrangesizes(b...)...)

_isequal() = true
_isequal(a) = true
_isequal(a,b,c...) = a == b && _isequal(b,c...)

checkrangesizes(a...) = _isequal(_checkrangesizes(a...)...) || throw(DimensionMismatch())

function SemiclassicalJacobiFamily{T}(data::Vector, t, a, b, c) where T
    checkrangesizes(a, b, c)
    SemiclassicalJacobiFamily{T,typeof(a),typeof(b),typeof(c)}(data, t, a, b, c, (length(data),))
end

SemiclassicalJacobiFamily(t, a, b, c) = SemiclassicalJacobiFamily{float(promote_type(typeof(t),eltype(a),eltype(b),eltype(c)))}(t, a, b, c)
SemiclassicalJacobiFamily{T}(t, a, b, c) where T = SemiclassicalJacobiFamily{T}([SemiclassicalJacobi{T}(t, first(a), first(b), first(c))], t, a, b, c)

Base.broadcasted(::Type{SemiclassicalJacobi}, t::Number, a::Number, b::Number, c::Number) = SemiclassicalJacobi(t, a, b, c)
Base.broadcasted(::Type{SemiclassicalJacobi{T}}, t::Number, a::Number, b::Number, c::Number) where T = SemiclassicalJacobi{T}(t, a, b, c)
Base.broadcasted(::Type{SemiclassicalJacobi}, t::Number, a::Union{AbstractUnitRange,Number}, b::Union{AbstractUnitRange,Number}, c::Union{AbstractUnitRange,Number}) = 
    SemiclassicalJacobiFamily(t, a, b, c)
Base.broadcasted(::Type{SemiclassicalJacobi{T}}, t::Number, a::Union{AbstractUnitRange,Number}, b::Union{AbstractUnitRange,Number}, c::Union{AbstractUnitRange,Number}) where T = 
    SemiclassicalJacobiFamily{T}(t, a, b, c)


_broadcast_getindex(a,k) = a[k]
_broadcast_getindex(a::Number,k) = a

function LazyArrays.cache_filldata!(P::SemiclassicalJacobiFamily, inds::AbstractUnitRange)
    t,a,b,c = P.t,P.a,P.b,P.c
    for k in inds
        P.data[k] = SemiclassicalJacobi(t, _broadcast_getindex(a,k), _broadcast_getindex(b,k), _broadcast_getindex(c,k), P.data[k-1])
    end
    P
end

include("lowering.jl")

end