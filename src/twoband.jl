twoband(ρ::T) where T = UnionDomain(-one(T)..(-ρ), ρ..one(T))
twoband_0(ρ::T) where T = UnionDomain(-one(T)..(-ρ), zero(T), ρ..one(T))

"""
    TwoBandWeight(ρ, a, b, c)

is a quasi-vector representing `|x|^(2c) * (x^2-ρ^2)^b * (1-x^2)^a`  for `ρ ≤ |x| ≤ 1`
"""
struct TwoBandWeight{T} <: Weight{T}
    ρ::T
    a::T
    b::T
    c::T
end

TwoBandWeight(ρ::R, a::A, b::B, c::C) where {R,A,B,C} = TwoBandWeight{promote_type(R,A,B,C)}(ρ, a, b, c)

copy(w::TwoBandWeight) = w

axes(w::TwoBandWeight{T}) where T = (Inclusion(twoband(w.ρ)),)

==(w::TwoBandWeight, v::TwoBandWeight) = w.a == v.a && w.b == v.b && w.ρ == v.ρ && w.c == v.c

function getindex(w::TwoBandWeight, x::Real)
    @boundscheck checkbounds(w, x)
    abs(x)^(2w.c) * (x^2- w.ρ^2)^w.b * (1-x^2)^w.a
end

function sum(w::TwoBandWeight{T}) where T
    if 2w.a == 2w.b == -1 && 2w.c == 1
        convert(T,π)
    else
        a,b,c,ρ = w.a,w.b,w.c,w.ρ
        convert(T,π)/2 * sec((b + c)*convert(T,π)) * (
            - ρ^(one(T) + 2b + 2c) * gamma(one(T) + b) * _₂F₁(-a, one(T)/2 + c, convert(T,3)/2 + b + c, ρ^2) / gamma(one(T)/2 - c)
            + gamma(one(T) + a) * _₂F₁(-b, -one(T)/2 - a - b - c, one(T)/2 - b - c, ρ^2) / gamma(convert(T,3)/2 + a + b + c)
        )
    end
end


"""
    TwoBandJacobi(ρ, a, b, c)

is a quasi-matrix orthogonal `|x|^(2c) * (x^2 - ρ^2)^b * (1-x^2)^a` for `ρ ≤ |x| ≤ 1`.
"""
struct TwoBandJacobi{T} <: OrthogonalPolynomial{T}
    ρ::T
    a::T
    b::T
    c::T
    P::SemiclassicalJacobi{T}
    Q::SemiclassicalJacobi{T}
    TwoBandJacobi{T}(ρ::T, a::T, b::T, c::T, P::SemiclassicalJacobi{T}, Q::SemiclassicalJacobi{T}) where T = new{T}(ρ, a, b, c, P, Q)
end
function TwoBandJacobi{T}(ρ, a, b, c) where T
    t = inv(1-ρ^2)
    P = SemiclassicalJacobi(t, a, b, c-one(T)/2)
    TwoBandJacobi{T}(convert(T,ρ), convert(T,a), convert(T,b), convert(T,c), P, SemiclassicalJacobi(t, a, b, c+one(T)/2, P))
end
TwoBandJacobi(ρ, a, b, c) = TwoBandJacobi{float(promote_type(eltype(ρ),eltype(a),eltype(b),eltype(c)))}(ρ, a, b, c)

axes(P::TwoBandJacobi{T}) where T = (Inclusion(twoband(P.ρ)),oneto(∞))

==(w::TwoBandJacobi, v::TwoBandJacobi) = w.ρ == v.ρ && w.a == v.a && w.b == v.b && w.c == v.c

copy(A::TwoBandJacobi) = A

orthogonalityweight(Z::TwoBandJacobi) = TwoBandWeight(Z.ρ, Z.a, Z.b, Z.c)

# function  getindex(R::TwoBandJacobi, x::Real, j::Integer)
#     ρ = R.ρ
#     if isodd(j)
#         R.P[(1-x^2)/(1-ρ^2), (j+1)÷2]
#     else
#         x * R.Q[(1-x^2)/(1-ρ^2), j÷2]
#     end
# end

weight(W::HalfWeighted{:ab,T,<:TwoBandJacobi}) where T = TwoBandWeight(W.P.ρ, W.P.a,W.P.b,zero(T))
convert(::Type{WeightedBasis}, Q::HalfWeighted{:ab,T,<:TwoBandJacobi}) where T = TwoBandWeight(Q.P.ρ, Q.P.a,Q.P.b,zero(T)) .* Q.P

struct Interlace{T,AA,BB} <: LazyVector{T}
    a::AA
    b::BB
end

Interlace{T}(a::AbstractVector{T}, b::AbstractVector{T}) where T = Interlace{T,typeof(a),typeof(b)}(a,b)
Interlace(a::AbstractVector{T}, b::AbstractVector{V}) where {T,V} = Interlace{promote_type(T,V)}(a, b)

size(::Interlace) = (ℵ₀,)

getindex(A::Interlace{T}, k::Int) where T = convert(T, isodd(k) ? A.a[(k+1)÷2] : A.b[k÷2])::T

function jacobimatrix(R::TwoBandJacobi{T}) where T
    ρ = R.ρ
    L = R.P \ (SemiclassicalJacobiWeight(R.P.t,0,0,1) .* R.Q)
    # M = (L / L[1,1])' # equal to R.Q \ R.P

    Tridiagonal(Interlace(L.dv/L[1,1], (ρ^2-1) * L.ev), Zeros{T}(∞), Interlace((1-ρ^2) * L.dv,L.ev/(-L[1,1])))
    # Tridiagonal(Interlace(L.dv/L[1,1], (1-ρ^2) * L.ev), Zeros{T}(∞), Interlace((1-ρ^2) * L.dv, L.ev/L[1,1]))
end


const ConvKernel2{T,D1,D2} = BroadcastQuasiMatrix{T,typeof(-),Tuple{D1,QuasiAdjoint{T,D2}}}
const Hilbert2{T,D1,D2} = BroadcastQuasiMatrix{T,typeof(inv),Tuple{ConvKernel2{T,Inclusion{T,D1},Inclusion{T,D2}}}}

@simplify function *(H::Hilbert2, w::TwoBandWeight)
    if 2w.a == 2w.b == -1 && 2w.c == 1 && axes(H,2) == axes(w,1) && axes(H,1).domain ⊆ twoband_0(w.ρ)
        zeros(promote_type(eltype(H),eltype(w)), axes(H,1))
    else
       error("Not Implemented")
    end
end

function plotgrid(L::SubQuasiArray{T,2,<:TwoBandJacobi,<:Tuple{Inclusion,AbstractUnitRange}}) where T
    g = plotgrid(legendre(parent(L).ρ .. 1)[:,parentindices(L)[2]])
    [-g; g]
end

###
# Associated
###

axes(Q::Associated{T,<:TwoBandJacobi}) where T = (Inclusion(twoband_0(Q.P.ρ)), axes(Q.P,2))
function golubwelsch(V::SubQuasiArray{T,2,<:Normalized{<:Any,<:Associated{<:Any,<:TwoBandJacobi}},<:Tuple{Inclusion,AbstractUnitRange}}) where T
    x,w = golubwelsch(jacobimatrix(V))
    n = length(x)
    if isodd(n)
        x[(n+1)÷2] = zero(T) # make exactly zero
    else
        x[n÷2] = zero(T) # make exactly zero
        x[(n÷2)+1] = zero(T) # make exactly zero
    end
    w .*= sum(orthogonalityweight(parent(V)))
    x,w
end

@simplify function *(H::Hilbert2, wP::Weighted{<:Any,<:TwoBandJacobi}) 
    P = wP.P
    w = orthogonalityweight(P)
    A = recurrencecoefficients(P)[1]
    Q = associated(P)
    @assert axes(H,1) == axes(Q,1)
    Q * BandedMatrix(1 =>Fill(-A[1]*sum(w),∞))
end


##
# Derivative of double weighted TwoBandJacobi
##
function divmul(R::TwoBandJacobi, D::Derivative, HP::HalfWeighted{:ab,<:Any,<:TwoBandJacobi})
    T = promote_type(eltype(R), eltype(HP))
    ρ=convert(T,R.ρ); t=inv(one(T)-ρ^2)
    a,b,c = R.a,R.b,R.c

    Dₑ = -2*(one(T)-ρ^2) .* (R.Q \ (Derivative(axes(R.Q,1))*HalfWeighted{:ab}(HP.P.P)))
    D₀ = -2*(one(T)-ρ^2)^2 .* (Weighted(R.P) \ (Derivative(axes(R.P,1))*Weighted(HP.P.Q)))

    (dₑ, dlₑ, d₀, dl₀) = Dₑ.data[1,:], Dₑ.data[2,:], D₀.data[1,:], D₀.data[2,:]
    BandedMatrix(-1=>Interlace(dₑ, -d₀), -3=>Interlace(-dlₑ, dl₀))
end

@simplify function *(D::Derivative, HP::HalfWeighted{:ab,<:Any,<:TwoBandJacobi})
    P = HP.P
    ρ = P.ρ
    @assert P.a == 1 && P.b == 1 && P.c == 0

    R = TwoBandJacobi(ρ, P.a-1,P.b-1,P.c)
    R * divmul(R, D, HP)
end

###
# L^2 inner product of double weighted TwoBandJacobi
###
@simplify function *(A::QuasiAdjoint{<:Any,<:HalfWeighted{:ab,<:Any,<:TwoBandJacobi}}, B::HalfWeighted{:ab,<:Any,<:TwoBandJacobi})
    T = promote_type(eltype(A), eltype(B))
    P = B.P
    a,b,c = P.a,P.b,P.c
    
    @assert A.parent.P.a == a && A.parent.P.b == b && A.parent.P.c == c
    @assert a == 1 && b == 1 && c == 0

    ρ = convert(T,P.ρ)
    t = inv(one(T)-ρ^2)
    Pₗ = SemiclassicalJacobi{T}(t,a-one(T),b-one(T),c-one(T)/2)
    Qₗ = SemiclassicalJacobi{T}(t,a-one(T),b-one(T),c+one(T)/2)

    Lₑ = Pₗ \ HalfWeighted{:ab}(P.P)
    Lₒ = Weighted(Qₗ) \ Weighted(P.Q)
    
    mₑ = (one(T)-ρ^2)^4*(1-ρ^2)^(one(T)/2)*sum(orthogonalityweight(Pₗ))
    mₒ = (one(T)-ρ^2)^4*(1-ρ^2)^(convert(T,3)/2)*sum(orthogonalityweight(Qₗ))

    mₑ = Fill{T}(mₑ, ∞);  mₒ = Fill{T}(mₒ, ∞)

    # Sum of entries in each column squared.
    dₑ = mₑ.*((Lₑ .* Lₑ)' * Ones{T}(∞))
    d₀ = mₒ.*((Lₒ .* Lₒ)' * Ones{T}(∞))

    # Sum of entries x entries in next column.
    dlₑ = mₑ.* (Lₑ[2:∞,:] .* Lₑ[2:∞,2:∞])' * Ones{T}(∞)
    dl₀ = mₒ.* (Lₒ[2:∞,:] .* Lₒ[2:∞,2:∞])' * Ones{T}(∞)

    # Sum of entries and entries in second column over.
    dllₑ = mₑ.* (Lₑ[3:∞,:] .* Lₑ[3:∞,3:∞])' * Ones{T}(∞)
    dll₀ = mₒ.* (Lₒ[3:∞,:] .* Lₒ[3:∞,3:∞])' * Ones{T}(∞)
    
    BandedMatrix(0=>Interlace(dₑ, d₀), 
        -2=>Interlace(-dlₑ, -dl₀), 2=>Interlace(-dlₑ, -dl₀), 
        -4=>Interlace(dllₑ, dll₀), 4=>Interlace(dllₑ, dll₀))
end