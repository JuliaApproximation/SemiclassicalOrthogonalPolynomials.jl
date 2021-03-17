twoband(ρ) = UnionDomain(-1..(-ρ), ρ..1)


"""
    TwoBandWeight(a, b)

is a quasi-vector representing `|x|^(2c) * (x^2-ρ)^b * (1-x^2)^a`
"""
struct TwoBandWeight{T} <: Weight{T}
    ρ::T
    a::T
    b::T
    c::T
end


TwoBandWeight{T}() where T = TwoBandWeight{T}(zero(T), zero(T))
TwoBandWeight() = TwoBandWeight{Float64}()

copy(w::TwoBandWeight) = w

axes(w::TwoBandWeight{T}) where T = (Inclusion(twoband(w.ρ)),)

==(w::TwoBandWeight, v::TwoBandWeight) = w.a == v.a && w.b == v.b && w.ρ == v.ρ && w.c == v.c

getindex(w::TwoBandWeight, x::Real) = abs(x)^(2w.c) * (x^2- ρ^2)^w.b * (1-x^2)^w.a

"""
    TwoBandJacobi(a, b)

is a quasi-matrix orthogonal `|x|^(2c) * (x^2 - ρ^2)^b * (1-x^2)^a`.
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

function  getindex(R::TwoBandJacobi, x::Real, j::Integer)
    ρ = R.ρ
    if isodd(j)
        R.P[(1-x^2)/(1-ρ^2), (j+1)÷2]
    else
        x * R.Q[(1-x^2)/(1-ρ^2), j÷2]
    end
end


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
    Tridiagonal(Interlace(L.dv/L[1,1], (1-ρ^2) * L.ev), Zeros{T}(∞), Interlace((1-ρ^2) * L.dv,L.ev/L[1,1]))
end
