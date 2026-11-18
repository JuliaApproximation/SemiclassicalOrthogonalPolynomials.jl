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

isnormalized(J::SemiclassicalJacobi) = J.b ≠ -1 # there is no normalisation for b == -1
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

function _getsecondifpossible(v)
    length(v) > 1 && return v[2]
    return v[1]
end

SemiclassicalJacobiFamily(t, a, b, c) = SemiclassicalJacobiFamily{float(promote_type(typeof(t),eltype(a),eltype(b),eltype(c)))}(t, a, b, c)
function SemiclassicalJacobiFamily{T}(t, a, b, c) where T
    # We need to start with a hierarchy containing two entries
    return SemiclassicalJacobiFamily{T}([SemiclassicalJacobi{T}(t, first(a), first(b), first(c)),SemiclassicalJacobi{T}(t, _getsecondifpossible(a), _getsecondifpossible(b), _getsecondifpossible(c))], t, a, b, c)
end

Base.broadcasted(::Type{SemiclassicalJacobi}, t::Number, a::Number, b::Number, c::Number) = SemiclassicalJacobi(t, a, b, c)
Base.broadcasted(::Type{SemiclassicalJacobi{T}}, t::Number, a::Number, b::Number, c::Number) where T = SemiclassicalJacobi{T}(t, a, b, c)
Base.broadcasted(::Type{SemiclassicalJacobi}, t::Number, a::Union{AbstractUnitRange,Number}, b::Union{AbstractUnitRange,Number}, c::Union{AbstractUnitRange,Number}) =
    SemiclassicalJacobiFamily(t, a, b, c)
Base.broadcasted(::Type{SemiclassicalJacobi{T}}, t::Number, a::Union{AbstractUnitRange,Number}, b::Union{AbstractUnitRange,Number}, c::Union{AbstractUnitRange,Number}) where T =
    SemiclassicalJacobiFamily{T}(t, a, b, c)


_broadcast_getindex(a,k) = a[k]
_broadcast_getindex(a::Number,k) = a

function LazyArrays.cache_filldata!(P::SemiclassicalJacobiFamily{T,<:Number,<:Number,<:AbstractUnitRange}, inds::AbstractUnitRange) where T
    t,a,b,c = P.t,P.a,P.b,P.c
    for k in inds
        Pprev = P.data[k-2]
        P.data[k] = SemiclassicalJacobi{T}(Pprev.t, Pprev.a, Pprev.b, Pprev.c+2, semiclassical_jacobimatrix_raise_c_by_2(Pprev))
    end
    P
end

function LazyArrays.cache_filldata!(P::SemiclassicalJacobiFamily{<:Number,<:Number,<:AbstractUnitRange}, inds::AbstractUnitRange)
    t,a,b,c = P.t,P.a,P.b,P.c
    for k in inds
        # If P.data[k-2] is not normalised (aka b = -1), cholesky fails. With the current design, this is only a problem if P.b
        # is a range since we can translate between polynomials that both have b = -1.
        Pprev = P.b[k-2] == -1 ? P.data[k-1] : P.data[k-2] # isrange && P.b[k-2] == -1 could also be !isnormalized(P.data[k-2])
        P.data[k] = SemiclassicalJacobi(t, _broadcast_getindex(a,k), _broadcast_getindex(b,k), _broadcast_getindex(c,k), Pprev)
    end
    P
end

###
# here we construct hierarchies of c weight sums by means of contiguous recurrence relations
###

""""
A SemiclassicalJacobiCWeightFamily

is a vector containing a sequence of weights of the form `x^a * (1-x)^b * (t-x)^c` where `a` and `b` are scalars and `c` is a range of values with integer spacing; where `x in 0..1`. It is automatically generated when calling `SemiclassicalJacobiWeight.(t,a,b,cmin:cmax)`.
"""
struct SemiclassicalJacobiCWeightFamily{T, C} <: AbstractVector{SemiclassicalJacobiWeight{T}}
    data::Vector{SemiclassicalJacobiWeight{T}}
    t::T
    a::T
    b::T
    c::C
    datasize::Tuple{Int}
end

getindex(W::SemiclassicalJacobiCWeightFamily, inds) = getindex(W.data, inds)

size(W::SemiclassicalJacobiCWeightFamily) = (length(W.c),)

function SemiclassicalJacobiCWeightFamily{T}(data::Vector, t, a, b, c) where T
    checkrangesizes(a, b, c)
    SemiclassicalJacobiCWeightFamily{T,typeof(c)}(data, t, a, b, c, (length(data),))
end

SemiclassicalJacobiCWeightFamily(t, a, b, c) = SemiclassicalJacobiCWeightFamily{float(promote_type(typeof(t),eltype(a),eltype(b),eltype(c)))}(t, a, b, c)
function SemiclassicalJacobiCWeightFamily{T}(t::Number, a::Number, b::Number, c::Union{AbstractUnitRange,Number}) where T
    return SemiclassicalJacobiCWeightFamily{T}(SemiclassicalJacobiWeight.(t,a:a,b:b,c), t, a, b, c)
end

Base.broadcasted(::Type{SemiclassicalJacobiWeight}, t::Number, a::Number, b::Number, c::Union{AbstractUnitRange,Number}) =
SemiclassicalJacobiCWeightFamily(t, a, b, c)

_unweightedsemiclassicalsum(a,b,c,t) = pFq((a+1,-c),(a+b+2, ), 1/t)

function Base.broadcasted(::typeof(sum), W::SemiclassicalJacobiCWeightFamily{T}) where T
    a = W.a; b = W.b; c = W.c; t = W.t;
    cmin = minimum(c); cmax = maximum(c);
    @assert isinteger(cmax) && isinteger(cmin)
    # This is needed at high parameter values.
    # Manually setting setprecision(2048) allows accurate computation even for very high c.
    t,a,b = convert(BigFloat,t),convert(BigFloat,a),convert(BigFloat,b)
    F = zeros(BigFloat,cmax+1)
    F[1] = _unweightedsemiclassicalsum(a,b,0,t) # c=0
    cmax == 0 && return abs.(convert.(T,t.^c.*exp(loggamma(a+1)+loggamma(b+1)-loggamma(a+b+2)).*getindex(F,1:1)))
    F[2] = _unweightedsemiclassicalsum(a,b,1,t) # c=1
    @inbounds for n in 1:cmax-1
        F[n+2] = ((n-1)/t+1/t-n)/(n+a+b+2)*F[n]+(a+b+4+2*n-2-(n+a+1)/t)/(n+a+b+2)*F[n+1]
    end
    return abs.(convert.(T,t.^c.*exp(loggamma(a+1)+loggamma(b+1)-loggamma(a+b+2)).*getindex(F,W.c.+1)))
end

""""
sumquotient(wP, wQ) computes sum(wP)/sum(wQ) by taking into account cancellations, allowing more stable computations for high weight parameters.
"""
function sumquotient(wP::SemiclassicalJacobiWeight{T},wQ::SemiclassicalJacobiWeight{T}) where T
    @assert wP.t ≈ wQ.t
    @assert isinteger(wP.c) && isinteger(wQ.c)
    a = wP.a; b = wP.b; c = Int(wP.c); t = wP.t;
    # This is needed at high parameter values.
    t,a,b = convert(BigFloat,t),convert(BigFloat,a),convert(BigFloat,b)
    F = zeros(BigFloat,max(2,c+1))
    F[1] = _unweightedsemiclassicalsum(a,b,0,t) # c=0
    F[2] = _unweightedsemiclassicalsum(a,b,1,t) # c=1
    @inbounds for n in 1:c-1
        F[n+2] = ((n-1)/t+1/t-n)/(n+a+b+2)*F[n]+(a+b+4+2*n-2-(n+a+1)/t)/(n+a+b+2)*F[n+1]
    end
    a = wQ.a; b = wQ.b; c = Int(wQ.c);
    t,a,b = convert(BigFloat,t),convert(BigFloat,a),convert(BigFloat,b)
    G = zeros(BigFloat,max(2,c+1))
    G[1] = _unweightedsemiclassicalsum(a,b,0,t) # c=0
    G[2] = _unweightedsemiclassicalsum(a,b,1,t) # c=1
    @inbounds for n in 1:c-1
        G[n+2] = ((n-1)/t+1/t-n)/(n+a+b+2)*G[n]+(a+b+4+2*n-2-(n+a+1)/t)/(n+a+b+2)*G[n+1]
    end
    return abs.(convert.(T,t.^(Int(wP.c)-c).*exp(loggamma(wP.a+1)+loggamma(wP.b+1)-loggamma(wP.a+wP.b+2)-loggamma(a+1)-loggamma(b+1)+loggamma(a+b+2))*F[Int(wP.c)+1]/G[c+1]))
end