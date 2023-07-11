###########
# Special case of SemiclassicalJacobi(t,0,0,-1)
# We can build this from shifted and normalized Legendre() with explicit methods.
######

# compute n-th coefficient from direct evaluation formula
function αdirect(n, tt::T) where T
    t = big(tt)
    return convert(T,2*n*_₂F₁((one(T)+n)/2,(n+2)/2,(2*n+3)/2,1/t^2)/(t*2*(1+2*n)*_₂F₁(n/2,(n+one(T))/2,(2*n+one(T))/2,1/t^2)))
end

# inital value n=0 for α_{n,n-1}(t) coefficients
initialα(t) = t-2/(log1p(t)-log(t-1))

# cached implementation
mutable struct neg1c_αcfs{T} <: AbstractCachedVector{T}
    data::Vector{T}
    t::T
    datasize::Int
    neg1c_αcfs{T}(t::T) where T = new{T}([initialα(2*t-1),αdirect(2,2*t-1)], t, 2)
end
neg1c_αcfs(t::T) where T = neg1c_αcfs{T}(t)
size(α::neg1c_αcfs) = (ℵ₀,)

function resizedata!(α::neg1c_αcfs, nm) 
    νμ = length(α.data)
    if nm > νμ
        resize!(α.data,nm)
        cache_filldata!(α, νμ:nm)
        α.datasize = nm
    end
    α
end

# fills in the missing data using a final condition and backward recurrence
cache_filldata!(α::neg1c_αcfs, inds) = backαcoeff!(α.data, 2*α.t-1, inds)
function backαcoeff!(α, t, inds)
    α[end] = αdirect(length(α),t)
    @inbounds for n in reverse(inds)[1:end-1]
       α[n-1] = (n-1)/(t*(2*n-1)-n*α[n])
    end
end

# cached implementation of normalization constants
mutable struct neg1c_normconstant{T} <: AbstractCachedVector{T}
    data::Vector{T}
    t::T
    datasize::Int
    neg1c_normconstant{T}(t::T) where T = new{T}(neg1c_normconstinitial(t, 10), t, 10)
end
neg1c_normconstant(t::T) where T = neg1c_normconstant{T}(t)
size(B::neg1c_normconstant) = (ℵ₀,)

function neg1c_normconstinitial(t::T, N::Integer) where T
    # generate α coefficients for OPs via recurrence
    α = zeros(T,N-1)
    α[1] = initialα(2*t-1)
    backαcoeff!(α,2*t-1,(2:N-1))
    return [one(T);sqrt(2*acoth(2*t-1))*sqrt.((1:N-1)./(2 .*α))]
end

function resizedata!(B::neg1c_normconstant, nm) 
    νμ = length(B.data)
    if nm > νμ
        resize!(B.data, nm)
        cache_filldata!(B, νμ:nm)
        B.datasize = nm
    end
    B
end
function cache_filldata!(B::neg1c_normconstant{T}, inds::UnitRange{Int}) where T
    n = maximum(inds)
    m = minimum(inds)-1
    # generate missing α coefficients
    α = zeros(T, n)
    backαcoeff!(α,2*B.t-1,(m:n))
    # normalization constants
    norm = sqrt(2*acoth(2*B.t-1))*sqrt.((m:n)./(2 .*α[m:n]))
    B.data[m+1:n] = norm[1:end-1]
end

# bidiagonal operator which converts from OPs wrt (t-x)^-1 to shifted Legendre
function neg1c_tolegendre(t::T) where T
    nc = neg1c_normconstant(t)
    α = neg1c_αcfs(t)
    return Bidiagonal(nc,(nc.*Vcat(zeros(T,1),-α))[2:end],:U)
end

# Evaluate the n-th normalized OP wrt 1/(t-x) at point x
function evalQn(n::Integer, x, t::T) where T
    t <= 1 && throw(ArgumentError("t must be greater than 1."))
    n == 0 && return one(T)
    α = αdirect(n,2*t-1)
    return sqrt(2*acoth(2*t-1))*(-α*jacobip(n-1,0,0,2*x-1)+jacobip(n,0,0,2*x-1))*sqrt(n/(2*α))
end
# Evaluate the n-th NOT normalized OP wrt 1/(t-x) at point x
function evalϕn(n::Integer, x, t::T) where T
    t <= 1 && throw(ArgumentError("t must be greater than 1."))
    n == 0 && return one(T)
    return -αdirect(n,2*t-1)*jacobip(n-1,0,0,2*x-1)+jacobip(n,0,0,2*x-1)
end