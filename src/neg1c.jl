###
#   α coefficients implementation, cached, direct and via recurrences
###

# inital value n=0 for α_{n,n-1}(t) coefficients
initialα(t) = t-BigInt(2)/(log1p(t)-log(t-1))

# compute n-th coefficient from direct evaluation formula
αdirect(n, t) = gamma((2*n+1)/BigInt(2))*gamma(n+1)*HypergeometricFunctions._₂F₁general2((1+n)/BigInt(2),(2+n)/BigInt(2),(2*n+3)/BigInt(2),BigInt(1)/t^2)/(t*2*gamma(BigInt(n))*gamma((2*n+3)/BigInt(2))*HypergeometricFunctions._₂F₁general2(n/BigInt(2),(n+1)/BigInt(2),(2*n+1)/BigInt(2),BigInt(1)/t^2))
# this version takes a pre-existing vector v and fills in the missing data guided by indices in inds using explicit formula
function αdirect!(α, t, inds) 
    @inbounds for n in inds
        α[n] = gamma((2*n+1)/BigInt(2))*gamma(1+n)*HypergeometricFunctions._₂F₁general2((1+n)/BigInt(2),(2+n)/BigInt(2),(2*n+3)/BigInt(2),1/t^2)/(t*2*gamma(n)*gamma((2*n+3)/BigInt(2))*HypergeometricFunctions._₂F₁general2(n/BigInt(2),(n+1)/BigInt(2),(2*n+1)/BigInt(2),1/t^2))
    end
end

# takes a previously computed vector of α_{n,n-1}(t) that has been increased in size and fills in the missing data guided by indices in inds via forward recurrence
function αcoefficients!(α, t, inds)
    @inbounds for n in inds
       α[n] = (t*(2*n-1)/n-(n-1)/(n*α[n-1]))
    end
end

# takes a previously computed vector of α_{n,n-1}(t) that has been increased in size and fills in the missing data guided by indices in inds, using a final condition computation and backward recurrence
function backαcoeff!(α, t, inds)
    α[end] = αdirect(BigInt(length(α)),t)
    @inbounds for n in reverse(inds)[1:end-1]
       α[n-1] = BigInt(n-1)/(t*(2*n-1)-n*α[n]) #BigInt(1-n)/(t-2*BigInt(n)*t+BigInt(n)*α[n])
    end
end

# cached implementation using stable back recurrence to fill data
mutable struct neg1c_αcfs{T} <: AbstractCachedVector{T}
    t::T
    data::Vector{T}
    datasize::Tuple{Int}
    array
    neg1c_αcfs{T}(t::T) where T = new{T}(t, [initialα(2*t-1),αdirect(2,2*t-1)], (2,))
end
neg1c_αcfs(t::T) where T = neg1c_αcfs{T}(t)
size(α::neg1c_αcfs) = (ℵ₀,)
cache_filldata!(α::neg1c_αcfs, inds) = backαcoeff!(α.data, 2*α.t-1, inds)

function resizedata!(α::neg1c_αcfs, nm) 
    olddata = copy(α.data)
    νμ = length(olddata)
    nm = maximum(nm)
    nm = max(νμ,nm)
    if νμ ≠ nm
        α.data = similar(olddata,maximum(nm))
        α.data[1:νμ] = olddata[1:νμ]
    end
    if maximum(nm) > νμ
        inds = Array(νμ-1:maximum(nm))
        cache_filldata!(α, inds)
        α.datasize = (nm,)
    end
    α
end

###
#   cached implementation of normalization constants
###
mutable struct neg1c_normconstant{T} <: AbstractCachedVector{T}
    t::T
    data::Vector{T}
    datasize::Tuple{Int}
    array
    neg1c_normconstant{T}(t::T) where T = new{T}(t, neg1c_normconstinitial(t,10), (10,))
end
neg1c_normconstant(t::T) where T = neg1c_normconstant{T}(t)
size(B::neg1c_normconstant) = (ℵ₀,)
cache_filldata!(B::neg1c_normconstant, inds) = neg1c_normconstextension!(B.data, inds, B.t)

function resizedata!(B::neg1c_normconstant, nm) 
    olddata = copy(B.data)
    νμ = length(olddata)
    nm = maximum(nm)
    nm = max(νμ,nm)
    if νμ ≠ nm
        B.data = similar(olddata,maximum(nm))
        B.data[1:νμ] = olddata[1:νμ]
    end
    if maximum(nm) > νμ
        inds = Array(νμ-1:maximum(nm))
        cache_filldata!(B, inds)
        B.datasize = (nm,)
    end
    B
end

function neg1c_normconstinitial(t::T, N) where T
    # generate α coefficients for OPs via recurrence
    α = zeros(T,N-1)
    α[1] = initialα(2*t-1)
    backαcoeff!(α,2*t-1,(2:N-1))
    # normalization constants
    B = [one(T)]
    append!(B,sqrt(2*acoth(2*t-1))*sqrt.((1:N-1)./(2 .*α)))
    return B
end

function neg1c_normconstextension!(B::Vector, inds, t::T) where T
    n = maximum(inds)
    m = minimum(inds)-1
    # generate missing α coefficients
    α = zeros(T, n)
    backαcoeff!(α,2*t-1,(m:n))
    # normalization constants
    norm = sqrt(2*acoth(2*t-1))*sqrt.((m:n)./(2 .*α[m:n]))
    B[m+1:n] = norm[1:end-1]
    B
end

###
#   bidiagonal operator which converts from OPs wrt (t-x)^-1 to shifted Legendre
###
function neg1c_tolegendre(t::T) where T
    norm = neg1c_normconstant(t)
    α = neg1c_αcfs(t)
    supD = norm.*Vcat(zeros(T,1),α).*(-1).^(1:∞)
    D = norm.*(-1).^(0:∞)
    return Bidiagonal(D,supD[2:end],:U)
end

###
#   explicit polynomial evaluations
###
# Evaluate the n-th normalized OP wrt 1/(t-x) at point x
function evalQn(n::Integer, x, t::T) where T
    # this version recomputes α based on t
    t <= 1 && throw(ArgumentError("t must be greater than 1."))
    n == 0 && return one(T)
    α = αdirect(n,2*t-1)
    return sqrt(2*acoth(2*t-1))*(α*jacobip(n-1,0,0,1-2*x)+jacobip(n,0,0,1-2*x))*sqrt(n/(2*α))
end
# Evaluate the n-th non-normalized OP wrt 1/(t-x) at point x
function evalϕn(n::Integer, x, t::T) where T
    # this version recomputes α based on t
    t <= 1 && throw(ArgumentError("t must be greater than 1."))
    n == 0 && return one(T)
    return αdirect(n,2*t-1)*jacobip(n-1,0,0,1-2*x)+jacobip(n,0,0,1-2*x)
end