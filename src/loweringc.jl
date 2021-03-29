##
# Methods to compute the Jacobi operator for (t,a,b,c-1) based on knowledge of (t,a,b,c)
##

# directly evaluate α[1] via hypergeometric functions, making use of the fact that Q_1 == A[1]x + B[1]
function initialαc_gen(t, a, b, c)
    A,B,_ = recurrencecoefficients(SemiclassicalJacobi(t,a,b,c))
    return -(A[1]*t^(c-1)*gamma(2+a)*gamma(1+b)/gamma(3+a+b)*_₂F₁general2(a+2,-c+1,a+3+b,1/t)/(t^(c-1)*gamma(1+a)gamma(1+b)/gamma(2+a+b)*_₂F₁general2(1+a,-c+1,2+a+b,1/t))+B[1])
end
# this alternative version just uses the built-in sums and Q_1 == A[1]x + B[1]
function initialαc_alt(t, a, b, c)
    A,B,_ = recurrencecoefficients(SemiclassicalJacobi(t,a,b,c))
    return -(A[1]*sum(SemiclassicalJacobiWeight(t, a+1, b, c-1)) + B[1]*sum(SemiclassicalJacobiWeight(t, a, b, c-1)))/(sum(SemiclassicalJacobiWeight(t, a, b, c-1)))
end

# compute α[n+1] vector based on initial conditions. careful, the forward recurrence is unstable
function αforward!(α, t, a, b, c, inds)
    A,B,C = recurrencecoefficients(SemiclassicalJacobi(t,a,b,c))
    @inbounds for n in inds[1:end-1]
       α[n+1] = -t*A[n+1]-B[n+1]-C[n+1]/α[n]
    end
end
# use miller recurrence algorithm to find vector of α, using explicit knowledge of α[1] to fix the normalization
function αmillerbackwards(m::Integer, scale::Integer, P::SemiclassicalJacobi)
    n = scale+m # for now just an arbitrary sufficiently high value >m
    α = zeros(n)
    α[end] = 1
    A,B,C = recurrencecoefficients(P)
    @inbounds for j in reverse(2:n)
       α[j-1] = -C[j]/(α[j]+A[j]*P.t+B[j])
    end
    return (((-(A[1]*P.t^(P.c-1)*gamma(2+P.a)*gamma(1+P.b)/gamma(3+P.a+P.b)*_₂F₁general2(P.a+2,-P.c+1,P.a+3+P.b,1/P.t)/(P.t^(P.c-1)*gamma(1+P.a)gamma(1+P.b)/gamma(2+P.a+P.b)*_₂F₁general2(1+P.a,-P.c+1,2+P.a+P.b,1/P.t))+B[1]))/α[1]).*α)[1:m]
end
function αmillerbackwards(m::Integer, scale::Integer, t, a, b, c)
    n = scale+m # for now just an arbitrary sufficiently high value >m
    α = zeros(n)
    α[end] = 1
    A,B,C = recurrencecoefficients(SemiclassicalJacobi(t,a,b,c))
    @inbounds for j in reverse(2:n)
       α[j-1] = -C[j]/(α[j]+A[j]*t+B[j])
    end
    return (((-(A[1]*t^(c-1)*gamma(2+a)*gamma(1+b)/gamma(3+a+b)*_₂F₁general2(a+2,-c+1,a+3+b,1/t)/(t^(c-1)*gamma(1+a)gamma(1+b)/gamma(2+a+b)*_₂F₁general2(1+a,-c+1,2+a+b,1/t))+B[1]))/α[1]).*α)[1:m]
end
# fill in missing values in an existing vector via miller recurrence guided by indices in inds
function αfillerbackwards!(α, scale::Integer, P::SemiclassicalJacobi, inds)
    oldval = α[inds[1]];
    n = scale+maximum(inds); # for now just an arbitrary sufficiently high value >m
    v = similar(α,n);
    v[end] = 1;
    A,B,C = recurrencecoefficients(P);
    @inbounds for j in reverse(minimum(inds):n)[1:end-1]
       v[j-1] = -C[j]/(v[j]+A[j]*P.t+B[j]);
    end
    α[minimum(inds):maximum(inds)] = (((oldval)/v[inds[1]]).*v[minimum(inds):maximum(inds)]);
    α
end

# cached implementation using stable back recurrence to fill data
mutable struct αforlower{T} <: AbstractCachedVector{T}
    P::SemiclassicalJacobi{T}
    data::Vector{T}
    datasize::Tuple{Int}
    array
    αforlower{T}(P::SemiclassicalJacobi{T}) where T = new{T}(P, αmillerbackwards(28,200,P), (28,))
end
αforlower(P::SemiclassicalJacobi{T}) where T = αforlower{T}(P)
size(::αforlower) = (ℵ₀,)
cache_filldata!(α::αforlower, inds) = αfillerbackwards!(α.data, 200, α.P, inds)

function resizedata!(α::αforlower, nm) 
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

# returns Jacobi operator for (t,a,b,c-1) when input is SemiclassicalJacobi(t,a,b,c)
function lowercjacobimatrix(P::SemiclassicalJacobi)
    a = P.a; b = P.b; c = P.c; t = P.t;
    # we use data taken from the higher basis parameter Jacobi operator
    C,A,B = subdiagonaldata(P.X), diagonaldata(P.X), supdiagonaldata(P.X)
    # compute moment-based coefficients
    α = αforlower(P)
    # compute bands. the first case has to be computed with extra care to fit with our normalization convention
    offD = Vcat(C[1]/(sqrt(sum(SemiclassicalJacobiWeight(t,a,b,c-1)))/sqrt((((α[1]^2-2*α[1]*A[1]/C[1]+A[1]^2/C[1]^2)*sum(SemiclassicalJacobiWeight(t, a, b, c-1))+(2*α[1]/C[1]-2*A[1]/C[1]^2)*sum(SemiclassicalJacobiWeight(t, a+1, b, c-1))))+(1/C[1]^2)*(sum(SemiclassicalJacobiWeight(t, a+2, b, c-1))))), sqrt.(B[1:end].*C[2:end].*α[2:end]./α[1:end]))
    D = Vcat(A[1]-C[1]*α[1], C[1:end].*α[1:end]-C[2:end].*α[2:end]+A[2:end])
    return SymTridiagonal(D,offD)
end

###
# Methods for the special case of computing SemiclassicalJacobi(t,0,0,-1)
###
# As this can be built from SemiclassicalJacobi(t,0,0,0) which is just shifted and normalized Legendre(), we have more explicit methods at our disposal due to explicitly known coefficients for the Legendre bases.

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
    supD = norm.*Vcat(zeros(T,1),-α)
    D = norm
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
    return sqrt(2*acoth(2*t-1))*(-α*jacobip(n-1,0,0,2*x-1)+jacobip(n,0,0,2*x-1))*sqrt(n/(2*α))
end
# Evaluate the n-th non-normalized OP wrt 1/(t-x) at point x
function evalϕn(n::Integer, x, t::T) where T
    # this version recomputes α based on t
    t <= 1 && throw(ArgumentError("t must be greater than 1."))
    n == 0 && return one(T)
    return -αdirect(n,2*t-1)*jacobip(n-1,0,0,2*x-1)+jacobip(n,0,0,2*x-1)
end