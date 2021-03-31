##
# lowering c
# Methods to compute the Jacobi operator for (t,a,b,c-1) based on knowledge of (t,a,b,c)
##

# this alternative version just uses the built-in sums and Q_1 == A[1]x + B[1]
function initialαc_gen(P::SemiclassicalJacobi)
    t = P.t; a = P.a; b = P.b; c = P.c;
    A,B,_ = recurrencecoefficients(SemiclassicalJacobi(t, a, b, c))
    return -(A[1]*sum(SemiclassicalJacobiWeight(t, a+1, b, c-1)) + B[1]*sum(SemiclassicalJacobiWeight(t, a, b, c-1)))/(sum(SemiclassicalJacobiWeight(t, a, b, c-1)))
end
function initialαc_gen(t, a, b, c)
    A,B,_ = recurrencecoefficients(SemiclassicalJacobi(t, a, b, c))
    return -(A[1]*sum(SemiclassicalJacobiWeight(t, a+1, b, c-1)) + B[1]*sum(SemiclassicalJacobiWeight(t, a, b, c-1)))/(sum(SemiclassicalJacobiWeight(t, a, b, c-1)))
end

# compute α[n+1] vector based on initial conditions. careful, the forward recurrence is unstable
function αcforward!(α, t, a, b, c, inds)
    A,B,C = recurrencecoefficients(SemiclassicalJacobi(t, a, b, c))
    @inbounds for n in inds[1:end-1]
       α[n+1] = -t*A[n+1]-B[n+1]-C[n+1]/α[n]
    end
end
# use miller recurrence algorithm to find vector of α, using explicit knowledge of α[1] to fix the normalization
function αcmillerbackwards(m::Integer, scale::Integer, P::SemiclassicalJacobi)
    n = scale+m # for now just an arbitrary sufficiently high value >m
    α = zeros(n)
    α[end] = 1
    A,B,C = recurrencecoefficients(P)
    @inbounds for j in reverse(2:n)
       α[j-1] = -C[j]/(α[j]+A[j]*P.t+B[j])
    end
    return (((-(A[1]*P.t^(P.c-1)*gamma(2+P.a)*gamma(1+P.b)/gamma(3+P.a+P.b)*_₂F₁general2(P.a+2,-P.c+1,P.a+3+P.b,1/P.t)/(P.t^(P.c-1)*gamma(1+P.a)gamma(1+P.b)/gamma(2+P.a+P.b)*_₂F₁general2(1+P.a,-P.c+1,2+P.a+P.b,1/P.t))+B[1]))/α[1]).*α)[1:m]
end
function αcmillerbackwards(m::Integer, scale::Integer, t, a, b, c)
    n = scale+m # for now just an arbitrary sufficiently high value >m
    α = zeros(n)
    α[end] = 1
    A,B,C = recurrencecoefficients(SemiclassicalJacobi(t, a, b, c))
    @inbounds for j in reverse(2:n)
       α[j-1] = -C[j]/(α[j]+A[j]*t+B[j])
    end
    return (((-(A[1]*t^(c-1)*gamma(2+a)*gamma(1+b)/gamma(3+a+b)*_₂F₁general2(a+2,-c+1,a+3+b,1/t)/(t^(c-1)*gamma(1+a)gamma(1+b)/gamma(2+a+b)*_₂F₁general2(1+a,-c+1,2+a+b,1/t))+B[1]))/α[1]).*α)[1:m]
end
# fill in missing values in an existing vector via miller recurrence guided by indices in inds
function αcfillerbackwards!(α, addscale::Integer, mulscale::Integer, P::SemiclassicalJacobi, inds)
    maxI = maximum(inds)
    minI = minimum(inds)
    oldval = α[minI];
    n = addscale+mulscale*maxI; # for now just an arbitrary sufficiently high value >m
    k = 1.;
    A,B,C = recurrencecoefficients(P);
    @inbounds for j in reverse(maxI:n)
        k = -C[j+1]/(k+A[j+1]*P.t+B[j+1]);
    end
    α[end] = k
    @inbounds for j in reverse(minI:maxI)[1:end-1]
       α[j-1] = -C[j]/(α[j]+A[j]*P.t+B[j]);
    end
    α[minI:maxI] = ((oldval)/α[inds[1]]).*α[minI:maxI]
    α
end

# cached implementation using stable back recurrence to fill data
mutable struct CLoweringCoefficients{T} <: AbstractCachedVector{T}
    P::SemiclassicalJacobi{T}
    data::Vector{T}
    datasize::Tuple{Int}
    array
    CLoweringCoefficients{T}(P::SemiclassicalJacobi{T}) where T = new{T}(P, αcmillerbackwards(28, 200, P), (28,))
end
CLoweringCoefficients(P::SemiclassicalJacobi{T}) where T = CLoweringCoefficients{T}(P)
size(::CLoweringCoefficients) = (ℵ₀,)
cache_filldata!(α::CLoweringCoefficients, inds) = αcfillerbackwards!(α.data, 500, 10, α.P, inds)

function getindex(α::CLoweringCoefficients, I::UnitRange)
    resizedata!(α, maximum(I))
    α.data[I]
end

function resizedata!(α::CLoweringCoefficients, nm) 
    νμ = length(α.data)
    if nm > νμ
        olddata = copy(α.data)
        α.data = similar(olddata,maximum(nm))
        α.data[1:νμ] = olddata[1:νμ]
        inds = Array(νμ:maximum(nm))
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
    α = CLoweringCoefficients(P)
    # compute bands. the first case has to be computed with extra care to fit with our normalization convention
    offD = Vcat(C[1]/(sqrt(sum(SemiclassicalJacobiWeight(t,a,b,c-1)))/sqrt((((α[1]^2-2*α[1]*A[1]/C[1]+A[1]^2/C[1]^2)*sum(SemiclassicalJacobiWeight(t, a, b, c-1))+(2*α[1]/C[1]-2*A[1]/C[1]^2)*sum(SemiclassicalJacobiWeight(t, a+1, b, c-1))))+(1/C[1]^2)*(sum(SemiclassicalJacobiWeight(t, a+2, b, c-1))))), sqrt.(B.*C[2:end].*α[2:end]./α))
    D = Vcat(A[1]-C[1]*α[1], C.*α-C[2:end].*α[2:end]+A[2:end])
    return SymTridiagonal(D,offD)
end

##
# lowering b
# Methods to compute the Jacobi operator for (t,a,b-1,c) based on knowledge of (t,a,b,c)
##

function initialαb_gen(P::SemiclassicalJacobi)
    t = P.t; a = P.a; b = P.b; c = P.c;
    A,B,_ = recurrencecoefficients(P)
    return -(A[1]*sum(SemiclassicalJacobiWeight(t, a+1, b-1, c)) + B[1]*sum(SemiclassicalJacobiWeight(t, a, b-1, c)))/(sum(SemiclassicalJacobiWeight(t, a, b-1, c)))
end
function initialαb_gen(t, a, b, c)
    A,B,_ = recurrencecoefficients(SemiclassicalJacobi(t,a,b,c))
    return -(A[1]*sum(SemiclassicalJacobiWeight(t, a+1, b-1, c)) + B[1]*sum(SemiclassicalJacobiWeight(t, a, b-1, c)))/(sum(SemiclassicalJacobiWeight(t, a, b-1, c)))
end
function αbforward!(α, P::SemiclassicalJacobi, inds)
    A,B,C = recurrencecoefficients(P)
    @inbounds for n in inds[1:end-1]
       α[n+1] = -A[n+1]-B[n+1]-C[n+1]/α[n]
    end
end
function αbfillerbackwards!(α, addscale::Integer, mulscale::Integer, P::SemiclassicalJacobi, inds)
    maxI = maximum(inds)
    minI = minimum(inds)
    oldval = α[minI];
    n = addscale+mulscale*maxI; # for now just an arbitrary sufficiently high value >m
    k = 1.;
    A,B,C = recurrencecoefficients(P);
    @inbounds for j in reverse(maxI:n)
        k = -C[j+1]/(k+A[j+1]+B[j+1]);
    end
    α[end] = k
    @inbounds for j in reverse(minI:maxI)[1:end-1]
       α[j-1] = -C[j]/(α[j]+A[j]+B[j]);
    end
    α[minI:maxI] = ((oldval)/α[inds[1]]).*α[minI:maxI]
    α
end

mutable struct BLoweringCoefficients{T} <: AbstractCachedVector{T}
    P::SemiclassicalJacobi{T}
    data::Vector{T}
    datasize::Tuple{Int}
    array
    BLoweringCoefficients{T}(P::SemiclassicalJacobi{T}) where T = new{T}(P, [initialαb_gen(P)], (1,))
end
BLoweringCoefficients(P::SemiclassicalJacobi{T}) where T = BLoweringCoefficients{T}(P)
size(::BLoweringCoefficients) = (ℵ₀,)
cache_filldata!(α::BLoweringCoefficients, inds) = αbfillerbackwards!(α.data, 500, 10, α.P, inds)

function getindex(α::BLoweringCoefficients, I::UnitRange)
    resizedata!(α, maximum(I))
    α.data[I]
end

function resizedata!(α::BLoweringCoefficients, nm) 
    νμ = length(α.data)
    if nm > νμ
        olddata = copy(α.data)
        α.data = similar(olddata,maximum(nm))
        α.data[1:νμ] = olddata[1:νμ]
        inds = Array(νμ:maximum(nm))
        cache_filldata!(α, inds)
        α.datasize = (nm,)
    end
    α
end

# returns Jacobi operator for (t,a,b-1,c) when input is SemiclassicalJacobi(t,a,b,c)
function lowerbjacobimatrix(P::SemiclassicalJacobi)
    a = P.a; b = P.b; c = P.c; t = P.t;
    # we use data taken from the higher basis parameter Jacobi operator
    C,A,B = subdiagonaldata(P.X), diagonaldata(P.X), supdiagonaldata(P.X)
    # compute moment-based coefficients
    α = BLoweringCoefficients(P)
    # compute bands. the first case has to be computed with extra care to fit with our normalization convention
    offD = Vcat(C[1]/(sqrt(sum(SemiclassicalJacobiWeight(t,a,b-1,c)))/sqrt((((α[1]^2-2*α[1]*A[1]/C[1]+A[1]^2/C[1]^2)*sum(SemiclassicalJacobiWeight(t, a, b-1, c))+(2*α[1]/C[1]-2*A[1]/C[1]^2)*sum(SemiclassicalJacobiWeight(t, a+1, b-1, c))))+(1/C[1]^2)*(sum(SemiclassicalJacobiWeight(t, a+2, b-1, c))))), sqrt.(B.*C[2:end].*α[2:end]./α))
    D = Vcat(A[1]-C[1]*α[1], C.*α-C[2:end].*α[2:end]+A[2:end])
    return SymTridiagonal(D,offD)
end


##
# lowering a
# Methods to compute the Jacobi operator for (t,a-1,b,c) based on knowledge of (t,a,b,c)
##
function initialαa_gen(P)
    t = P.t; a = P.a; b = P.b; c = P.c;
    A,B,_ = recurrencecoefficients(P)
    return -(A[1]*sum(SemiclassicalJacobiWeight(t, a, b, c)) + B[1]*sum(SemiclassicalJacobiWeight(t, a-1, b, c)))/(sum(SemiclassicalJacobiWeight(t, a-1, b, c)))
end
function initialαa_gen(t, a, b, c)
    A,B,_ = recurrencecoefficients(SemiclassicalJacobi(t,a,b,c))
    return -(A[1]*sum(SemiclassicalJacobiWeight(t, a, b, c)) + B[1]*sum(SemiclassicalJacobiWeight(t, a-1, b, c)))/(sum(SemiclassicalJacobiWeight(t, a-1, b, c)))
end
function αaforward!(α, P::SemiclassicalJacobi, inds)
    A,B,C = recurrencecoefficients(P)
    @inbounds for n in inds[1:end-1]
       α[n+1] = -B[n+1]-C[n+1]/α[n]
    end
end
function αafillerbackwards!(α, addscale::Integer, mulscale::Integer, P::SemiclassicalJacobi, inds)
    maxI = maximum(inds)
    minI = minimum(inds)
    oldval = α[minI];
    n = addscale+mulscale*maxI; # for now just an arbitrary sufficiently high value >m
    k = 1.;
    A,B,C = recurrencecoefficients(P);
    @inbounds for j in reverse(maxI:n)
        k = -C[j+1]/(k+B[j+1]);
    end
    α[end] = k
    @inbounds for j in reverse(minI:maxI)[1:end-1]
       α[j-1] = -C[j]/(α[j]+B[j]);
    end
    α[minI:maxI] = ((oldval)/α[inds[1]]).*α[minI:maxI]
    α
end

# cached implementation using forward recurrence to fill data
mutable struct ALoweringCoefficients{T} <: AbstractCachedVector{T}
    P::SemiclassicalJacobi{T}
    data::Vector{T}
    datasize::Tuple{Int}
    array
    ALoweringCoefficients{T}(P::SemiclassicalJacobi{T}) where T = new{T}(P, [initialαa_gen(P)], (1,))
end
ALoweringCoefficients(P::SemiclassicalJacobi{T}) where T = ALoweringCoefficients{T}(P)
size(::ALoweringCoefficients) = (ℵ₀,)
cache_filldata!(α::ALoweringCoefficients, inds) = αafillerbackwards!(α.data, 500, 10, α.P, inds)

function getindex(α::ALoweringCoefficients, I::UnitRange)
    resizedata!(α, maximum(I))
    α.data[I]
end

function resizedata!(α::ALoweringCoefficients, nm) 
    νμ = length(α.data)
    if nm > νμ
        olddata = copy(α.data)
        α.data = similar(olddata,maximum(nm))
        α.data[1:νμ] = olddata[1:νμ]
        inds = Array(νμ:maximum(nm))
        cache_filldata!(α, inds)
        α.datasize = (nm,)
    end
    α
end

# returns Jacobi operator for (t,a-1,b,c) when input is SemiclassicalJacobi(t,a,b,c)
function lowerajacobimatrix(P::SemiclassicalJacobi)
    a = P.a; b = P.b; c = P.c; t = P.t;
    # we use data taken from the higher basis parameter Jacobi operator
    C,A,B = subdiagonaldata(P.X), diagonaldata(P.X), supdiagonaldata(P.X)
    # compute moment-based coefficients
    α = ALoweringCoefficients(P)
    α[1000]
    # compute bands. the first case has to be computed with extra care to fit with our normalization convention
    offD = Vcat(C[1]/(sqrt(sum(SemiclassicalJacobiWeight(t,a-1,b,c)))/sqrt((((α[1]^2-2*α[1]*A[1]/C[1]+A[1]^2/C[1]^2)*sum(SemiclassicalJacobiWeight(t, a-1, b, c))+(2*α[1]/C[1]-2*A[1]/C[1]^2)*sum(SemiclassicalJacobiWeight(t, a, b, c))))+(1/C[1]^2)*(sum(SemiclassicalJacobiWeight(t, a+1, b, c))))), sqrt.(B.*C[2:end].*α[2:end]./α))
    D = Vcat(A[1]-C[1]*α[1], C.*α-C[2:end].*α[2:end]+A[2:end])
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
initialα(t) = t-2/(log1p(t)-log(t-1))

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
    α[end] = αdirect(length(α),t)
    @inbounds for n in reverse(inds)[1:end-1]
       α[n-1] = (n-1)/(t*(2*n-1)-n*α[n])
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
    νμ = length(α.data)
    if nm > νμ
        olddata = copy(α.data)
        α.data = similar(olddata,maximum(nm))
        α.data[1:νμ] = olddata[1:νμ]
        inds = Array(νμ:maximum(nm))
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
    neg1c_normconstant{T}(t::T) where T = new{T}(t, neg1c_normconstinitial(t, 10), (10,))
end
neg1c_normconstant(t::T) where T = neg1c_normconstant{T}(t)
size(B::neg1c_normconstant) = (ℵ₀,)
cache_filldata!(B::neg1c_normconstant, inds) = neg1c_normconstextension!(B.data, inds, B.t)

function resizedata!(B::neg1c_normconstant, nm) 
    νμ = length(B.data)
    if nm > νμ
        olddata = copy(B.data)
        B.data = similar(olddata,maximum(nm))
        B.data[1:νμ] = olddata[1:νμ]
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