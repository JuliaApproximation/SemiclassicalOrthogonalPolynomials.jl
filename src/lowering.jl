###########
# Generic methods for obtaining Jacobi matrices with one of the a, b and c parameters lowered by 1.
# passing symbols :a, :b or :c into lowindex determines which parameter is lowered
######
function initialα_gen(P::SemiclassicalJacobi, lowindex::Symbol)
    t = P.t; a = P.a; b = P.b; c = P.c;
    A,B,_ = recurrencecoefficients(P)
    # this makes use of Q_1 == A[1]x + B[1]
    if lowindex == :a
        return -(A[1]*sum(SemiclassicalJacobiWeight(t, a, b, c)) + B[1]*sum(SemiclassicalJacobiWeight(t, a-1, b, c)))/(sum(SemiclassicalJacobiWeight(t, a-1, b, c)))
    elseif lowindex == :b
        return -(A[1]*sum(SemiclassicalJacobiWeight(t, a+1, b-1, c)) + B[1]*sum(SemiclassicalJacobiWeight(t, a, b-1, c)))/(sum(SemiclassicalJacobiWeight(t, a, b-1, c)))
    else # lowindex == :c
        return -(A[1]*sum(SemiclassicalJacobiWeight(t, a+1, b, c-1)) + B[1]*sum(SemiclassicalJacobiWeight(t, a, b, c-1)))/(sum(SemiclassicalJacobiWeight(t, a, b, c-1)))
    end
end

mutable struct LoweredJacobiMatrix{T} <: AbstractCachedMatrix{T}
    P::SemiclassicalJacobi{T}
    lowindex::Symbol # options for lowering are :a, :b and :c
    data::Vector{T}
    datasize::Tuple{Int,Int}
    array
    LoweredJacobiMatrix{T}(P::SemiclassicalJacobi{T}, lowindex::Symbol) where T = new{T}(P, lowindex, [initialα_gen(P, lowindex)], (1,1))
end
LoweredJacobiMatrix(P::SemiclassicalJacobi{T}, lowindex::Symbol) where T = LoweredJacobiMatrix{T}(P, lowindex)
size(::LoweredJacobiMatrix) = (ℵ₀,ℵ₀)
cache_filldata!(J::LoweredJacobiMatrix, inds) =  αgenfillerbackwards!(J.data, 1000, 2, J.P, J.lowindex, inds)
MemoryLayout(J::LoweredJacobiMatrix) = MemoryLayout(J.P.X)

function αgenfillerbackwards!(α::Vector{T}, addscale::Int, mulscale::Int, P::SemiclassicalJacobi{T}, lowindex::Symbol, inds::UnitRange{Int64}) where T
    maxI::Int = maximum(inds)
    minI::Int = minimum(inds)
    oldval = α[minI];
    n = addscale+mulscale*maxI; # for now just an arbitrary sufficiently high value >m
    k = one(T);
    A,B,C = recurrencecoefficients(P)
    if lowindex == :a
        @inbounds for j = n:-1:maxI
            k = -C[j+1]/(k+B[j+1]);
        end
        α[maxI] = k
        @inbounds for j = maxI:-1:minI+1
            α[j-1] = -C[j]/(α[j]+B[j]);
        end
    elseif lowindex == :b
        @inbounds for j = n:-1:maxI
            k = -C[j+1]/(k+A[j+1]+B[j+1]);
        end
        α[maxI] = k
        @inbounds for j = maxI:-1:minI+1
            α[j-1] = -C[j]/(α[j]+A[j]+B[j]);
        end
    else # lowindex == :c
        @inbounds for j = n:-1:maxI
            k = -C[j+1]/(k+A[j+1]*P.t+B[j+1]);
        end
        α[maxI] = k
        @inbounds for j = maxI:-1:minI+1
            α[j-1] = -C[j]/(α[j]+A[j]*P.t+B[j]);
        end
    end
    α[minI:maxI] = ((oldval)/α[inds[1]]).*α[minI:maxI]
    α
end

function getindex(J::LoweredJacobiMatrix, I::Vararg{Int,2})
    # check if resize is necessary
    nm = maximum(I)
    resizedata!(J, nm+1)
    # prepare data
    a = J.P.a; b = J.P.b; c = J.P.c; t = J.P.t;
    C,A,B = subdiagonaldata(J.P.X), diagonaldata(J.P.X), supdiagonaldata(J.P.X)
    # generate off-diagonal bands
    if J.lowindex == :a
        offD = Vcat(C[1]/(sqrt(sum(SemiclassicalJacobiWeight(t,a-1,b,c)))/sqrt((((J.data[1]^2-2*J.data[1]*A[1]/C[1]+A[1]^2/C[1]^2)*sum(SemiclassicalJacobiWeight(t, a-1, b, c))+(2*J.data[1]/C[1]-2*A[1]/C[1]^2)*sum(SemiclassicalJacobiWeight(t, a, b, c))))+(1/C[1]^2)*(sum(SemiclassicalJacobiWeight(t, a+1, b, c))))), sqrt.(B[1:nm].*C[2:nm+1].*J.data[2:nm+1]./J.data[1:nm]))
    elseif J.lowindex == :b
        offD = Vcat(C[1]/(sqrt(sum(SemiclassicalJacobiWeight(t,a,b-1,c)))/sqrt((((J.data[1]^2-2*J.data[1]*A[1]/C[1]+A[1]^2/C[1]^2)*sum(SemiclassicalJacobiWeight(t, a, b-1, c))+(2*J.data[1]/C[1]-2*A[1]/C[1]^2)*sum(SemiclassicalJacobiWeight(t, a+1, b-1, c))))+(1/C[1]^2)*(sum(SemiclassicalJacobiWeight(t, a+2, b-1, c))))), sqrt.(B[1:nm].*C[2:nm+1].*J.data[2:nm+1]./J.data[1:nm]))
    else # J.lowindex == :c
        offD = Vcat(C[1]/(sqrt(sum(SemiclassicalJacobiWeight(t,a,b,c-1)))/sqrt((((J.data[1]^2-2*J.data[1]*A[1]/C[1]+A[1]^2/C[1]^2)*sum(SemiclassicalJacobiWeight(t, a, b, c-1))+(2*J.data[1]/C[1]-2*A[1]/C[1]^2)*sum(SemiclassicalJacobiWeight(t, a+1, b, c-1))))+(1/C[1]^2)*(sum(SemiclassicalJacobiWeight(t, a+2, b, c-1))))), sqrt.(B[1:nm].*C[2:nm+1].*J.data[2:nm+1]./J.data[1:nm]))
    end
    # generate diagonal
    D = Vcat(A[1]-C[1]*J.data[1], C[1:nm].*J.data[1:nm]-C[2:nm+1].*J.data[2:nm+1]+A[2:nm+1])
    # return operator
    return SymTridiagonal(D,offD)[I[1],I[2]]
end

function getindex(J::LoweredJacobiMatrix, I::Vararg{UnitRange,2})
    # check if resize is necessary
    nm = maximum(maximum.(I))
    resizedata!(J, nm+1)
    # prepare data
    a = J.P.a; b = J.P.b; c = J.P.c; t = J.P.t;
    C,A,B = subdiagonaldata(J.P.X), diagonaldata(J.P.X), supdiagonaldata(J.P.X)
    # generate off-diagonal bands
    if J.lowindex == :a
        offD = Vcat(C[1]/(sqrt(sum(SemiclassicalJacobiWeight(t,a-1,b,c)))/sqrt((((J.data[1]^2-2*J.data[1]*A[1]/C[1]+A[1]^2/C[1]^2)*sum(SemiclassicalJacobiWeight(t, a-1, b, c))+(2*J.data[1]/C[1]-2*A[1]/C[1]^2)*sum(SemiclassicalJacobiWeight(t, a, b, c))))+(1/C[1]^2)*(sum(SemiclassicalJacobiWeight(t, a+1, b, c))))), sqrt.(B[1:nm].*C[2:nm+1].*J.data[2:nm+1]./J.data[1:nm]))
    elseif J.lowindex == :b
        offD = Vcat(C[1]/(sqrt(sum(SemiclassicalJacobiWeight(t,a,b-1,c)))/sqrt((((J.data[1]^2-2*J.data[1]*A[1]/C[1]+A[1]^2/C[1]^2)*sum(SemiclassicalJacobiWeight(t, a, b-1, c))+(2*J.data[1]/C[1]-2*A[1]/C[1]^2)*sum(SemiclassicalJacobiWeight(t, a+1, b-1, c))))+(1/C[1]^2)*(sum(SemiclassicalJacobiWeight(t, a+2, b-1, c))))), sqrt.(B[1:nm].*C[2:nm+1].*J.data[2:nm+1]./J.data[1:nm]))
    else # J.lowindex == :c
        offD = Vcat(C[1]/(sqrt(sum(SemiclassicalJacobiWeight(t,a,b,c-1)))/sqrt((((J.data[1]^2-2*J.data[1]*A[1]/C[1]+A[1]^2/C[1]^2)*sum(SemiclassicalJacobiWeight(t, a, b, c-1))+(2*J.data[1]/C[1]-2*A[1]/C[1]^2)*sum(SemiclassicalJacobiWeight(t, a+1, b, c-1))))+(1/C[1]^2)*(sum(SemiclassicalJacobiWeight(t, a+2, b, c-1))))), sqrt.(B[1:nm].*C[2:nm+1].*J.data[2:nm+1]./J.data[1:nm]))
    end
    # generate diagonal
    D = Vcat(A[1]-C[1]*J.data[1], C[1:nm].*J.data[1:nm]-C[2:nm+1].*J.data[2:nm+1]+A[2:nm+1])
    return SymTridiagonal(D,offD)[I[1],I[2]]
end

function resizedata!(J::LoweredJacobiMatrix, nm) 
    νμ = length(J.data)
    if nm > νμ
        olddata = copy(J.data)
        J.data = similar(olddata,maximum(nm))
        J.data[1:νμ] = olddata[1:νμ]
        inds = νμ:maximum(nm)
        cache_filldata!(J, inds)
        J.datasize = (nm,nm)
    end
    J
end

###########
# Methods for the special case of computing SemiclassicalJacobi(t,0,0,-1)
######
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
