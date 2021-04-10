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

mutable struct LoweringCoefficients{T} <: AbstractCachedVector{T}
    P::SemiclassicalJacobi{T}
    lowind::Symbol # options for lowering are :a, :b and :c
    data::Vector{T}
    datasize::Tuple{Int}
    array
    LoweringCoefficients{T}(P::SemiclassicalJacobi{T}, lowindex::Symbol) where T = new{T}(P, lowindex, [initialα_gen(P, lowindex)], (1,))
end
LoweringCoefficients(P::SemiclassicalJacobi{T}, lowindex::Symbol) where T = LoweringCoefficients{T}(P, lowindex)
size(::LoweringCoefficients) = (ℵ₀,)

function getindex(α::LoweringCoefficients, I::UnitRange{Int64})
    nm = maximum(I)
    resizedata!(α, nm)
    return getindex(α.data,I)
end

function getindex(α::LoweringCoefficients, I::Int64)
    resizedata!(α, I)
    return getindex(α.data,I)
end

function resizedata!(α::LoweringCoefficients, nm) 
    νμ = length(α.data)
    if nm > νμ
        olddata = copy(α.data)
        nmax = maximum(nm)
        α.data = similar(olddata,nmax)
        α.data[1:νμ] = olddata[1:νμ]
        inds = νμ:nmax
        cache_filldata!(α, inds)
        α.datasize = (nmax,)
    end
    return α
end
cache_filldata!(α::LoweringCoefficients, inds) =  αgenfillerbackwards!(α.data, 100, α.P, α.lowind, inds)

# fill in coefficients via Miller recurrence
function αgenfillerbackwards!(α::Vector{T}, addscale::Int, P::SemiclassicalJacobi{T}, lowindex::Symbol, inds::UnitRange{Int}) where T
    maxI::Int = maximum(inds)
    minI::Int = minimum(inds)
    oldval = α[minI]
    n = maxI
    k = one(T)
    k2 = zero(T)
    A,B,C = recurrencecoefficients(P)
    if lowindex == :a
        while abs(k2-k) > 1e-10
            k2, k = k, one(T)
            n += addscale
            @inbounds for j = n:-1:maxI
                k = -C[j+1]/(k+B[j+1])
            end
        end
        α[maxI] = k
        @inbounds for j = maxI:-1:minI+1
            α[j-1] = -C[j]/(α[j]+B[j])
        end
    elseif lowindex == :b
        while abs(k2-k) > 1e-10
            k2, k = k, one(T)
            n += addscale
            @inbounds for j = n:-1:maxI
                k = -C[j+1]/(k+A[j+1]+B[j+1])
            end
        end
        α[maxI] = k
        @inbounds for j = maxI:-1:minI+1
            α[j-1] = -C[j]/(α[j]+A[j]+B[j])
        end
    else # lowindex == :c
        while abs(k2-k) > 1e-10
            k2, k = k, one(T)
            n += addscale
            @inbounds for j = n:-1:maxI
                k = -C[j+1]/(k+A[j+1]*P.t+B[j+1])
            end
        end
        α[maxI] = k
        @inbounds for j = maxI:-1:minI+1
            α[j-1] = -C[j]/(α[j]+A[j]*P.t+B[j])
        end
    end
    α[minI:maxI] = ((oldval)/α[inds[1]]).*α[minI:maxI]
    α
end

function symlowered_jacobimatrix(Q::SemiclassicalJacobi, lowindex::Symbol)
    bands = SemiclassicalLoweredJacobiBands(Q,lowindex)
    return SymTridiagonal(bands[1,:],bands[2,:])
end

mutable struct SemiclassicalLoweredJacobiBands{T} <: AbstractCachedMatrix{T}
    P::SemiclassicalJacobi{T}
    data::Array{T} # to avoid redundant re-computations, the bands are stored as a (∞,2)-array
    αcfs::AbstractCachedVector{T}
    datasize::Tuple{Int,Int}
end
size(r::SemiclassicalLoweredJacobiBands) = (2,ℵ₀)

function SemiclassicalLoweredJacobiBands(P::SemiclassicalJacobi{T}, lowindex::Symbol) where T
    a,b,c,t = P.a, P.b, P.c, P.t
    cachedα = LoweringCoefficients(P,lowindex)
    αcfs = cachedα[1]
    C, A = P.X[2,1], P.X[1,1]
    if lowindex == :a
        SemiclassicalLoweredJacobiBands{T}(P, [A-C*αcfs, C/(sqrt(sum(SemiclassicalJacobiWeight(t,a-1,b,c)))/sqrt((((αcfs^2-2*αcfs*A/C+A^2/C^2)*sum(SemiclassicalJacobiWeight(t, a-1, b, c))+(2*αcfs/C-2*A/C^2)*sum(SemiclassicalJacobiWeight(t, a, b, c))))+(1/C^2)*(sum(SemiclassicalJacobiWeight(t, a+1, b, c)))))], cachedα, (2,1))
    elseif lowindex == :b
        SemiclassicalLoweredJacobiBands{T}(P, [A-C*αcfs, C/(sqrt(sum(SemiclassicalJacobiWeight(t,a,b-1,c)))/sqrt((((αcfs^2-2*αcfs*A/C+A^2/C^2)*sum(SemiclassicalJacobiWeight(t, a, b-1, c))+(2*αcfs/C-2*A/C^2)*sum(SemiclassicalJacobiWeight(t, a+1, b-1, c))))+(1/C^2)*(sum(SemiclassicalJacobiWeight(t, a+2, b-1, c)))))], cachedα, (2,1))
    elseif lowindex == :c
        SemiclassicalLoweredJacobiBands{T}(P, [A-C*αcfs, C/(sqrt(sum(SemiclassicalJacobiWeight(t,a,b,c-1)))/sqrt((((αcfs^2-2*αcfs*A/C+A^2/C^2)*sum(SemiclassicalJacobiWeight(t, a, b, c-1))+(2*αcfs/C-2*A/C^2)*sum(SemiclassicalJacobiWeight(t, a+1, b, c-1))))+(1/C^2)*(sum(SemiclassicalJacobiWeight(t, a+2, b, c-1)))))], cachedα, (2,1))
    end
end

function getindex(r::SemiclassicalLoweredJacobiBands, I::Vararg{Int,2})
    r.αcfs[maximum(I)+1] # expanding the cache of coefficients here prevents redundant recomputation later
    resizedata!(r,I)
    getindex(r.data,I[1],I[2])
end
function getindex(r::SemiclassicalLoweredJacobiBands, I::Int, J::UnitRange{Int})
    r.αcfs[maximum(J)+1] # expanding the cache of coefficients here prevents redundant recomputation later
    resizedata!(r,J)
    getindex(r.data,I,J)
end
function resizedata!(r::SemiclassicalLoweredJacobiBands, nm) 
    νμ = length(r.data[1,:])
    nmax = maximum(nm)
    if nmax > νμ
        olddata = copy(r.data)
        r.data = similar(olddata,2,nmax)
        r.data[1:2,1:νμ] = olddata
        inds = νμ:nmax
        cache_filldata!(r, inds)
        r.datasize = (2,nmax)
    end
    return r
end
cache_filldata!(r::SemiclassicalLoweredJacobiBands, inds) =  loweringjacobibandfill!(r.data, r.αcfs, r.P, inds)

function loweringjacobibandfill!(data, αcfs, P::SemiclassicalJacobi, inds)
    lowI = minimum(inds):maximum(inds)-1
    shiftI = minimum(inds)+1:maximum(inds)
    C,A = subdiagonaldata(P.X), diagonaldata(P.X)
    data[1,shiftI] = -C[shiftI].*αcfs[shiftI]+C[lowI].*αcfs[lowI]+A[shiftI]
    data[2,shiftI] = sqrt.(C[lowI].*C[shiftI].*αcfs[shiftI]./αcfs[lowI])
    data
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
αdirect(n, t) = 2*n*_₂F₁general2((1+n)/2,(n+2)/2,(2*n+3)/2,1/t^2)/(t*2*(1+2*n)*_₂F₁general2(n/2,(n+1)/2,(2*n+1)/2,1/t^2))
# this version takes a pre-existing vector v and fills in the missing data guided by indices in inds using explicit formula
function αdirect!(α, t, inds) 
    @inbounds for n in inds
        α[n] = 2*n*_₂F₁general2((1+n)/2,(2+n)/2,(2*n+3)/2,1/t^2)/(t*2*(1+2*n)*_₂F₁general2(n/2,(n+1)/2,(2*n+1)/2,1/t^2))
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

function neg1c_normconstinitial(t::T, N::Integer) where T
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