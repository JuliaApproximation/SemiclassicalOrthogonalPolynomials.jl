# inital value n=0 for α_{n,n-1}(t) coefficients
initialα(t) = t-BigInt(2)/(log1p(t)-log(t-1))

# compute n-th coefficient from direct evaluation formula
αdirect(n,t) = gamma((2*n+1)/BigInt(2))*gamma(n+1)*HypergeometricFunctions._₂F₁general2((1+n)/BigInt(2),(2+n)/BigInt(2),(2*n+3)/BigInt(2),BigInt(1)/t^2)/(t*2*gamma(BigInt(n))*gamma((2*n+3)/BigInt(2))*HypergeometricFunctions._₂F₁general2(n/BigInt(2),(n+1)/BigInt(2),(2*n+1)/BigInt(2),BigInt(1)/t^2))
# this version takes a pre-existing vector v and fills in the missing data guided by indices in inds using explicit formula
function αdirect!(α,t,inds) 
    @inbounds for n in inds
        α[n] = gamma((2*n+1)/BigInt(2))*gamma(1+n)*HypergeometricFunctions._₂F₁general2((1+n)/BigInt(2),(2+n)/BigInt(2),(2*n+3)/BigInt(2),1/t^2)/(t*2*gamma(n)*gamma((2*n+3)/BigInt(2))*HypergeometricFunctions._₂F₁general2(n/BigInt(2),(n+1)/BigInt(2),(2*n+1)/BigInt(2),1/t^2))
    end
end

# takes a previously computed vector of α_{n,n-1}(t) that has been increased in size and fills in the missing data guided by indices in inds via forward recurrence
function αcoefficients!(α,t,inds)
    @inbounds for n in inds
       α[n] = (t*(2*n-1)/n-(n-1)/(n*α[n-1]))
    end
end

# takes a previously computed vector of α_{n,n-1}(t) that has been increased in size and fills in the missing data guided by indices in inds, using a final condition computation and backward recurrence
function backαcoeff!(α,t,inds)
    α[end] = αdirect(BigInt(length(α)),t)
    @inbounds for n in reverse(inds)
       α[n-1] = BigInt(n-1)/(t*(2*n-1)-n*α[n]) #BigInt(1-n)/(t-2*BigInt(n)*t+BigInt(n)*α[n])
    end
end

# Evaluate the n-th OP wrt 1/(t-x) at point x, with one of two methods:
# either with or without recomputing the α coefficients.
function evalϕn(n::Integer,x,α::AbstractArray)
    # this version accepts a vector of coefficients of appropriate length to avoid recomputing α
    n == 0 && return 2
    return α[n]*ClassicalOrthogonalPolynomials.jacobip(n-1,0,0,-x)+ClassicalOrthogonalPolynomials.jacobip(n,0,0,-x)
end
function evalϕn(n::Integer,x,t::Real)
    # this version recomputes α based on t
    t <= 1 && error("t must be greater than 1.")
    n == 0 && return 2
    return αdirect(n,t)*ClassicalOrthogonalPolynomials.jacobip(n-1,0,0,-x)+ClassicalOrthogonalPolynomials.jacobip(n,0,0,-x)
end

# jacobimatrix for OP wrt 1/(t-x)
mutable struct JacobiMatrixM1{T} <: AbstractCachedMatrix{T}
    t::T
    data
    datasize::Tuple{Int,Int}
    array
    JacobiMatrixM1{T}(t::T) where T = new{T}(t, initialjacobi(t,10), (10,10))
end
JacobiMatrixM1(t::T) where T = JacobiMatrixM1{T}(t)
size(K::JacobiMatrixM1) = (∞,∞)

# data filling
cache_filldata!(J::JacobiMatrixM1, inds) = jacobiopm1extension!(J.data, inds, J.t)

# force square-shaped LazyArrays caching and resizing
function getindex(J::JacobiMatrixM1{T}, I::CartesianIndex) where T
    resizedata!(J, Tuple(I))
    J.data[I]
end
function getindex(J::JacobiMatrixM1{T}, I::Vararg{Int,2}) where T
    resizedata!(J, Tuple([I...]))
    getindex(J.data,I...)
end
function getindex(J::JacobiMatrixM1{T}, I::Vararg{UnitRange,2}) where T
    resizedata!(J, (maximum(I[1]),maximum(I[2])))
    view(J.data,I[1],I[2])
end

function resizedata!(J::JacobiMatrixM1, nm) 
    olddata = J.data
    νμ = size(olddata)
    nm = (maximum(nm),maximum(nm))
    nm = max.(νμ,nm)
    nm = (maximum(nm),maximum(nm))
    if νμ ≠ nm
        J.data = similar(olddata,maximum(nm),maximum(nm))
        for j = 1:maximum(νμ)
            for ii = 1:maximum(νμ)
                if (ii in [j-1,j,j+1])
                    J.data[j,ii] = olddata[j,ii]
                end
            end
        end
    end
    if maximum(nm) > maximum(νμ)
        inds = Array(maximum(νμ)-1:maximum(nm))
        cache_filldata!(J, inds)
        J.datasize = nm
    end
    J
end

function jacobiopm1extension!(J,inds,t)
    n = BigInt(maximum(inds))
    m = BigInt(minimum(inds))
    α0 = zeros(BigFloat, n+2)
    backαcoeff!(α0,t,BigInt.(m:n+2))
    # build bands
        # subdiagonal
        N = (BigInt(m-1):BigInt(n-1))
        SubD = (-1)*(N.+1)./(2 .*N.+1)
        # diagonal
        N = (BigInt(m-1):BigInt(n-1))
        D = (-1).*(N./(2 .*N.-1).*α0[N]-(N.+1)./(2 .*N.+1).*α0[N.+1])
        # superdiagonal
        N = (BigInt(m):BigInt(n-1))
        SuperD = [BigFloat("0")]
        append!(SuperD,(-1)*(N.-1)./(2 .*N.-1).*α0[N]./α0[N.-1])
    newdata = (BandedMatrices._BandedMatrix(Vcat(SuperD',D',SubD'),(1:length(m:n)),1,1))
    for j in inds
        for ii in inds
            if (ii in [j-1,j,j+1])
                 J[ii,j] = newdata[ii-m+1,j-m+1]
            end
        end
    end
    J
end

# build the first nxn block of the tridiagonal Jacobi matrix for OPs wrt 1/(t-x)
function initialjacobi(t,n)
    # build coefficients
    α0 = zeros(BigFloat, n+2)
    backαcoeff!(α0,t,BigInt.(2:n+2))
    # build bands
        # subdiagonal
        N = (BigInt(0):BigInt(n-1))
        SubD = (-1)*(N.+1)./(2 .*N.+1)
        # diagonal
        N = (BigInt(1):BigInt(n-1))
        D = [α0[1]]
        append!(D,(-1).*(N./(2 .*N.-1).*α0[N]-(N.+1)./(2 .*N.+1).*α0[N.+1]))
        # superdiagonal
        N = (BigInt(2):BigInt(n-1))
        SuperD = [BigFloat("0"),(3*α0[1]^2-2*α0[1]*α0[2]-1)/BigInt(3)]
        append!(SuperD,(-1)*(N.-1)./(2 .*N.-1).*α0[N]./α0[N.-1])
    # build operator
    return BandedMatrices._BandedMatrix(Vcat(SuperD',D',SubD'),(1:n),1,1)[1:n,1:n]
end

# multiply to convert from OPs wrt 1/(t-x) to Legendre. Use \ to convert from Legendre to OPs wrt 1/(t-x).
function converttolaplace(t,N)
    α = zeros(BigFloat,N-1)
    α[1] = initialα(t)
    backαcoeff!(α,t,BigInt.(2:N-1))
    α0 = [BigFloat("0")]
    append!(α0,α)
    return BandedMatrices._BandedMatrix(Vcat((BigFloat("-1")).^(1:N)' .* α0',(-1).^(0:N-1)'), N, 0,1)
end