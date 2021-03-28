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
function αmillerbackwards(m, scale, P::SemiclassicalJacobi)
    n = scale*m # for now just an arbitrary sufficiently high value >m, can make this more sophisticated if needed
    α = zeros(n)
    α[end] = 1
    A,B,C = recurrencecoefficients(P)
    @inbounds for j in reverse(2:n)
       α[j-1] = -C[j]/(α[j]+A[j]*P.t+B[j])
    end
    return (((-(A[1]*P.t^(P.c-1)*gamma(2+P.a)*gamma(1+P.b)/gamma(3+P.a+P.b)*_₂F₁general2(P.a+2,-P.c+1,P.a+3+P.b,1/P.t)/(P.t^(P.c-1)*gamma(1+P.a)gamma(1+P.b)/gamma(2+P.a+P.b)*_₂F₁general2(1+P.a,-P.c+1,2+P.a+P.b,1/P.t))+B[1]))/α[1]).*α)[1:m]
end
function αmillerbackwards(m, scale, t, a, b, c)
    n = scale*m # for now just an arbitrary sufficiently high value >m, can make this more sophisticated if needed
    α = zeros(n)
    α[end] = 1
    A,B,C = recurrencecoefficients(SemiclassicalJacobi(t,a,b,c))
    @inbounds for j in reverse(2:n)
       α[j-1] = -C[j]/(α[j]+A[j]*t+B[j])
    end
    return (((-(A[1]*t^(c-1)*gamma(2+a)*gamma(1+b)/gamma(3+a+b)*_₂F₁general2(a+2,-c+1,a+3+b,1/t)/(t^(c-1)*gamma(1+a)gamma(1+b)/gamma(2+a+b)*_₂F₁general2(1+a,-c+1,2+a+b,1/t))+B[1]))/α[1]).*α)[1:m]
end
# fill in missing values in an existing vector via miller backwards guided by indices in inds
function αfillerbackwards!(α, scale, P::SemiclassicalJacobi, inds)
    oldval = α[inds[1]];
    n = scale*maximum(inds); # for now just an arbitrary sufficiently high value >m, can make this more sophisticated if needed
    v = similar(α,n);
    v[end] = 1;
    A,B,C = recurrencecoefficients(P);
    @inbounds for j in reverse(minimum(inds):n)[1:end-1]
       v[j-1] = -C[j]/(v[j]+A[j]*P.t+B[j]);
    end
    v = (((oldval)/v[inds[1]]).*v[minimum(inds):maximum(inds)]);
    α[minimum(inds):maximum(inds)] = v;
    α
end

# cached implementation using stable back recurrence to fill data
mutable struct αforlower{T} <: AbstractCachedVector{T}
    P::SemiclassicalJacobi{T}
    data::Vector{T}
    datasize::Tuple{Int}
    array
    αforlower{T}(P::SemiclassicalJacobi{T}) where T = new{T}(P, αmillerbackwards(20,10,P), (20,))
end
αforlower(P::SemiclassicalJacobi{T}) where T = αforlower{T}(P)
size(::αforlower) = (ℵ₀,)
cache_filldata!(α::αforlower, inds) = αfillerbackwards!(α.data, 10, α.P, inds)

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