# inital value n=0 for α_{n,n-1}(t) coefficients
initialα(t) = 2*(t*acoth(t)-1)/(log1p(t)-log(t-1))

# takes a previously computed vector of α_{n,n-1}(t) that has been increased in size and fills in the missing data guided by indices in inds
function αcoefficients!(α,t,inds)
    @inbounds for n in inds
       α[n] = (t*(2*n-1)-(n-1)/α[n-1])/n
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
    α = zeros(n+1)'
    α[1] = initialα(t)
    αcoefficients!(α,t,2:n)
    return α[n]*ClassicalOrthogonalPolynomials.jacobip(n-1,0,0,-x)+ClassicalOrthogonalPolynomials.jacobip(n,0,0,-x)
end