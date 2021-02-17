# inital value n=0 for α_{n,n-1}(t) coefficients
initialα(t) = t-2/(log1p(t)-log(t-1))

# compute n-th coefficient from direct evaluation formula
αdirect(n,t) = gamma((2*n+1)/2)*gamma(1+n)*_₂F₁((1+n)/2,(2+n)/2,(2n+3)/2,1/t^2)/(2*t*gamma(n)*gamma((2*n+3)/2)*_₂F₁(n/2,(n+1)/2,(2*n+1)/2,1/t^2))
# this version takes a pre-existing vector v and fills in the missing data guided by indices in inds using explicit formula
function αdirect!(α,t,inds) 
    @inbounds for n in inds
        α[n] = gamma((2*n+1)/2)*gamma(1+n)*_₂F₁((1+n)/2,(2+n)/2,(2n+3)/2,1/t^2)/(2*t*gamma(n)*gamma((2*n+3)/2)*_₂F₁(n/2,(n+1)/2,(2*n+1)/2,1/t^2))
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
       α[n-1] = (1-n)/(t-2*n*t+n*α[n])
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