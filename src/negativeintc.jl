# inital value n=0 for α_{n,n-1}(t) coefficients
initialα(t) = 2*(t*acoth(t)-1)/(log1p(t)-log(t-1))

# takes a previously computed vector of α_{n,n-1}(t) that has been increased in size and fills in the missing data guided by indices in inds
function αcoefficients!(α,t,inds)
    @inbounds for n in inds
       α[n] = (t*(2*n-1)-(n-1)/α[n-1])/n
    end
end