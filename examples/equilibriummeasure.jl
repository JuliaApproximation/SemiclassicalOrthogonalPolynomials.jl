using SemiclassicalOrthogonalPolynomials, ClassicalOrthogonalPolynomials, ForwardDiff, Plots, StaticArrays
import ForwardDiff: derivative, jacobian, Dual
import SemiclassicalOrthogonalPolynomials: Weighted
import ClassicalOrthogonalPolynomials: associated, affine

Base.floatmin(::Type{Dual{T,V,N}}) where {T,V,N} = Dual{T,V,N}(floatmin(V))
Base.big(d::Dual{T,V,N}) where {T,V,N} = Dual{T}(big(d.value), ForwardDiff.Partials(map(big,d.partials.values)))

#### 
#
# We first do a single interval.
# Equilibrium measures for a symmetric potential
#  with one interval of support 
# a measure w(x) supported on
# [-b,b]
# such that
#       1. H*w == V'
#       2. sum(w) == 1
#       3.  w is bounded
# 
#  rescaling x == b*t these becomes find 
# a measure w̃(t) supported on
# [-1,1]
# such that
#       1. H*w̃ == V'(b*t)
#       2. sum(w̃) == 1/b
#       3.  w is bounded
#
# Note (1) and (2) can always be satisfied
# thus the constraint comes from (3).
# The following gives the evaluation of the
# unweighted-component of the measure evaluated
# at (a,b)
#####


V = x -> x^2
function equilibriumcoefficients(T, b)
    U = associated(T)
    W = Weighted(T)
    t = axes(W,1)
    H = @. inv(t - t')
    H̃ = U \H*W
    [1/(b*sum(W[:,1])); 2H̃[:,2:end] \ ( U \ derivative.(V, b*t))]
end
function equilibriumendpointvalue(b::Number)
    T = ChebyshevT{typeof(b)}()
    dot(T[end,:], equilibriumcoefficients(T,b))
end

function equilibrium(b::Number)
    T = ChebyshevT{typeof(b)}()
    U = ChebyshevU{typeof(b)}()
    # convert to Weighted(U) to make value at ±b accurate
    Weighted(U)[affine(-b..b,axes(T,1)),:] * ((Weighted(T) \ Weighted(U))[3:end,:] \ equilibriumcoefficients(T,b)[3:end])
end

μ = equilibrium(sqrt(2))

T = Chebyshev()
b = sqrt(2)
μ = Weighted(T) * equilibriumcoefficients(T, b)
x = axes(μ,1)

plot(μ)

xx = 0.7; 2b*(log.(abs.(x .- x'))*μ)[xx] - V(b*xx)



b = 1.0 # initial guess
for _ = 1:10
    b -= derivative(equilibriumendpointvalue,b) \ equilibriumendpointvalue(b)
end
b

plot(equilibrium(b))


#####
# Equilibrium measures for a symmetric potential 
# with two intervals of support consists of finding 
# a measure w(x) supported on
# [-b,-a] ∪ [a,b]
# such that
#       1. H*w == V'
#       2. sum(w) == 1
#       3.  w is bounded
# 
#  rescaling x == b*t these becomes find 
# a measure w̃(t) supported on
# [-1,-a/b] ∪ [a/b,1]
# such that
#       1. H*w̃ == V'(b*t)
#       2. sum(w̃) == 1/b
#       3.  w is bounded
#
# Note (1) and (2) can always be satisfied
# thus the two constraints come from (3).
# The following gives the evaluation of the
# unweighted-component of the measure evaluated
# at (a,b)
#####
V = x -> x^4 - 10x^2
function equilibriumcoefficients(P,a,b)
    W = Weighted(P)
    Q = associated(P)
    t = axes(W,1)
    x = axes(Q,1)
    H = @. inv(x - t')
    H̃ = Q \ H*W
    [1/(b*sum(W[:,1])); 2H̃[:,2:end] \( Q \ derivative.(V, b*x))]
end
function equilibriumendpointvalues(ab::SVector{2})
    a,b = ab
    # orthogonal polynomials w.r.t.
    # abs(x) / (sqrt(1-x^2) * sqrt(x^2 - ρ^2))
    P = TwoBandJacobi(a/b, -one(a)/2, -one(a)/2, one(a)/2)
    Vector(P[[a/b,1],:] * equilibriumcoefficients(P,a,b))
end

function equilibrium(ab)
    a,b = ab
    P = TwoBandJacobi(a/b, -1/2, -1/2, 1/2)
    Weighted(P) * equilibriumcoefficients(P,a,b)
end

ab = SVector(2.,3.)
ab -= jacobian(equilibriumendpointvalues,ab) \ equilibriumendpointvalues(ab)
a,b = ab
xx = range(-4,4;length=1000)
μ = equilibrium(ab)
μx = x -> a < abs(x) < b ? μ[x/b] : 0.0
plot!(xx, μx.(xx))

plot(equilibrium(ab))

