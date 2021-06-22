using SemiclassicalOrthogonalPolynomials, ClassicalOrthogonalPolynomials, ForwardDiff, Plots, StaticArrays
import ForwardDiff: derivative, jacobian, Dual
import SemiclassicalOrthogonalPolynomials: Weighted
import ClassicalOrthogonalPolynomials: associated

Base.floatmin(::Type{Dual{T,V,N}}) where {T,V,N} = Dual{T,V,N}(floatmin(V))


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
    H̃ = U \ (H*W)
    [1/(b*sum(W[:,1])); H̃[:,2:end] \( U \ derivative.(V, b*t))]
end
function equilibriumendpointvalue(b::Number)
    T = ChebyshevT{typeof(b)}()
    dot(T[end,:], equilibriumcoefficients(T,b))
end

function equilibrium(b::Number)
    T = ChebyshevT{typeof(b)}()
    Weighted(T) * equilibriumcoefficients(T,b)
end

equilibriumendpointvalue(sqrt(2))

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
    Q = Associated(P)
    t = axes(W,1)
    H = @. inv(t - t')
    H̃ = Q \ (H*W)
    [1/(b*sum(W[:,1])); H̃[:,2:end] \( Q \ derivative.(V, b*t))]
end
function equilibriumendpointvalues(ab::SVector{2})
    a,b = ab
    P = TwoBandJacobi(a/b, -1/2, -1/2, 1/2)
    P[[a/b,1],:] * equilibriumcoefficients(P,a,b)
end

function equilibrium(ab)
    a,b = ab
    P = TwoBandJacobi(a/b, -1/2, -1/2, 1/2)
    Weighted(P) * equilibriumcoefficients(P,a,b)
end

(a,b) = ab = SVector(2.,2.44948974278318)

w = equilibrium(ab)
plot(equilibrium(ab))
sum(w)
1/b

(H * w)[x]
wT1 = Weighted(ChebyshevT())[affine(-1 .. -a/b, -1..1),:]
wT2 = Weighted(ChebyshevT())[affine(a/b ..1, -1..1),:]
t1 = axes(wT1,1)
t2 = axes(wT2,1)


w = W * [0; 0; 1; zeros(∞)]
w1 = wT1 * (wT1 \ view(w,t1))
w2 = wT2 * (wT2 \ view(w,t2))

w2[x]

x = 0.9
(inv.(x .- t1') * w1) + (inv.(t2 .- t2') * w2)[x]
(H * w)[x]

sum(W[:,1])

w[x]

A,B,C = ClassicalOrthogonalPolynomials.recurrencecoefficients(P)

A[2] * x

Q[x,1]

w2[x]


derivative(x -> V(b*x),x)

H * 



ab = ab - jacobian(equilibriumendpointvalues,ab) \ equilibriumendpointvalues(ab)

w = W * [1/sum(W[:,1]); H̃[:,2:end] \( Q \ derivative.(V, b*t))]
plot(w)
