using OrthogonalPolynomialsQuasi, FillArrays, LazyArrays, ArrayLayouts

P = Legendre(); x = axes(P,1)
X = P \ (x .* P)
w = P * (P \ exp.(x))
W = P \ (w .* P)
M = P'P
M*W

W*p0
Debugger.@enter M*p0

C = zeros(∞,10)
C[1,1] = 1
p0 = view(C,:,1)
γ = sqrt(dot(p0, W*p0))
lmul!(inv(γ), p0)


v = view(C,:,2)
muladd!(1.0,X, p0, 0.0, v)
β = dot(M * p0, W * v)
BLAS.axpy!(-β, p0, v)
γ = sqrt(dot(v, W * v))
lmul!(inv(γ), v)



M * p0

C[1,1] = 1
C[:,1]

@which [rand(5); Zeros(∞)]

X*C[:,1]



p0 = [1; zeros(∞)];
v = X*p0

dot(v, W*v)

[1; Zeros(∞)]




P \ (x .* p0)

x