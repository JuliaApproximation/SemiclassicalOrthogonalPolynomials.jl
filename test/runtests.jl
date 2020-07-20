using OrthogonalPolynomialsQuasi, FillArrays, LazyArrays, ArrayLayouts

P = Legendre(); x = axes(P,1)
X = P \ (x .* P)
w = P * Vcat(Vector((P \ exp.(x))[1:20]),Zeros(∞));
W = P \ (w .* P)
C = zeros(∞,10)
C[1,1] = 1
v = view(C,:,2)
muladd!(1.0,X, view(C,:,1), 0.0, v)

dot(view(C,:,1), W * v)






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