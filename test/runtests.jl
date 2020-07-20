using OrthogonalPolynomialsQuasi, FillArrays, LazyArrays, ArrayLayouts

function lanczos(N, X, M, W)
    ip = (f,g) -> dot(M*f, W*g)

    C = zeros(∞,N)
    γ = zeros(∞)
    β = zeros(∞)

    C[1,1] = 1
    p0 = view(C,:,1)
    γ[1] = sqrt(ip(p0,p0))
    lmul!(inv(γ[1]), p0)

    for n = 2:N
        v = view(C,:,n)
        p1 = view(C,:,n-1)
        muladd!(1.0,X, p1, 0.0, v)
        β[n-1] = ip(v,p1)
        BLAS.axpy!(-β[n-1],p1,v)
        if n > 2
            p0 = view(C,:,n-2)
            BLAS.axpy!(-γ[n-1],p0,v)
        end
        γ[n] = sqrt(ip(v,v))
        lmul!(inv(γ[n]), v)
    end
    γ,β,C
end

P = Legendre(); x = axes(P,1)
X = P \ (x .* P)
w = P * (P \ exp.(x))
W = P \ (w .* P)
M = P'P
Debugger.@enter lanczos(5, X, M, W)


BLAS.axpy!(

for 

muladd!(1.0,X, p1, 0.0, v)
β = ip(p0, v)
BLAS.axpy!(-β, p0, v)
β = ip(p1, v)
BLAS.axpy!(-β, p1, v)
γ = sqrt(ip(v, v))
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