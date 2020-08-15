using OrthogonalPolynomialsQuasi, FillArrays, LazyArrays, ArrayLayouts
import LazyArrays: resizedata!, paddeddata
import OrthogonalPolynomialsQuasi: OrthogonalPolynomial

# We roughly follow DLMF notation
# P[x,n] = k[n] * x^(n-1) 
# 

function lanczos!(Ns, X, W, R, γ, β)
    for n = Ns
        if n == 1
            R[1,1] = 1;
            p0 = view(R,:,1);
            γ[1] = sqrt(dot(p0,W,p0))
            lmul!(inv(γ[1]), p0)
        else
            v = view(R,:,n);
            p1 = view(R,:,n-1);
            muladd!(1.0,X, p1, 0.0, v);
            β[n-1] = dot(v,W,p1)
            BLAS.axpy!(-β[n-1],p1,v);
            if n > 2
                p0 = view(R,:,n-2)
                BLAS.axpy!(-γ[n-1],p0,v)    
            end
            γ[n] = sqrt(dot(v,W,v));
            lmul!(inv(γ[n]), v)
        end
    end
    γ,β,R
end

function lanczos(N, X, W)
    R = zeros(∞,N); # Conversion operator to Legendre
    resizedata!(R, N, N);
    γ = zeros(∞);
    β = zeros(∞);
    lanczos!(1:N, X, W, R, γ, β)
end

Q = Normalized(Legendre())
x = axes(Q,1)
w = Q * (Q \ (1 .- x.^2));
W = Q\ (w .* Q)
X = Q \ (x .* Q)

@time lanczos(1000, X, W)

x = [1; 2; zeros(∞)]
@time dot(x, W, x)

(1-0.1^2)

const PaddedVector{T} = CachedVector{T,Vector{T},Zeros{T,1,Tuple{OneToInf{Int}}}}

struct OrthonormalConversion{T, IP} <: AbstractMatrix{T}
    ip::IP
    γ::PaddedVector{T}
    β::PaddedVector{T}
    R
end


OrthonormalConversion{T}(ip::IP) where IP<:Function = OrthonormalConversion{T, IP}(ip)
OrthonormalConversion(M::AbstractMatrix{T}, W::AbstractMatrix{T}) where T = OrthonormalConversion{T}((f,g) -> dot(M*f, W*g))




struct OrthonormalPolynomial{T} <: OrthogonalPolynomial{T}
    A::Vector{T}
    B::Vector{T}
    C::Vector{T}
    datalength::Int
end




P = Legendre(); x = axes(P,1)
X = P \ (x .* P)
w = P * (P \ exp.(x))
W = P \ (w .* P)
M = P'P


w = P * (P \ (@. sqrt(1.01 - x)))
W = P \ (w .* P)
N = 100
@time lanczos(N, X, M, W)