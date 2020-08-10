using OrthogonalPolynomialsQuasi, FillArrays, LazyArrays, ArrayLayouts
import LazyArrays: resizedata!, paddeddata
import OrthogonalPolynomialsQuasi: OrthogonalPolynomial

# We roughly follow DLMF notation
# P[x,n] = k[n] * x^(n-1) 
# 

function lanczos(N, X, ip)
    R = zeros(∞,N); # Conversion operator to Legendre
    γ = zeros(∞);
    β = zeros(∞);

    R[1,1] = 1;
    p0 = view(R,:,1);
    γ[1] = sqrt(ip(p0,p0))
    lmul!(inv(γ[1]), p0);

    for n = 2:N
        resizedata!(R, n, n);
        v = view(R,:,n);
        p1 = view(R,:,n-1);
        muladd!(1.0,X, p1, 0.0, v);
        β[n-1] = ip(v,p1)
        BLAS.axpy!(-β[n-1],p1,v);
        if n > 2
            p0 = view(R,:,n-2)
            BLAS.axpy!(-γ[n-1],p0,v)    
        end
        γ[n] = sqrt(ip(v,v));
        lmul!(inv(γ[n]), v)
    end
    γ,β,R
end

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