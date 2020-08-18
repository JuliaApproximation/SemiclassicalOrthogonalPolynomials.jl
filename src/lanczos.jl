# We roughly follow DLMF notation
# P[x,n] = k[n] * x^(n-1) 
# Note that we have
# Q[x,n] = (1/γ[n] - β[n-1]/γ[n]) * x * P[x,n-1] - γ[n-1]/γ[n] * P[x,n]
# 

function lanczos!(Ns, X::AbstractMatrix{T}, W::AbstractMatrix{T}, γ::AbstractVector{T}, β::AbstractVector{T}, R::AbstractMatrix{T}) where T
    for n = Ns
        v = view(R,:,n);
        p1 = view(R,:,n-1);
        muladd!(one(T), X, p1, zero(T), v); # TODO: `mul!(v, X, p1)`
        β[n-1] = dot(v,W,p1)
        BLAS.axpy!(-β[n-1],p1,v);
        if n > 2
            p0 = view(R,:,n-2)
            BLAS.axpy!(-γ[n-1],p0,v)    
        end
        γ[n] = sqrt(dot(v,W,v));
        lmul!(inv(γ[n]), v)
    end
    γ,β,R
end

const PaddedVector{T} = CachedVector{T,Vector{T},Zeros{T,1,Tuple{OneToInf{Int}}}}
const PaddedMatrix{T} = CachedMatrix{T,Matrix{T},Zeros{T,2,NTuple{2,OneToInf{Int}}}}

mutable struct LanczosData{T,XX,WW}
    X::XX
    W::WW
    γ::PaddedVector{T}
    β::PaddedVector{T}
    R::PaddedMatrix{T}
    ncols::Int

    function LanczosData{T,XX,WW}(X, W, γ, β, R) where {T,XX,WW}
        R[1,1] = 1;
        p0 = view(R,:,1);
        γ[1] = sqrt(dot(p0,W,p0))
        lmul!(inv(γ[1]), p0)
        new{T,XX,WW}(X, W, γ, β, R, 1)
    end
end

LanczosData(X::XX, W::WW, γ::AbstractVector{T}, β, R) where {T,XX,WW} = LanczosData{T,XX,WW}(X, W, γ, β, R)
LanczosData(X::AbstractMatrix, W::AbstractMatrix) = LanczosData(X, W, zeros(∞), zeros(∞), zeros(∞,∞))

function LanczosData(w::AbstractQuasiVector, Q::AbstractQuasiMatrix)
    x = axes(Q,1)
    X = Q \ (x .* Q)
    W = Q \ (w .* Q)
    LanczosData(X, W)
end

function resizedata!(L::LanczosData, N)
    N ≤ L.ncols && return L
    resizedata!(L.R, N, N)
    resizedata!(L.γ, N)
    resizedata!(L.β, N)
    lanczos!(L.ncols+1:N, L.X, L.W, L.γ, L.β, L.R)
    L.ncols = N
    L
end

struct LanczosConversion{T,XX,WW} <: LazyMatrix{T}
    data::LanczosData{T,XX,WW}
end

size(::LanczosConversion) = (∞,∞)
bandwidths(::LanczosConversion) = (0,∞)
colsupport(L::LanczosConversion, j) = 1:maximum(j)

function getindex(R::LanczosConversion, k, j)
    resizedata!(R.data, min(maximum(k), maximum(j)))
    R.data.R[k,j]
end

# struct LanczosJacobiMatrix{T,XX,WW} <: AbstractBandedMatrix{T}
#     data::LanczosData{T,XX,WW}
# end


struct LanczosRecurrence{ABC,T,XX,WW} <: LazyVector{T}
    data::LanczosData{T,XX,WW}
end

LanczosRecurrence{ABC}(data::LanczosData{T,XX,WW}) where {ABC,T,XX,WW} = LanczosRecurrence{ABC,T,XX,WW}(data)

size(P::LanczosRecurrence) = (∞,)

resizedata!(A::LanczosRecurrence, n) = resizedata!(A.data, n)

function _lanczos_getindex(A::LanczosRecurrence{:A}, I) 
    resizedata!(A, maximum(I)+1)
    inv.(A.data.γ.data[I .+ 1])
end

function _lanczos_getindex(B::LanczosRecurrence{:B}, I) 
    resizedata!(B, maximum(I)+1)
    B.data.β.data[I] ./ B.data.γ.data[I .+ 1]
end

function _lanczos_getindex(C::LanczosRecurrence{:C}, I)
    resizedata!(C, maximum(I)+1)
    C.data.γ.data[I] ./ C.data.γ.data[I  .+ 1]
end

getindex(A::LanczosRecurrence, I::Integer) = _lanczos_getindex(A, I)
getindex(A::LanczosRecurrence, I::AbstractVector) = _lanczos_getindex(A, I)
getindex(K::LanczosRecurrence, k::InfUnitRange) = view(K, k)
getindex(K::SubArray{<:Any,1,<:LanczosRecurrence}, k::InfUnitRange) = view(K, k)


struct LanczosPolynomial{T,XX,WW,Weight,Basis} <: OrthogonalPolynomial{T}
    w::Weight # Weight of orthogonality
    P::Basis # Basis we use to represent the OPs
    data::LanczosData{T,XX,WW}
end


function LanczosPolynomial(w::AbstractQuasiVector)
    P = qr(basis(w)).Q
    LanczosPolynomial(w, P, LanczosData(w, P))
end

axes(Q::LanczosPolynomial) = (axes(Q.w,1),OneToInf())

_p0(Q::LanczosPolynomial) = inv(Q.data.γ[1])*_p0(Q.P)

recurrencecoefficients(Q::LanczosPolynomial) = LanczosRecurrence{:A}(Q.data),LanczosRecurrence{:B}(Q.data),LanczosRecurrence{:C}(Q.data)

# Sometimes we want to expand out, sometimes we don't

QuasiArrays.ApplyQuasiArray(Q::LanczosPolynomial) = ApplyQuasiArray(*, arguments(ApplyLayout{typeof(*)}(), Q)...)


\(A::OrthogonalPolynomial, Q::LanczosPolynomial) = (A \ Q.P) * LanczosConversion(Q.data)

ArrayLayouts.mul(Q::LanczosPolynomial, C::AbstractArray) = ApplyQuasiArray(*, Q, C)
transform_ldiv(Q::LanczosPolynomial, C::AbstractQuasiArray) = LanczosConversion(Q.data) \ (Q.P \ C)
arguments(::ApplyLayout{typeof(*)}, Q::LanczosPolynomial) = Q.P, LanczosConversion(Q.data)
LazyArrays._mul_arguments(Q::LanczosPolynomial) = arguments(ApplyLayout{typeof(*)}(), Q)
LazyArrays._mul_arguments(Q::QuasiAdjoint{<:Any,<:LanczosPolynomial}) = arguments(ApplyLayout{typeof(*)}(), Q)

