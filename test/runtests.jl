using SemiclassicalOrthogonalPolynomials, OrthogonalPolynomialsQuasi
import OrthogonalPolynomialsQuasi: recurrencecoefficients

@testset "Lanczos" begin
    @testset "Legendre" begin
        P = Legendre()
        w = P * [1; zeros(∞)]
        Q = LanczosPolynomial(w)
        A,B,C = recurrencecoefficients(Q)
        Q̃ = Normalized(P)
        Ã,B̃,C̃ = recurrencecoefficients(Q̃)
        @test @inferred(A[1:10]) ≈ Ã[1:10] ≈ [A[k] for k=1:10]
        @test @inferred(B[1:10]) ≈ B̃[1:10] ≈ [B[k] for k=1:10]
        @test @inferred(C[2:10]) ≈ C̃[2:10] ≈ [C[k] for k=2:10]

        @test A[1:10] isa Vector{Float64}
        @test B[1:10] isa Vector{Float64}
        @test C[1:10] isa Vector{Float64}

        R = P \ Q;
    end

    @testset "Jacobi via Lanczos" begin
        P = Legendre(); x = axes(P,1)
        w = P * (P \ (1 .- x.^2))
        Q = LanczosPolynomial(w)
        A,B,C = recurrencecoefficients(Q)

        @test @inferred(Q[0.1,1]) == 1
        Q[0.1,2]
end
Q.data.R
Q[0.1,1]




X = P \ (x .* P)
W = P \ (w .* P)
M = P'P


w = P * (P \ (@. sqrt(1.01 - x)))
W = P \ (w .* P)
N = 100
@time lanczos(N, X, M, W)