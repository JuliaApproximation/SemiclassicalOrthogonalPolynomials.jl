using SemiclassicalOrthogonalPolynomials, OrthogonalPolynomialsQuasi, BandedMatrices, Test
import OrthogonalPolynomialsQuasi: recurrencecoefficients, resizedata!

@testset "Lanczos" begin
    @testset "Legendre" begin
        P = Legendre();
        w = P * [1; zeros(∞)];
        Q = LanczosPolynomial(w);
        @test Q.data.W[1:10,1:10] isa BandedMatrix
        v = [randn(5); Zeros(∞)]
        @time Q.data.W.args[2]*v
        @ent Q.data.W[1:10,1:10]

        Q̃ = Normalized(P);
        A,B,C = recurrencecoefficients(Q)
        Ã,B̃,C̃ = recurrencecoefficients(Q̃)
        @test @inferred(A[1:10]) ≈ Ã[1:10] ≈ [A[k] for k=1:10]
        @test @inferred(B[1:10]) ≈ B̃[1:10] ≈ [B[k] for k=1:10]
        @test @inferred(C[2:10]) ≈ C̃[2:10] ≈ [C[k] for k=2:10]

        @test A[1:10] isa Vector{Float64}
        @test B[1:10] isa Vector{Float64}
        @test C[1:10] isa Vector{Float64}

        @test Q[0.1,1:10] ≈ Q̃[0.1,1:10]

        R = P \ Q
        @test R[1:10,1:10] ≈ (P \ Q̃)[1:10,1:10]
    end

    @testset "Jacobi via Lanczos" begin
        P = Legendre(); x = axes(P,1)
        w = P * (P \ (1 .- x.^2))
        Q = LanczosPolynomial(w)
        A,B,C = recurrencecoefficients(Q)

        @test @inferred(Q[0.1,1]) == 1
        Q[0.1,2]
    end
end
Q.data.R
Q[0.1,1]




X = P \ (x .* P)
W = P \ (w .* P)
M = P'P

x = axes(P,1)
w = P * (P \ (@. sqrt(1.01 - x)))
Q = LanczosPolynomial(w);
R = P \ Q;

