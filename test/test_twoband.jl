using SemiclassicalOrthogonalPolynomials, ClassicalOrthogonalPolynomials, Test

@testset "Two Band" begin
    @testset "Chebyshev case" begin
        ρ = 0.5
        T = TwoBandJacobi(ρ, -1/2, -1/2, 1/2)
        @test T[0.6,1:5] ≈ [T[0.6,k] for k=1:5]

        U = TwoBandJacobi(ρ, 1/2, 1/2, -1/2)
        @test U[0.6,1:5] ≈ [U[0.6,k] for k=1:5]
    end
end