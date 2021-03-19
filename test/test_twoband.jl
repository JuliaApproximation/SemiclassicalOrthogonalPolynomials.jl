using SemiclassicalOrthogonalPolynomials, ClassicalOrthogonalPolynomials, Test
import ClassicalOrthogonalPolynomials: orthogonalityweight

@testset "Two Band" begin
    @testset "TwoBandWeight" begin
        w = TwoBandWeight(0.6,1,2,3)
        @test w[0.7] == w[-0.7] == 0.7^6 * (0.7^2-0.6^2)^2 * (1-0.7^2)
        @test_throws BoundsError w[0.4]
        @test_throws BoundsError w[-0.4]
        @test_throws BoundsError w[1.1]
        @test copy(w) == w
    end
    @testset "Chebyshev case" begin
        ρ = 0.5
        T = TwoBandJacobi(ρ, -1/2, -1/2, 1/2)
        @test T[0.6,1:5] ≈ [T[0.6,k] for k=1:5]

        U = TwoBandJacobi(ρ, 1/2, 1/2, -1/2)
        @test U[0.6,1:5] ≈ [U[0.6,k] for k=1:5]

        @test copy(T) == T
        @test U ≠ T
        @test orthogonalityweight(T) == TwoBandWeight(ρ, -1/2, -1/2, 1/2)
    end
end