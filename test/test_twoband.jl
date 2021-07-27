using SemiclassicalOrthogonalPolynomials, ClassicalOrthogonalPolynomials, Test
import ClassicalOrthogonalPolynomials: orthogonalityweight, Weighted, associated, plotgrid

@testset "Two Band" begin
    @testset "TwoBandWeight" begin
        w = TwoBandWeight(0.6,1,2,3)
        @test w[0.7] == w[-0.7] == 0.7^6 * (0.7^2-0.6^2)^2 * (1-0.7^2)
        @test_throws BoundsError w[0.4]
        @test_throws BoundsError w[-0.4]
        @test_throws BoundsError w[1.1]
        @test copy(w) == w
        @test_broken sum(w)
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

        # bug
        @test !issymmetric(jacobimatrix(T)[1:10,1:10])
    end

    @testset "Hilbert" begin
        ρ = 0.5
        w = TwoBandWeight(ρ, -1/2, -1/2, 1/2)
        x = axes(w,1)
        H = inv.(x .- x')
        @test iszero(H*w)
        @test sum(w) ≈ π
        
        T = TwoBandJacobi(ρ, -1/2, -1/2, 1/2)
        Q = associated(T)
        @test (Q[0.6,1:100]' * (Q[:,Base.OneTo(100)] \ exp.(x))) ≈ exp(0.6)
        @test (Q[-0.6,1:100]' * (Q[:,Base.OneTo(100)] \ exp.(x))) ≈ exp(-0.6)

        @test (Q * (Q \ exp.(x)))[0.6] ≈ exp(0.6)
        @test_broken Q \ (H * Weighted(T)) # need to deal with Hcat

        @test_broken H*TwoBandWeight(ρ, 1/2, 1/2, -1/2)
    end

    @testset "plotgrid" begin
        ρ = 0.5
        P = TwoBandJacobi(ρ, -1/2, -1/2, 1/2)
        @test all(x -> ρ ≤ abs(x) ≤ 1, plotgrid(P[:,1:5]))
    end

    @testset "associated transform error" begin
        a,b = 0.5, 0.9
        Vp = x -> 4x^3 - 20x
        Q = associated(TwoBandJacobi((a/b), -1/2, -1/2, 1/2))
        x = axes(Q,1)
        @test norm((Q[:,Base.OneTo(30)] \ Vp.(b*x))[1]) ≤ 1E-13
    end
end