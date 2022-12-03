using SemiclassicalOrthogonalPolynomials, ClassicalOrthogonalPolynomials, BandedMatrices, LinearAlgebra, Test
import ClassicalOrthogonalPolynomials: orthogonalityweight, Weighted, associated, plotgrid
import SemiclassicalOrthogonalPolynomials: Interlace, HalfWeighted, WeightedBasis

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

        R = TwoBandJacobi(ρ, 1, 1, 0)
        x=0.6; τ=(1-x^2)*R.P.t; n=10
        @test R[x, 1:n] ≈ Interlace((-1).^(2:n÷2+1).*R.P[τ, 1:n÷2], (-1).^(2:n÷2+1).*x.*R.Q[τ, 1:n÷2])[1:n]

        # bug
        @test !issymmetric(jacobimatrix(T)[1:10,1:10])
    end

    @testset "Associated" begin
        ρ = 0.5
        T = TwoBandJacobi(ρ, -1/2, -1/2, 1/2)
        Q = associated(T)
        x = axes(Q,1)
        @test 0 in x
        @test (Q[0.6,1:101]' * (Q[:,Base.OneTo(101)] \ exp.(x))) ≈ exp(0.6)
        @test (Q[-0.6,1:101]' * (Q[:,Base.OneTo(101)] \ exp.(x))) ≈ exp(-0.6)

        @test (Q * (Q \ exp.(x)))[0.6] ≈ exp(0.6)
    end

    @testset "Hilbert" begin
        ρ = 0.5
        w = TwoBandWeight(ρ, -1/2, -1/2, 1/2)
        T = TwoBandJacobi(ρ, -1/2, -1/2, 1/2)
        Q = associated(T)
        t = axes(w,1)
        x = axes(Q,1)
        H = inv.(x .- t')
        @test iszero(H*w)
        @test sum(w) ≈ π
        
        B = Q \ H * Weighted(T)
        @test B isa BandedMatrix

        @test Q[0.6,:]'* (B * (T\exp.(t))) ≈ -4.701116657130821 # Mathematica

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

    @testset "derivative half weighted twoband" begin    
        ρ = 0.2
        R = TwoBandJacobi(ρ, 0, 0, 0)
        D = R \ Derivative(axes(R,1))*HalfWeighted{:ab}(TwoBandJacobi(ρ, 1, 1, 0))

        @test (D.l, D.u) == (3, -1)
        @test D[1:5,1] ≈ [0.0, -0.33858064516128994, 0.0, -1.030932383768154, 0.0]
        @test D[1:5,2] ≈ [0.0, 0.0, -0.4609776443497252, 0.0, -0.34580926348402935]
    end

    @testset "L2 inner product" begin
        ρ = 0.2
        HP = HalfWeighted{:ab}(TwoBandJacobi(ρ, 1, 1, 0))
        M = HP' * HP
        
        @test (M.l, M.u) == (4, 4)

        Mm = M[1:20, 1:20]
        @test Mm == Mm'

        # The following is checked via QuadGK, e.g. 2*quadgk(x->HP[x,n]*HP[x,n], ρ, 1, rtol=1e-3)[1]
        @test diag(Mm[1:5, 1:5]) ≈ [0.0399723828825396, 0.01926644161200575, 0.028656015290810938, 0.014202710309357196, 0.02719735154820007]
        @test diag(Mm[2:6, 1:5]) ≈ [0, 0, 0, 0, 0]
        @test diag(Mm[3:7, 1:5]) ≈ [0.003417521501385307, -0.0013645130261003458, 0.0008788935542028268, -0.0005306108105399927, 0.0003587270849050795]
        @test diag(Mm[4:8, 1:5]) ≈ [0, 0, 0, 0, 0]
        @test diag(Mm[5:9, 1:5]) ≈ [-0.011436406405959944, -0.004811623256062165, -0.012192463544540394, -0.005436023256389608, -0.012469708623429861]
    end

    @testset "halfweight" begin
        ρ = 0.5
        T = TwoBandJacobi(ρ, -1/2, -1/2, 1/2)
        @test HalfWeighted{:ab}(T)[0.6,1:10] ≈ convert(WeightedBasis, HalfWeighted{:ab}(T))[0.6,1:10] ≈ (1-0.6^2)^(-1/2) * (0.6^2-ρ^2)^(-1/2) * T[0.6,1:10]
        
    end
end