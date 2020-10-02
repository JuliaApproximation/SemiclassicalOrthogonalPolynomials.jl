using SemiclassicalOrthogonalPolynomials, OrthogonalPolynomialsQuasi, ContinuumArrays, BandedMatrices, QuasiArrays, Test, LazyArrays
import BandedMatrices: _BandedMatrix
import SemiclassicalOrthogonalPolynomials: normalized_op_lowering

@testset "Jacobi" begin
    P = Normalized(Legendre())
    L = normalized_op_lowering(P,1)
    L̃ = P \ WeightedJacobi(1,0)
    # off by diagonal
    @test (L ./ L̃)[5,5] ≈ (L ./ L̃)[6,5]
end1


@testset "Half-range Chebyshev" begin
    @testset "negative" begin
        T = SemiclassicalJacobi(2, 0, -1/2, -1/2)

        P₋ = jacobi(0,-1/2,0..1)
        x = axes(P₋,1)
        y = @.(sqrt(x)*sqrt(2-x))
        T̃ = LanczosPolynomial(1 ./ y, P₋)
        @test T[0.1,1:10] ≈ T̃[0.1,1:10]/T̃[0.1,1]
        @test T.P \ T == Eye(∞)/T̃[0.1,1]

        W = SemiclassicalJacobi(2, 0, 1/2, -1/2, T.P)
        L = T \ (SemiclassicalJacobiWeight(2,0,1,0) .* W)
        @test bandwidths(L) == (1,0)
    end


    @testset "Derivation" begin
        P₋ = jacobi(0,-1/2,0..1)
        P₊ = jacobi(0,1/2,0..1)
        x = axes(P₋,1)
        y = @.(sqrt(x)*sqrt(2-x))
        T = LanczosPolynomial(1 ./ y, P₋)
        W = LanczosPolynomial(@.(sqrt(x)/sqrt(2-x)), P₊)
        X = T \ (x .* T)

        @testset "Christoffel–Darboux" begin
            x,y = 0.1,0.2
            n = 10
            β = X[n,n+1]
            @test (x-y) * T[x,1:n]'*T[y,1:n] ≈ T[x,n:n+1]' * [0 -β; β 0] * T[y,n:n+1]

            # y = 0.0

            @test x * T[x,1:n]'T[0,1:n] ≈ -X[n,n+1]*(T[x,n]*T[0,n+1] - T[x,n+1]*T[0,n])

            # y = 2.0
            @test (2-x) * T[x,1:n]'*Base.unsafe_getindex(T,2,1:n) ≈ T[x,n:n+1]' * [0 β; -β 0] * Base.unsafe_getindex(T,2,n:n+1) ≈
                        β*(T[x,n]*Base.unsafe_getindex(T,2,n+1) - T[x,n+1]*Base.unsafe_getindex(T,2,n))

            @testset "T and W" begin
                W̃ = (x,n) -> -X[n,n+1]*(T[x,n]*T[0,n+1] - T[x,n+1]*T[0,n])/x
                @test norm(diff(W̃.([0.1,0.2,0.3],5) ./ W[[0.1,0.2,0.3],5])) ≤ 1E-14

                L = _BandedMatrix(Vcat((-X.ev .* T[0,2:end])', (X.ev .* T[0,1:end])'), ∞, 1, 0)
                x = 0.1
                @test x*W̃(x,1) ≈ T[x,1:2]' * L[1:2,1]
                @test (x*W̃.(x,1:10)')[:,1:9] ≈ W̃.(x,1:10)' * (L[1:10,1:10] \ X[1:10,1:10] * L[1:10,1:9])

                @test (L[1:10,1:10] \ X[1:10,1:10] * L[1:10,1:9])[2:4,3] ≈ L[2:4,2:4] \ (X*L)[2:4,3] 
            end
        end
    end

    
end

@testset "Old" begin
    P₋ = jacobi(-1/2,0,0..1)
    T = LanczosPolynomial(1 ./ y, P₋)

    @test bandwidths(U.P \ T.P) == (0,1)
    @test U.w == U.w
    R = U \ T;

    x̃ = 0.1; ỹ = y[x̃]
    n = 5
    # R is upper-tridiagonal
    @test T[x̃,n] ≈ dot(R[n-2:n,n], U[x̃,n-2:n])

    J_U = jacobimatrix(U)
    J_T = jacobimatrix(T)
    
    H_1 = T
    H_2 = y .* U

    @test (T \ (y .* H_2[:,1]))[1:3] ≈ R[1,1:3]

    n = 5
    @test x̃ * H_1[x̃,n] ≈ dot(J_T[n-1:n+1,n],H_1[x̃,n-1:n+1])
    @test x̃ * H_2[x̃,n] ≈ dot(J_U[n-1:n+1,n],H_2[x̃,n-1:n+1])
    @test ỹ * H_1[x̃,n] ≈ dot(R[n-2:n,n], H_2[x̃,n-2:n])
    @test ỹ * H_2[x̃,n] ≈ (1 - x̃^2)*U[x̃,n] ≈ dot(R[n,n:n+2], H_1[x̃,n:n+2])
end