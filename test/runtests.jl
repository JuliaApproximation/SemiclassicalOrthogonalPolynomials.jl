using SemiclassicalOrthogonalPolynomials, OrthogonalPolynomialsQuasi, ContinuumArrays, BandedMatrices, QuasiArrays, Test, LazyArrays
import BandedMatrices: _BandedMatrix
import SemiclassicalOrthogonalPolynomials: normalized_op_lowering
import OrthogonalPolynomialsQuasi: recurrencecoefficients

@testset "Jacobi" begin
    P = Normalized(Legendre())
    L = normalized_op_lowering(P,1)
    L̃ = P \ WeightedJacobi(1,0)
    # off by diagonal
    @test (L ./ L̃)[5,5] ≈ (L ./ L̃)[6,5]
end

@testset "SemiclassicalJacobiWeight" begin
    @test sum(SemiclassicalJacobiWeight(2,0.1,0.2,0.3)) ≈ 0.8387185832077594 #Mathematica
end

@testset "Half-range Chebyshev" begin
    @testset "±1/2" begin
        T = SemiclassicalJacobi(2, 0, -1/2, -1/2)
        W = SemiclassicalJacobi(2, 0, 1/2, -1/2, T)
        X = jacobimatrix(T)
        A, B, C = recurrencecoefficients(T)
        L = T \ (SemiclassicalJacobiWeight(2,0,1,0) .* W)
        @test bandwidths(L) == (1,0)

        @testset "Relationship with Lanczos" begin
            P₋ = jacobi(0,-1/2,0..1)
            x = axes(P₋,1)
            y = @.(sqrt(x)*sqrt(2-x))
            T̃ = LanczosPolynomial(1 ./ y, P₋)
            @test T[0.1,1:10] ≈ T̃[0.1,1:10]/T̃[0.1,1]
            @test T.P \ T == Eye(∞)/T̃[0.1,1]

            P₊ = jacobi(0,1/2,0..1)
            W̃ = LanczosPolynomial(@.(sqrt(x)/sqrt(2-x)), P₊)
            A_W̃, B_W̃, C_W̃ = recurrencecoefficients(W̃)

            # this expresses W in terms of W̃
            kᵀ = cumprod(A)
            k_W̃ = cumprod(A_W̃)
            W̄ = (x,n) -> n == 1 ? one(x) : kᵀ[n]*L[n+1,n]/(W̃[0.1,1]k_W̃[n-1]) *W̃[x,n]

            W̄(0.1,5)
            W[0.1,5]
            @testset "x*W == T*L tells us k_n for W" begin
                @test T[0.1,1] == 1
                @test T[0.1,2] == A[1]*0.1 + B[1] == kᵀ[1]*0.1 + B[1]
                @test T[0.1,3] ≈ (A[2]*0.1 + B[2])*(A[1]*0.1 + B[1]) - C[2] ≈ kᵀ[2]*0.1^2 + (A[2]B[1]+B[2]A[1])*0.1 + B[2]B[1]-C[2]
                @test A[1]*L[2,1] ≈ 1
                @test 0.1 ≈ dot(T[0.1,1:2],L[1:2,1])
                @test 0.1*W̄(0.1,2) ≈ dot(T[0.1,2:3],L[2:3,2])
                @test 0.1*W̄(0.1,3) ≈ dot(T[0.1,3:4],L[3:4,3])
            end

            @testset "Jacobi operator" begin
                X_W_N = (L[1:12,1:12] \ X[1:12,1:11] * L[1:11,1:10])[1:11,:]
                @test 0.1*W̄.(0.1,1:10)' ≈ W̄.(0.1,1:11)' * X_W_N
    
                @test L[1:2,1:2] \ X[1:2,1:2]*L[1:2,1] ≈ X_W_N[1:2,1]
                @test L[1:3,1:3] \ X[1:3,2:3]*L[2:3,2] ≈ X_W_N[1:3,2]
                @test L[2:4,2:4] \ X[2:4,3:4]*L[3:4,3] ≈ X_W_N[2:4,3]

                x = axes(W,1)
                X_W = W \ (x .* W)
                @test X_W isa ConjugateJacobiMatrix
                @test X_W[1:11,1:10] ≈ X_W_N
            end
        end

        @testset "Evaluation" begin
            x = axes(W,1)
            X_W = W \ (x .* W)
            @test W[0.1,1:11]'*X_W[1:11,1:10] ≈ 0.1 * W[0.1,1:10]'
            @test 0.1*W[0.1,1:10]' ≈ T[0.1,1:11]' * L[1:11,1:10]

        end
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