using SemiclassicalOrthogonalPolynomials, ClassicalOrthogonalPolynomials, ContinuumArrays, BandedMatrices, QuasiArrays, Test, LazyArrays, LinearAlgebra
import BandedMatrices: _BandedMatrix
import SemiclassicalOrthogonalPolynomials: op_lowering
import ClassicalOrthogonalPolynomials: recurrencecoefficients, orthogonalityweight

@testset "Jacobi" begin
    P = Normalized(Legendre())
    L = op_lowering(P,1)
    L̃ = P \ WeightedJacobi(1,0)
    # off by scaling
    @test (L ./ L̃)[5,5] ≈ (L ./ L̃)[6,5]
    L = op_lowering(P,-1)
    L̃ = P \ WeightedJacobi(0,1)
    @test (L ./ L̃)[5,5] ≈ (L ./ L̃)[6,5]

    t = 2
    P̃ = SemiclassicalJacobi(t, 0, 0, 0)
    @test P̃[0.1,1:10] ≈ P[2*0.1-1,1:10]/P[0.1,1]
end

@testset "SemiclassicalJacobiWeight" begin
    a,b,c = 0.2,0.1,0.3
    w = SemiclassicalJacobiWeight(2,a,b,c)
    @test w[0.1] ≈ 0.1^a * (1-0.1)^b * (2-0.1)^c
    @test sum(w) ≈ 0.8387185832077594 #Mathematica
end

@testset "Half-range Chebyshev" begin
    @testset "T and W" begin
        T = SemiclassicalJacobi(2, -1/2, 0, -1/2)
        W = SemiclassicalJacobi(2,  1/2, 0, -1/2, T)
        w_T = orthogonalityweight(T)
        w_W = orthogonalityweight(W)
        X = jacobimatrix(T)
        A, B, C = recurrencecoefficients(T)
        L = T \ (SemiclassicalJacobiWeight(2,1,0,0) .* W);
        R = W \ T;
        @test bandwidths(L) == (1,0)

        @time @testset "Relationship with Lanczos" begin
            P₋ = jacobi(0,-1/2,0..1)
            x = axes(P₋,1)
            y = @.(sqrt(x)*sqrt(2-x))
            T̃ = LanczosPolynomial(1 ./ y, P₋)
            @test T[0.1,1:10] ≈ T̃[0.1,1:10]/T̃[0.1,1]
            @test (T.P \ T)[1:10,1:10] ≈ Eye(10)/T̃[0.1,1]
            @test Normalized(T).scaling[1:10] ≈ fill(1/sqrt(sum(w_T)), 10)

            P₊ = jacobi(0,1/2,0..1)
            W̃ = LanczosPolynomial(@.(sqrt(x)/sqrt(2-x)), P₊)
            A_W̃, B_W̃, C_W̃ = recurrencecoefficients(W̃)

            @test Normalized(W)[0.1,1:10] ≈ W̃[0.1,1]W[0.1,1:10] ≈ W̃[0.1,1:10]
            
            # this expresses W in terms of W̃
            kᵀ = cumprod(A)
            k_W̃ = cumprod(A_W̃)
            W̄ = (x,n) -> n == 1 ? one(x) : kᵀ[n]*L[n+1,n]/(W̃[0.1,1]k_W̃[n-1]) *W̃[x,n]
            @test W̄.(0.1,1:5) ≈ W[0.1,1:5]

            @testset "x*W == T*L tells us k_n for W" begin
                @test T[0.1,1] == 1
                @test T[0.1,2] ≈ A[1]*0.1 + B[1] ≈ kᵀ[1]*0.1 + B[1]
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
                X_W = W \ (x .* W);
                @test X_W isa ClassicalOrthogonalPolynomials.SymTridiagonal
                @test X_W[1:11,1:10] ≈ X_W_N
            end
        end

        @testset "Evaluation" begin
            x = axes(W,1)
            A_W,B_W,C_W = recurrencecoefficients(W)
            X_W = W \ (x .* W);

            @test W[0.1,2] ≈ A_W[1]*0.1 + B_W[1]
            @test W[0.1,3] ≈ (A_W[2]*0.1 + B_W[2])*(A_W[1]*0.1 + B_W[1]) - C_W[2]

            @test W[0.1,1:11]'*X_W[1:11,1:10] ≈ 0.1 * W[0.1,1:10]'
            @test 0.1*W[0.1,1:10]' ≈ T[0.1,1:11]' * L[1:11,1:10]

            @test  W[0.1,1:3] ≈ [1,-2.029703176007828,2.2634440711109365]
        end

        @testset "Mass matrix" begin
            @test (T'*(w_T .* T))[1:10,1:10] ≈ sum(w_T)I
            M = W'*(w_W .* W)
            @test (sum(w_T)*inv(R)'L)[1:10,1:10] ≈ M[1:10,1:10]
            @test T[0.1,1:10]' ≈ W[0.1,1:10]' * R[1:10,1:10]
        end
    end

    @time @testset "T and V" begin
        T = SemiclassicalJacobi(2, -1/2, 0, -1/2)
        V = SemiclassicalJacobi(2, -1/2, 0, 1/2, T)
        w_T = orthogonalityweight(T)
        w_V = orthogonalityweight(V)
        X = jacobimatrix(T)
        A, B, C = recurrencecoefficients(T)
        L = T \ (SemiclassicalJacobiWeight(2,0,0,1) .* V);
        @test eltype(L) == Float64
        @test bandwidths(L) == (1,0)

        @testset "Relationship with Lanczos" begin
            P₋ = jacobi(0,-1/2,0..1)
            x = axes(T,1)
            Ṽ = LanczosPolynomial(@.(sqrt(2-x)/sqrt(x)), P₋)
            A_Ṽ, B_Ṽ, C_Ṽ = recurrencecoefficients(Ṽ)

            @test Normalized(V)[0.1,1:10] ≈ Ṽ[0.1,1:10]

            # this expresses W in terms of W̃
            kᵀ = cumprod(A)
            k_Ṽ = cumprod(A_Ṽ)
            V̄ = (x,n) -> n == 1 ? one(x) : -kᵀ[n]*L[n+1,n]/(Ṽ[0.1,1]k_Ṽ[n-1]) *Ṽ[x,n]

            @test V̄.(0.1,1:5) ≈ V[0.1,1:5]

            @testset "x*W == T*L tells us k_n for W" begin
                @test (2-0.1)*V̄(0.1,2) ≈ dot(T[0.1,2:3],L[2:3,2])
                @test (2-0.1)*V̄(0.1,3) ≈ dot(T[0.1,3:4],L[3:4,3])
            end

            @testset "Jacobi operator" begin
                X_V_N = (L[1:12,1:12] \ X[1:12,1:11] * L[1:11,1:10])[1:11,:]
                @test 0.1*V̄.(0.1,1:10)' ≈ V̄.(0.1,1:11)' * X_V_N
    
                @test L[1:2,1:2] \ X[1:2,1:2]*L[1:2,1] ≈ X_V_N[1:2,1]
                @test L[1:3,1:3] \ X[1:3,2:3]*L[2:3,2] ≈ X_V_N[1:3,2]
                @test L[2:4,2:4] \ X[2:4,3:4]*L[3:4,3] ≈ X_V_N[2:4,3]

                x = axes(V,1)
                X_V = V \ (x .* V)
                @test X_V isa ClassicalOrthogonalPolynomials.SymTridiagonal
                @test X_V[1:11,1:10] ≈ X_V_N
            end
        end

        @testset "Evaluation" begin
            x = axes(V,1)
            A_V,B_V,C_V = recurrencecoefficients(V)
            X_V = V \ (x .* V)

            @test V[0.1,2] ≈ A_V[1]*0.1 + B_V[1]
            @test V[0.1,3] ≈ (A_V[2]*0.1 + B_V[2])*(A_V[1]*0.1 + B_V[1]) - C_V[2]

            @test V[0.1,1:11]'*X_V[1:11,1:10] ≈ 0.1 * V[0.1,1:10]'
            @test (2-0.1)*V[0.1,1:10]' ≈ T[0.1,1:11]' * L[1:11,1:10]
        end

        @testset "Mass matrix" begin
            R = V \ T;
            @test T[0.1,1:10]' ≈ V[0.1,1:10]' * R[1:10,1:10]
        end
    end

    @time @testset "U" begin
        T = SemiclassicalJacobi(2, -1/2, 0, -1/2)
        W = SemiclassicalJacobi(2, 1/2, 0, -1/2, T)
        V = SemiclassicalJacobi(2, -1/2, 0, 1/2, T)
        U = SemiclassicalJacobi(2, 1/2, 0, 1/2, T)

        @testset "Jacobi matrix derivation" begin
            L_1 = T \ (SemiclassicalJacobiWeight(2,0,0,1) .* V);
            L_2 = V \ (SemiclassicalJacobiWeight(2,1,0,0) .* U);


            X_V = jacobimatrix(V)
            X_U = jacobimatrix(U)
            @test (inv(L_2[1:12,1:12]) * X_V[1:12,1:11] * L_2[1:11,1:10])[1:10,1:10] ≈ X_U[1:10,1:10]

            @test 0.1*U[0.1,1:10]' ≈ V[0.1,1:11]' * L_2[1:11,1:10]
            @test 0.1 * V[0.1,1:10]'V[0.0,1:10] ≈ V[0.1,1:11]'*X_V[1:11,1:10]*V[0.0,1:10]

            L = T \ (SemiclassicalJacobiWeight(2,1,0,1) .* U);
            @test L[1:10,1:10] ≈ L_1[1:10,1:10] * L_2[1:10,1:10]
            @test (2-0.1)*0.1 * U[0.1,1:10]' ≈ T[0.1,1:12]' * L[1:12,1:10]

            L̃_1 = T \ (SemiclassicalJacobiWeight(2,1,0,0) .* W);
            L̃_3 = inv(L̃_1[1:11,1:11])*L[1:11,1:10]
            @test (2-0.1) * U[0.1,1:10]' ≈ W[0.1,1:11]' * L̃_3[1:11,1:10]
            L_3 = W \ (SemiclassicalJacobiWeight(2,0,0,1) .* U);
            @test L_3 isa ClassicalOrthogonalPolynomials.Bidiagonal
            @test L̃_3 ≈ L_3[1:11,1:10]
            @test (2-0.1) * U[0.1,1:10]' ≈ W[0.1,1:11]' * L_3[1:11,1:10]

            R = U \ T;
            @test T[0.1,1:10]' ≈ U[0.1,1:10]' * R[1:10,1:10]
        end

        @testset "Evaluation" begin
            @test U[0.1,1:4] ≈ [1.1283791670955114, -2.03015300715061, 2.131688648563451, -1.3816046198529939] * sqrt(sum(SemiclassicalJacobiWeight(2, 1/2,0,1/2)))
        end
    end

    @testset "Expansions" begin
        T = SemiclassicalJacobi(2, -1/2, 0, -1/2)
        U = SemiclassicalJacobi(2, 1/2, 0, 1/2, T)
        x = axes(T,1)

        u = T * (T \ exp.(x))
        @test u[0.1] ≈ exp(0.1)

        @test T[:,1:20] \ exp.(x) ≈ u.args[2][1:20]

        u = U * (U \ exp.(x))
        @test u[0.1] ≈ exp(0.1)
        @test U[:,1:20] \ exp.(x) ≈ u.args[2][1:20]
    end

    @time @testset "Derivation" begin
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
                @test norm(diff(W̃.([0.1,0.2,0.3],5) ./ W[[0.1,0.2,0.3],5])) ≤ 10E-14

                L = _BandedMatrix(Vcat((-X.ev .* T[0,2:end])', (X.ev .* T[0,1:end])'), ∞, 1, 0)
                x = 0.1
                @test x*W̃(x,1) ≈ T[x,1:2]' * L[1:2,1]
                @test (x*W̃.(x,1:10)')[:,1:9] ≈ W̃.(x,1:10)' * (L[1:10,1:10] \ X[1:10,1:10] * L[1:10,1:9])

                @test (L[1:10,1:10] \ X[1:10,1:10] * L[1:10,1:9])[2:4,3] ≈ L[2:4,2:4] \ (X*L)[2:4,3]
            end
        end
    end

    @testset "P" begin
        P = SemiclassicalJacobi(2.0,0,0,0)
        P̃ = Normalized(legendre(0..1))
        @test P̃[0.1,1:10] ≈ P[0.1,1:10]
        Q = SemiclassicalJacobi(2.0,0,0,1)
        Q \ P
        x = axes(Q,1)
        X = Q \ (x .* Q)
        @time X[1:1000,1:1000];
    end

    @testset "BigFloat" begin
        # T = SemiclassicalJacobi(2, -BigFloat(1)/2, 0, -BigFloat(1)/2)
    end
end

@testset "Legendre" begin
    t = 2
    P =   SemiclassicalJacobi(t, 0, 0, 0)
    P¹¹ = SemiclassicalJacobi(t, 1, 1, 0, P)
    Q = SemiclassicalJacobi(t, 0, 0, 1, P)
    R = SemiclassicalJacobi(t, 1, 1, 1, P)

    @time @test P[0.1,1:10] ≈ Normalized(legendre(0..1))[0.1,1:10]
    @test P¹¹[0.1,1:10] ≈ Normalized(jacobi(1,1,0..1))[0.1,1:10]/Normalized(jacobi(1,1,0..1))[0.1,1]
    x = axes(P,1)
    @time @test LanczosPolynomial(t .- x, legendre(0..1))[0.1,1:10] ≈ Q[0.1,1:10] / sqrt(sum(orthogonalityweight(Q)))

    @time @testset "expansion" begin
        @test (P * (P \ exp.(x)))[0.1] ≈ exp(0.1)
        @test (Q * (Q \ exp.(x)))[0.1] ≈ exp(0.1)
        @test (R * (R \ exp.(x)))[0.1] ≈ exp(0.1)
    end
end

@testset "Semiclassical operator asymptotics" begin
    t = 2.2
    P = SemiclassicalJacobi(t, 0, 0, 0)
    # ratio asymptotics
    φ = z -> (z + sqrt(z-1)sqrt(z+1))/2
    U = ChebyshevU()

    @testset "ratio asymptotics" begin
        n = 200; 
        @test 2φ(t)*Base.unsafe_getindex(U,t,n)/(Base.unsafe_getindex(U,t,n+1)) ≈ 1
        @test 2φ(2t-1)*Base.unsafe_getindex(P,t,n)/(Base.unsafe_getindex(P,t,n+1)) ≈ 1 atol=1E-3

    
        L1 = P \ WeightedSemiclassicalJacobi(t,0,0,1,P)
        @test L1[n+1,n]/L1[n,n] ≈ -1/(2φ(2t-1)) atol=1E-3
    end

    @testset "single raising" begin
        R_0 = SemiclassicalJacobi(t, 1, 0, 0, P) \ P;
        R_1 = SemiclassicalJacobi(t, 0, 1, 0, P) \ P;
        R_t = SemiclassicalJacobi(t, 0, 0, 1, P) \ P;

        sqrt(0.5)
        @test R_0[999,999:1000] ≈ sqrt.([0.5,0.5]) atol=1e-2
        @test R_1[999,999:1000] ≈ [sqrt(0.5),-sqrt(0.5)] atol=1e-2
        @test R_t[200,201]/R_t[200,200] ≈ -1/(2*φ(2t-1)) atol=1e-2 
    end

    @testset "T,V,W,U" begin
        T = SemiclassicalJacobi(t, -1/2, 0, -1/2)
        V = SemiclassicalJacobi(t, -1/2, 0, 1/2, T)
        U = SemiclassicalJacobi(t,  1/2, 0, 1/2, T)

        R_t = V \ T;
        n = 1000
        c = -1/(2φ(2*t-1))
        @test R_t[n,n+1]/R_t[n,n] ≈ c atol=1E-3
        R = U \ T;
        @test R[n,n+1]/R[n,n] ≈ 1+c atol=1E-3
        @test R[n,n+2]/R[n,n]  ≈ c atol=1E-3
        L =T \ (SemiclassicalJacobiWeight(t, 1, 0, 1) .* U)
        @test L[n+1,n]/L[n,n] ≈ 1+c atol=1E-3
        @test L[n+2,n]/L[n,n] ≈ c atol=1E-3
    end

    @testset "P,Q" begin
        Q = SemiclassicalJacobi(t, 1, 1, 1, P)
        R = Q \ P
        c = -1/(2*φ(2t-1))
        # (1 + c*z)*(1-z^2) == 1 + c*z - z^2 - c*z^2
        @test R[200,201]/R[200,200] ≈ c atol=1e-2
        @test R[200,202]/R[200,200] ≈ -1 atol=1e-2
        @test R[200,203]/R[200,200] ≈ -c atol=1e-2
    end
end

@testset "equilibrium measure" begin
    ρ = 0.5
    t = inv(1-ρ^2)
    P = SemiclassicalJacobi(t, -1/2, -1/2, 0)
    Q = SemiclassicalJacobi(t, -1/2, -1/2, 1/2)

    P̃ = SemiclassicalJacobi(t, 1/2, 1/2, -1);
    Q̃ = SemiclassicalJacobi(t, 1/2, 1/2, 0, Q);

    jacobimatrix(P)[1:5,1:5]
    jacobimatrix(P̃)[1:5,1:5]

    jacobimatrix(Q)[1:5,1:5]
    jacobimatrix(Q̃)[1:5,1:5]
end

@testset "Normalized" begin
    # here we derive the formula for the Jacobi operators
    for (a,b,c) in ((-0.5,-0.4,-0.3),(0,0,0))
        P = SemiclassicalJacobi(2,a,b,c)
        Q = SemiclassicalJacobi(2,a,b,c+1,P)

        X_P = jacobimatrix(P)
        L = (P \ (SemiclassicalJacobiWeight(2,0,0,1) .* Q))
        X = jacobimatrix(Normalized(Q))
        ℓ = OrthogonalPolynomialRatio(P,2)
        a,b = X_P.dv,X_P.ev
        @test X[1,1] ≈ a[1]-b[1]ℓ[1]
        @test X[1,2] ≈ X[2,1] ≈ sqrt(b[1] * (a[1]ℓ[1]+b[1]*(1-ℓ[1]^2)-a[2]ℓ[1]))
        for n = 2:5
            @test X[n,n] ≈ ℓ[n-1]b[n-1]+a[n]-b[n]ℓ[n]
            @test X[n,n+1] ≈ X[n+1,n] ≈ sqrt(b[n] * (b[n-1]ℓ[n-1]ℓ[n]+a[n]ℓ[n]+b[n]*(1-ℓ[n]^2)-a[n+1]ℓ[n]))
        end

        T = Float64
        bl = b .* ℓ
        X̃ = ClassicalOrthogonalPolynomials.SymTridiagonal(a .- bl .+ Vcat(zero(T),bl),
                           sqrt.(bl .* a .+ b .^2 .- bl .^ 2 .- b .* a[2:∞] .* ℓ .+ Vcat(zero(T),bl .* bl[2:∞])));
        @test X̃[1:10,1:10] ≈ X[1:10,1:10]
    end
end

include("test_twointerval.jl")
include("test_derivative.jl")