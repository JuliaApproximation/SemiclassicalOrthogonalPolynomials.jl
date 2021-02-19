using SemiclassicalOrthogonalPolynomials, ClassicalOrthogonalPolynomials, ContinuumArrays, BandedMatrices, QuasiArrays, Test, LazyArrays, LinearAlgebra
import BandedMatrices: _BandedMatrix
import SemiclassicalOrthogonalPolynomials: op_lowering, initialα, αcoefficients!, evalϕn, backαcoeff!, αdirect!, αdirect
import ClassicalOrthogonalPolynomials: recurrencecoefficients, orthogonalityweight

@testset "OrthogonalPolynomialRatio" begin
    P = Legendre()
    R = OrthogonalPolynomialRatio(P,0.1)
    @test P[0.1,1:10] ./ P[0.1,2:11] ≈ R[1:10]
end

@testset "Jacobi" begin
    P = Normalized(Legendre())
    L = op_lowering(P,1)
    L̃ = P \ WeightedJacobi(1,0)
    # off by scaling
    @test (L ./ L̃)[5,5] ≈ (L ./ L̃)[6,5]

    t = 2
    P̃ = Normalized(SemiclassicalJacobi(t, 0, 0, 0))
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
        L = T \ (SemiclassicalJacobiWeight(2,1,0,0) .* W)
        @test bandwidths(L) == (1,0)

        @testset "Relationship with Lanczos" begin
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

            @test Normalized(W)[0.1,1:10] ≈ W̃[0.1,1:10]

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
                X_W = W \ (x .* W)
                @test X_W isa ConjugateTridiagonal
                @test X_W[1:11,1:10] ≈ X_W_N
            end
        end

        @testset "Evaluation" begin
            x = axes(W,1)
            A_W,B_W,C_W = recurrencecoefficients(W)
            X_W = W \ (x .* W)

            @test W[0.1,2] ≈ A_W[1]*0.1 + B_W[1]
            @test W[0.1,3] ≈ (A_W[2]*0.1 + B_W[2])*(A_W[1]*0.1 + B_W[1]) - C_W[2]

            @test W[0.1,1:11]'*X_W[1:11,1:10] ≈ 0.1 * W[0.1,1:10]'
            @test 0.1*W[0.1,1:10]' ≈ T[0.1,1:11]' * L[1:11,1:10]

            @test  Normalized(W)[0.1,1:3] ≈ [1.323608096788513,-2.6865315578413216,2.9959128991503907]
        end

        @testset "Mass matrix" begin
            @test (T'*(w_T .* T))[1:10,1:10] ≈ sum(w_T)I
            M = W'*(w_W .* W)
            # emperical from mathematica with recurrence
            # broken since I changed the scaling
            @test_broken M[1:3,1:3] ≈ Diagonal([0.5707963267948967,0.5600313808965515,0.5574362259623227])

            R = W \ T;
            @test T[0.1,1:10]' ≈ W[0.1,1:10]' * R[1:10,1:10]
        end
    end

    @testset "T and V" begin
        T = SemiclassicalJacobi(2, -1/2, 0, -1/2)
        V = SemiclassicalJacobi(2, -1/2, 0, 1/2, T)
        w_T = orthogonalityweight(T)
        w_V = orthogonalityweight(V)
        X = jacobimatrix(T)
        A, B, C = recurrencecoefficients(T)
        L = T \ (SemiclassicalJacobiWeight(2,0,0,1) .* V)
        @test eltype(L) == Float64
        @test bandwidths(L) == (1,0)

        @testset "Relationship with Lanczos" begin
            P₋ = jacobi(0,-1/2,0..1)
            x = axes(T,1)
            Ṽ = LanczosPolynomial(@.(sqrt(2-x)/sqrt(x)), P₋)
            A_Ṽ, B_Ṽ, C_Ṽ = recurrencecoefficients(Ṽ)

            @test Normalized(V)[0.1,1:10] ≈ Ṽ[0.1,1:10]

            # this expresses W in terms of W̃
            kᵀ = cumprod(A)
            k_Ṽ = cumprod(A_Ṽ)
            V̄ = (x,n) -> n == 1 ? one(x) : -kᵀ[n]*L[n+1,n]/(Ṽ[0.1,1]k_Ṽ[n-1]) *Ṽ[x,n]

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
                @test X_V isa ConjugateTridiagonal
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

    @testset "U" begin
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
            L̄_3 = SemiclassicalOrthogonalPolynomials.InvMulBidiagonal(L̃_1, L)
            @test L̄_3[1:11,1:10] ≈ L̃_3

            L_3 = W \ (SemiclassicalJacobiWeight(2,0,0,1) .* U);
            @test (2-0.1) * U[0.1,1:10]' ≈ W[0.1,1:11]' * L_3[1:11,1:10]

            R = U \ T;
            @test T[0.1,1:10]' ≈ U[0.1,1:10]' * R[1:10,1:10]
        end

        @testset "Evaluation" begin
            @test Normalized(U)[0.1,1:4] ≈ [1.1283791670955114, -2.03015300715061, 2.131688648563451, -1.3816046198529939]
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

    @testset "Normalized" begin
        T = SemiclassicalJacobi(2, -0.5, 0, -0.5)
        T̃ = Normalized(T)
        @test T̃[0.1,1:10] ≈ T[0.1,1:10]/sqrt(sum(orthogonalityweight(T)))
        U = SemiclassicalJacobi(2, 0.5, 0, 0.5, T)
        Ũ = Normalized(U)
        K = U \ Ũ
        Ki = Ũ \ U
        @test Ũ[0.1,1:10]' ≈ U[0.1,1:10]'* K[1:10,1:10]
        @test Ũ[0.1,1:10]'* Ki[1:10,1:10] ≈ U[0.1,1:10]'
        X_U = jacobimatrix(U)
        X_Ũ = jacobimatrix(Ũ);
        @test X_Ũ[1:10,1:10] ≈ Ki[1:10,1:10] * X_U[1:10,1:10] * K[1:10,1:10]
        R = U \ T;
        R̃ = Ũ \ T;
        @test R̃[1:10,1:10] ≈ Ki[1:10,1:10] * R[1:10,1:10]
        R̃ = Ũ \ T̃;
        @test R̃[1:10,1:10] ≈ Ki[1:10,1:10] * R[1:10,1:10]/sqrt(sum(orthogonalityweight(T)))

        L̃ = T \ (SemiclassicalJacobiWeight(2,1,0,1) .* Ũ);
        @test (2-0.1)*0.1*Ũ[0.1,1:10]' ≈ T[0.1,1:12]'* L̃[1:12,1:10]
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

    @test P[0.1,1:10] ≈ Normalized(legendre(0..1))[0.1,1:10]
    @test Normalized(P¹¹)[0.1,1:10] ≈ 2Normalized(jacobi(1,1,0..1))[0.1,1:10]
    x = axes(P,1)
    @test LanczosPolynomial(t .- x, legendre(0..1))[0.1,1:10] ≈ Normalized(Q)[0.1,1:10]

    @testset "expansion" begin
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
        R_0 = Normalized(SemiclassicalJacobi(t, 1, 0, 0, P)) \ Normalized(P);
        R_1 = Normalized(SemiclassicalJacobi(t, 0, 1, 0, P)) \ Normalized(P);
        R_t = Normalized(SemiclassicalJacobi(t, 0, 0, 1, P)) \ Normalized(P);

        @test R_0[999,999:1000] ≈ [0.5,0.5] atol=1e-2
        @test R_1[999,999:1000] ≈ [0.5,-0.5] atol=1e-2
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

@testset "OPs for a=b=0, c=-1" begin
    @testset "inital α" begin
        t1 = 1.1
        t2 = 1.841
        t3 = 3.91899
        t4 = BigFloat("1.0000000000000000000001")
        # Mathematica
        @test initialα(t1) ≈ 0.4430825224938773
        @test initialα(t2) ≈ 0.1980462516542294
        @test initialα(t3) ≈ 0.0865853392346796
        @test initialα(t4) ≈ 0.9610516212042500
    end

    @testset "αdirect! consistency" begin
        N = 5
        t1 = 1.1
        t2 = 1.841
        t3 = 3.91899
        t4 = BigFloat("1.0000000000000000000001")
        α1 = zeros(eltype(float(t1)), N)
        α2 = zeros(eltype(float(t2)), N)
        α3 = zeros(eltype(float(t3)), N)
        α4 = zeros(eltype(float(t3)), N)
        αdirect!(α1,t1,1:N)
        αdirect!(α2,t2,1:N)
        αdirect!(α3,t3,1:N)
        αdirect!(α4,t4,1:N)
        @test αdirect.((1:N),t1) ≈ α1
        @test αdirect.((1:N),t2) ≈ α2
        @test αdirect.((1:N),t3) ≈ α3
        @test αdirect.((1:N),t4) ≈ α4
    end

    @testset "OPs for a=b=0, c=-1 - basic forward and back recurrence" begin
        # set parameters
        N = 30
        t1 = BigFloat("1.841")
        t2 = BigFloat("1.0000000000000000000001")
        # initialize α
        α1f = zeros(eltype(float(t1)), N)
        α1f[1] = initialα(t1)
        α2f = zeros(eltype(float(t2)), N)
        α2f[1] = initialα(t2)
        α1b = zeros(eltype(float(t1)), N)
        α2b = zeros(eltype(float(t2)), N)
        # compute coefficients via forward recurrence
        αcoefficients!(α1f,t1,BigInt.(2:N))
        αcoefficients!(α2f,t2,BigInt.(2:N))
        # compute coefficients via back recurrence
        backαcoeff!(α1b,t1,BigInt.(2:N))
        backαcoeff!(α2b,t2,BigInt.(2:N))
        # Mathematica α1
        @test α1b[4]  ≈ 0.262708732908399743 ≈ α1f[4]  
        @test α1b[6]  ≈ 0.272687692260606064 ≈ α1f[6]  
        @test α1b[10] ≈ 0.281264391787711543 ≈ α1f[10] 
        @test α1b[20] ≈ 0.288083843194443346 ≈ α1f[20]
        # Mathematica α2
        @test α2b[3]  ≈ 0.98621165663772362 ≈ α2f[3]
        @test α2b[5]  ≈ 0.99152243369113373 ≈ α2f[5]
        @test α2b[10] ≈ 0.99562287407137481 ≈ α2f[10]  
        @test α2b[20] ≈ 0.99774034482793111 ≈ α2f[20] 
    end

    @testset "OPs for a=b=0, c=-1 - high n, high and low t back recurrence" begin
        # set parameters
        N = 10000
        t0 = BigFloat("2.0")
        t1 = BigFloat("371.138")
        t2 = BigFloat("1.0000000000000000000001")
        t3 = BigFloat("500")
        # initialize α
        α0 = zeros(eltype(float(t0)), N)
        α1 = zeros(eltype(float(t1)), N)
        α2 = zeros(eltype(float(t2)), N)
        α3 = zeros(eltype(float(t3)), N)
        # compute coefficients via back recurrence
        backαcoeff!(α0,t0,BigInt.(2:N))
        backαcoeff!(α1,t1,BigInt.(2:N))
        backαcoeff!(α2,t2,BigInt.(2:N))
        backαcoeff!(α3,t3,BigInt.(2:N))
        # Mathematica α1
        @test α0[1] ≈ Float64(initialα(t0))
        @test α1[1] ≈ Float64(initialα(t1))
        @test α2[1] ≈ Float64(initialα(t2))
        @test α3[1] ≈ Float64(initialα(t3))
    end

    @testset "OPs for a=b=0, c=-1 - evaluation" begin
        t = 1.1
        x = 0.1
        n = 5
        α = zeros(n+1)'
        α[1] = initialα(t)
        αcoefficients!(α,t,2:n)
        evalϕn(1,0.1,α)
        # compare versions with and without recomputing α with Mathematica results
        @test evalϕn(0,x,α) == evalϕn(0,x,t) == 2
        @test evalϕn(1,x,α) ≈ evalϕn(1,x,t) ≈ 0.3430825224938977
        @test evalϕn(2,x,α) ≈ evalϕn(2,x,t) ≈ -0.5371542038747678
    end

    @testset "OPs for a=b=0, c=-1 - B operator consistency" begin
        # basis
        t = 1.1
        P = Legendre()
        x = axes(P,1)
        # compute coefficients for basis
        N = 10
        α = zeros(N)'
        α[1] = initialα(t)
        αcoefficients!(α,t,2:N)
        # generate B operator
        B = BandedMatrices._BandedMatrix(Vcat((-1).^(1:N)' .* α,(-1).^(0:N-1)'), N, 0,1)
        # represent some ϕ_n(x) in Legendre polynomial basis
        n = 1
        ϕ1 = P \ @.(α[n+1]*ClassicalOrthogonalPolynomials.jacobip(n-1,0,0,-x)+ClassicalOrthogonalPolynomials.jacobip(n,0,0,-x))
        n = 2
        ϕ2 = P \ @.(α[n+1]*ClassicalOrthogonalPolynomials.jacobip(n-1,0,0,-x)+ClassicalOrthogonalPolynomials.jacobip(n,0,0,-x))
        # test B operator consistency
        @test ϕ1[1:N] ≈ B[1:N,2]
        @test (B \ ϕ1[1:N])[2] ≈ 1
        @test ϕ2[1:N]≈ B[1:N,3]
        @test (B \ ϕ2[1:N])[3] ≈ 1
    end

    @testset "OPs for a=b=0, c=-1 - basic evaluation consistency" begin
        # basis
        t = 1.1
        P = Legendre()
        x = axes(P,1)
        # compute coefficients for basis
        N = 20
        α = zeros(N)'
        α[1] = initialα(t)
        αcoefficients!(α,t,2:N)
        # generate B operator
        B = BandedMatrices._BandedMatrix(Vcat((-1).^(1:N)' .* α,(-1).^(0:N-1)'), N, 0,1)
        # some test functions
        f1(x) = x^2
        f2(x) = (t-x)^2
        f3(x) = exp(t-x)
        # test basic expansion and evaluation via Legendre()
        y = 2*rand(1)[1]-1
        u1 = B \ (P[:,1:20] \ f1.(x))
        @test (P[:,1:20]*(B*u1))[y] ≈ f1(y)
        u2 = B \ (P[:,1:20] \ f2.(x))
        @test (P[:,1:20]*(B*u2))[y] ≈ f2(y)
        u3 = B \ (P[:,1:20] \ f3.(x))
        @test (P[:,1:20]*(B*u3))[y] ≈ f3(y)
    end
end

@testset "Derivative" begin
    @testset "basic Derivative" begin
        t = 2
        P = SemiclassicalJacobi(t, -0.5, -0.5, -0.5)
        Q = SemiclassicalJacobi(t, 0.5, 0.5, 0.5, P)

        @test (Q \ P.P)[1:10,1:10] ≈ 0.6175596179729587*(Q \ P)[1:10,1:10]

        x = axes(P,1)
        D = Derivative(x)

        for n = 3:10
            u = (D * (P.P * [[zeros(n);1]; zeros(∞)]))
            @test norm((Q \ u)[1:n-2]) ≤ 1000eps()
        end

        L = Q \ (D * P.P);
        # L is bidiagonal
        @test norm(triu(L[1:10,1:10],3)) ≤ 1000eps()
        @test L[:,5] isa Vcat

        A,B,C = recurrencecoefficients(P)
        α,β,γ = recurrencecoefficients(Q)

        P[0.1,1]
        Q[0.1,1]

        k = cumprod(A)
        κ = cumprod(α)
        j = Vector{Float64}(undef, 100)
        j[1] = B[1]
        for n = 1:length(j)-1
            j[n+1] = A[n+1]*j[n] + B[n+1]*k[n]
        end
        ξ = Vector{Float64}(undef, 100)
        ξ[1] = β[1]
        for n = 1:length(ξ)-1
            ξ[n+1] = α[n+1]*ξ[n] + β[n+1]*κ[n]
        end

        for n = 3:5
            @test Base.unsafe_getindex(P.P,100,n+1) ≈ (k[n]*100^n + j[n]*100^(n-1)) * P.P[0.1,1] rtol=0.001
            @test Base.unsafe_getindex(Q,100,n+1) ≈ (κ[n]*100^n + ξ[n]*100^(n-1)) rtol=0.001
        end



        
        @test k[1]*P.P[0.1,1] ≈ L[1,2]
        n = 2
        @test L[n,n+1] ≈ n*k[n]/κ[n-1]*P.P[0.1,1]
        @test L[n-1,n+1] ≈ ((n-1)*j[n] - n*k[n]*ξ[n-1]/κ[n-1])*P.P[0.1,1]
        for n = 3:6
            @test L[n,n+1] ≈ n*k[n]/κ[n-1]*P.P[0.1,1]
            @test L[n-1,n+1] ≈ ((n-1)*j[n]/κ[n-2] - n*k[n]*ξ[n-1]/(κ[n-2]κ[n-1]))*P.P[0.1,1]
        end

        n = 3
        dv = n -> k[n]/κ[n-1]
        ev1 = n -> j[n]/k[n]
        ev2 = n -> ξ[n-1]/κ[n]
        ev3 = n -> k[n]/κ[n-2]
        # ev = n -> (n-1)*j[n]/κ[n-2] - n*k[n]*ξ[n-1]/(κ[n-2]κ[n-1])
        @test dv(n+1) ≈ dv(n) * A[n+1]/α[n]

        n = 3
        @test ev1(n+1) ≈ ev1(n) + B[n+1]/A[n+1]
        @test ev2(n+1) ≈ ev2(n)*α[n]/α[n+1] + β[n]/(α[n]α[n+1])
        @test ev3(n+1) ≈ ev3(n) * A[n+1]/α[n-1]

        @test ((n-1)*j[n]/k[n] - n*ξ[n-1]/κ[n-1]) * k[n]/κ[n-2] ≈
            ((n-1)*(ev1(n-1) + B[n]/A[n]) - n*ξ[n-1]/κ[n-1]) * ev3(n) ≈
            ((n-1)*(ev1(n-1) + B[n]/A[n]) - n*(α[n-1]*ev2(n-1) + β[n-1]/α[n-1])) * ev3(n)
    end     

    @testset "annuli D_-" begin
        t = 2
        P = SemiclassicalJacobi(t, 0, 0, 0)
        Q = SemiclassicalJacobi(t, 1, 1, -1)
    end
end


@testset "Normalized" begin
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

