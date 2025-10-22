using SemiclassicalOrthogonalPolynomials, ClassicalOrthogonalPolynomials, LazyArrays, Test
import ClassicalOrthogonalPolynomials: recurrencecoefficients, _BandedMatrix, _p0, Weighted
import LazyArrays: Accumulate, AccumulateAbstractVector
import SemiclassicalOrthogonalPolynomials: MulAddAccumulate, HalfWeighted, toclassical
import LinearAlgebra: norm, triu

@testset "Derivative" begin
    @testset "DivDiffData" begin
        P = SemiclassicalJacobi(2,0,0,0);
        Q = SemiclassicalJacobi(2,1,1,1,P);
        D = SemiclassicalOrthogonalPolynomials.divdiff(Q, P);
        A,B,_ = recurrencecoefficients(P);
        α,β,_ = recurrencecoefficients(Q);
        d = AccumulateAbstractVector(*, A ./ Vcat(1,α));
        v1 = AccumulateAbstractVector(+, B ./ A);
        v2 = MulAddAccumulate(Vcat(0,0,α[2:∞]) ./ α, Vcat(0,β ./ α) ./ α);
        v3 = AccumulateAbstractVector(*, Vcat(A[1]A[2], A[3:∞] ./ α));
        D2 = _BandedMatrix(Vcat(((1:∞) .* d)', (((1:∞) .* (v1 .+ B[2:end]./A[2:end]) .- (2:∞) .* (α .* v2 .+ β ./ α)) .* v3)'), ∞, 2,-1)';
        @test D[1:100, 1:100] ≈ D2[1:100, 1:100]
    end

    @testset "WeightedDifffDataA" begin
        HQ, HP = HalfWeighted{:a}(SemiclassicalJacobi(2.0, 2.0, 1.0, 3.0)), HalfWeighted{:a}(SemiclassicalJacobi(2.0,3.0, 0.0, 2.0))
        Q = HQ.P
        P = HP.P
        t = P.t
        a = Q.a
        A,B,C = recurrencecoefficients(P)
        α,β,γ = recurrencecoefficients(Q)
        d = AccumulateAbstractVector(*, A ./ α)
        v1 = MulAddAccumulate(Vcat(0,0,α[2:∞] ./ α), Vcat(0,β))
        v2 = MulAddAccumulate(Vcat(0,0,A[2:∞] ./ α), Vcat(0,B[1], B[2:end] .* d))
        D = SemiclassicalOrthogonalPolynomials.divdiff(HQ, HP);
        D2 = _BandedMatrix(
            Vcat(
                ((a:∞) .* v2 .- ((a+1):∞) .* Vcat(1,v1[2:end] .* d))',
                (((a+1):∞) .* Vcat(1,d))'), SemiclassicalOrthogonalPolynomials.ℵ₀, 0,1)
        @test D[1:100, 1:100] ≈ D2[1:100, 1:100]
    end

    @testset "WeightedDiffDataB" begin
        HQ, HP = HalfWeighted{:b}(SemiclassicalJacobi(2.0, 2.0, 1.0, 3.0)), HalfWeighted{:b}(SemiclassicalJacobi(2.0,1.0,2.0,2.0))
        Q = HQ.P
        P = HP.P
        t = P.t
        b = Q.b
        A,B,C = recurrencecoefficients(P)
        α,β,γ = recurrencecoefficients(Q)
        d = AccumulateAbstractVector(*, A ./ α)
        d2 = AccumulateAbstractVector(*, A ./ Vcat(1,α))
        v1 = MulAddAccumulate(Vcat(0,0,α[2:∞] ./ α), Vcat(0,β))
        v2 = MulAddAccumulate(Vcat(0,0,A[2:∞] ./ α), Vcat(0,B[1], B[2:end] .* d))
        D = SemiclassicalOrthogonalPolynomials.divdiff(HQ, HP);
        D2 = _BandedMatrix(
            Vcat(
                (-(b:∞) .* v2 .+ ((b+1):∞) .* Vcat(1,v1[2:end] .* d) .+ Vcat(0,(1:∞) .* d2))',
                (-((b+1):∞) .* Vcat(1,d))'), SemiclassicalOrthogonalPolynomials.ℵ₀, 0,1)
        @test D[1:100, 1:100] ≈ D2[1:100, 1:100]
    end

    @testset "WeightedDiffDataC" begin
        HQ, HP = HalfWeighted{:c}(SemiclassicalJacobi(2.0, 2.0, 1.0, 3.0)), HalfWeighted{:c}(SemiclassicalJacobi(2.0, 1.0, 0.0, 4.0))
        Q = HQ.P
        P = HP.P
        t = P.t
        c = Q.c
        A,B,C = recurrencecoefficients(P)
        α,β,γ = recurrencecoefficients(Q)
        d = AccumulateAbstractVector(*, A ./ α)
        d2 = AccumulateAbstractVector(*, A ./ Vcat(1,α))
        v1 = MulAddAccumulate(Vcat(0,0,α[2:∞] ./ α), Vcat(0,β))
        v2 = MulAddAccumulate(Vcat(0,0,A[2:∞] ./ α), Vcat(0,B[1], B[2:end] .* d))
        D = SemiclassicalOrthogonalPolynomials.divdiff(HQ, HP);
        D2 = _BandedMatrix(
              Vcat(
                (-(c:∞) .* v2 .+ ((c+1):∞) .* Vcat(1,v1[2:end] .* d) .+ Vcat(0,(t:t:∞) .* d2))',
                (-((c+1):∞) .* Vcat(1,d))'), SemiclassicalOrthogonalPolynomials.ℵ₀, 0,1)
        @test D[1:100, 1:100] ≈ D2[1:100, 1:100]
    end

    @testset "WeightedDiffDataAB" begin
        HQ, HP = HalfWeighted{:ab}(SemiclassicalJacobi(2.0, 2.0, 1.0, 3.0)), HalfWeighted{:ab}(SemiclassicalJacobi(2.0, 3.0, 2.0, 2.0))
        Q = HQ.P
        P = HP.P
        A,B,_ = recurrencecoefficients(P)
        α,β,_ = recurrencecoefficients(Q)
        a,b = Q.a,Q.b
        d = AccumulateAbstractVector(*, Vcat(1,A) ./ α)
        e = AccumulateAbstractVector(*, Vcat(1,A ./ α))
        f = MulAddAccumulate(Vcat(0,0,A[2:end] ./ α[2:end]), Vcat(0, (B./ α) .* e))
        g = cumsum(β ./ α)
        D2 = _BandedMatrix(Vcat((((a+1):∞) .* e .- ((b+a+1):∞).*f .+ ((a+b+2):∞) .* e .* g )',
            (-((a+b+2):∞)  .* d)'),SemiclassicalOrthogonalPolynomials.ℵ₀,1,0)
        D = SemiclassicalOrthogonalPolynomials.divdiff(HQ, HP);
        @test D[1:100, 1:100] ≈ D2[1:100, 1:100]
    end

    @testset "WeightedDiffDataBC" begin
        HQ, HP = HalfWeighted{:bc}(SemiclassicalJacobi(2.0, 2.0, 1.0, 3.0)), HalfWeighted{:bc}(SemiclassicalJacobi(2.0, 1.0, 2.0, 4.0))
        Q = HQ.P
        P = HP.P
        t = P.t
        A,B,_ = recurrencecoefficients(P)
        α,β,_ = recurrencecoefficients(Q)
        b,c = Q.b,Q.c
        d = AccumulateAbstractVector(*, Vcat(1,A) ./ α)
        e = AccumulateAbstractVector(*, Vcat(1,A ./ α))
        f = MulAddAccumulate(Vcat(0,0,A[2:end] ./ α[2:end]), Vcat(0, (B./ α) .* e))
        g = cumsum(β ./ α)
        D = _BandedMatrix(Vcat((-((t+1)* (0:∞) .+ (t*(b+1) + c+1)) .* e .+ ((c+b+1):∞).*f .- ((b+c+2):∞) .* e .* g )',
                (((b+c+2):∞)  .* d)'),SemiclassicalOrthogonalPolynomials.ℵ₀,1,0)
        D2 = SemiclassicalOrthogonalPolynomials.divdiff(HQ, HP);
        @test D2[1:100, 1:100] ≈ D[1:100, 1:100]
    end

    @testset "WeightedDiffDataAC" begin
        HQ, HP = HalfWeighted{:ac}(SemiclassicalJacobi(2.0, 2.0, 1.0, 3.0)), HalfWeighted{:ac}(SemiclassicalJacobi(2.0, 3.0, 0.0, 4.0))
        Q = HQ.P
        P = HP.P
        t = P.t
        A,B,_ = recurrencecoefficients(P)
        α,β,_ = recurrencecoefficients(Q)
        a,c = Q.a,Q.c
        d = AccumulateAbstractVector(*, Vcat(1,A) ./ α)
        e = AccumulateAbstractVector(*, Vcat(1,A ./ α))
        f = MulAddAccumulate(Vcat(0,0,A[2:end] ./ α[2:end]), Vcat(0, (B./ α) .* e))
        g = cumsum(β ./ α)
        D = _BandedMatrix(Vcat((t* ((a+1):∞) .* e .- ((c+a+1):∞).*f .+ ((a+c+2):∞) .* e .* g )',
                (-((a+c+2):∞)  .* d)'),SemiclassicalOrthogonalPolynomials.ℵ₀,1,0)
        D2 = SemiclassicalOrthogonalPolynomials.divdiff(HQ, HP);
        @test D2[1:100, 1:100] ≈ D[1:100, 1:100]
    end

    @testset "basics" begin
        t = 2
        P = SemiclassicalJacobi(t, -0.5, -0.5, -0.5)
        Q = SemiclassicalJacobi(t,  0.5,  0.5,  0.5, P)
        x = axes(P,1)
        # D = Derivative(x)
        # D = Q \ (D*P) # Why is this suddenly broken

        # @test (D*(P \ exp.(x)))[1:50] ≈ (Q \ exp.(x))[1:50]
        D = diff(P).args[2]
        @test (D * transform(P, exp))[1:50] ≈ transform(Q, exp)[1:50]
    end

    @testset "Derivation" begin
        t = 2
        P = SemiclassicalJacobi(t, -0.5, -0.5, -0.5)
        P̃ = toclassical(P)
        Q = SemiclassicalJacobi(t,  0.5,  0.5,  0.5, P)

        @test (Q \ P̃)[1:10,1:10] ≈ 0.6175596179729587*(Q \ P)[1:10,1:10]

        x = axes(P,1)
        D = Derivative(x)

        for n = 3:10
            u = (D * (P̃ * [[zeros(n);1]; zeros(∞)]))
            @test norm((Q \ u)[1:n-2]) ≤ 3E-12
        end

        L = Q \ (D * P̃);
        # L is bidiagonal
        @test norm(triu(L[1:10,1:10],3)) ≤ 3E-12
        @test L[:,5][1:10] ≈ L[1:10,5]

        A,B,C = recurrencecoefficients(P);
        α,β,γ = recurrencecoefficients(Q);

        k = cumprod(A);
        κ = cumprod(α);
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
            @test Base.unsafe_getindex(P̃,100,n+1) ≈ (k[n]*100^n + j[n]*100^(n-1)) * P̃[0.1,1] rtol=0.001
            @test Base.unsafe_getindex(Q,100,n+1) ≈ (κ[n]*100^n + ξ[n]*100^(n-1)) rtol=0.001
        end

        n = 1
        @test k[1]*P̃[0.1,1] ≈ L[1,2]
        n = 2
        @test L[n,n+1] ≈ n*k[n]/κ[n-1]*P̃[0.1,1]
        @test L[n-1,n+1] ≈ ((n-1)*j[n] - n*k[n]*ξ[n-1]/κ[n-1])*P̃[0.1,1]
        for n = 3:6
            @test L[n,n+1] ≈ n*k[n]/κ[n-1]*P̃[0.1,1]
            @test L[n-1,n+1] ≈ ((n-1)*j[n]/κ[n-2] - n*k[n]*ξ[n-1]/(κ[n-2]κ[n-1]))*P̃[0.1,1]
        end

        dv = n -> isone(n) ? A[1] : k[n]/κ[n-1]
        ev1 = n -> j[n]/k[n]
        ev2 = n -> n ==1 ? 1/κ[n] : ξ[n-1]/κ[n]
        ev3 = n -> n == 1 ? k[n+1] : k[n+1]/κ[n-1]
        # ev = n -> (n-1)*j[n]/κ[n-2] - n*k[n]*ξ[n-1]/(κ[n-2]κ[n-1])

        @test ev2(1) ≈ 1/α[1]

        n = 1
        @test ev3(n) ≈ A[n]A[n+1]
        @test ev2(n+1) ≈ β[n]/(α[n]α[n+1])

        n = 3
        @test dv(n+1) ≈ dv(n) * A[n+1]/α[n]
        @test ev1(n+1) ≈ ev1(n) + B[n+1]/A[n+1]
        @test ev2(n+1) ≈ ev2(n)*α[n]/α[n+1] + β[n]/(α[n]α[n+1])
        @test ev3(n+1) ≈ ev3(n) * A[n+2]/α[n]

        @test L[n,n+1]/P̃[0.1,1] ≈ n * dv(n)
        @test L[n-1,n+1]/P̃[0.1,1] ≈ ((n-1)*j[n]/k[n] - n*ξ[n-1]/κ[n-1]) * k[n]/κ[n-2] ≈
            ((n-1)*(ev1(n-1) + B[n]/A[n]) - n*ξ[n-1]/κ[n-1]) * ev3(n-1) ≈
            ((n-1)*(ev1(n-1) + B[n]/A[n]) - n*(α[n-1]*ev2(n-1) + β[n-1]/α[n-1])) * ev3(n-1)

        n = 1
        @test L[n,n+2]/P̃[0.1,1] ≈ (ev1(1) + B[2]/A[2] - 2*(β[1]/α[1]))*ev3(1)
        n = 2
        @test L[n,n+2]/P̃[0.1,1] ≈ (n*(ev1(n) + B[n+1]/A[n+1]) - (n+1)*(α[n]*ev2(n) + β[n]/α[n])) * ev3(n)
        n = 3
        @test L[n,n+2]/P̃[0.1,1] ≈ (n*(ev1(n) + B[n+1]/A[n+1]) - (n+1)*(α[n]*ev2(n) + β[n]/α[n])) * ev3(n)


        d = AccumulateAbstractVector(*, A ./ Vcat(1,α))
        v1 = AccumulateAbstractVector(+, B ./ A)
        v2 = MulAddAccumulate(Vcat(0,0,α[2:∞]) ./ α, Vcat(0,β ./ α) ./ α);
        v3 = AccumulateAbstractVector(*, Vcat(A[1]A[2], A[3:∞] ./ α))

        @test d[1:10] ≈ dv.(1:10)
        @test v1[1:10] ≈ ev1.(1:10)
        @test v2[1] ≈ 0
        @test v2[2:10] ≈ ev2.(2:10)
        @test v3[1:10] ≈ ev3.(1:10)


        @test [L[n,n+1] for n=1:10]/P̃[0.1,1] ≈ ((1:∞) .* d)[1:10]
        @test [L[n,n+2] for n=1:10]/P̃[0.1,1] ≈ (((1:∞) .* (v1 .+ B[2:end]./A[2:end]) .- (2:∞) .* (α .* v2 .+ β ./ α)) .* v3)[1:10]


        D_M = _BandedMatrix(Vcat(((1:∞) .* d)', (((1:∞) .* (v1 .+ B[2:end]./A[2:end]) .- (2:∞) .* (α .* v2 .+ β ./ α)) .* v3)'), ∞, 2,-1)' * _p0(P̃)
        @test D_M[1:10,1:10] ≈ L[1:10,1:10]
    end

    @testset "Weighted" begin
        t = 2
        P = SemiclassicalJacobi(t, 0, 0, 0)
        Q = SemiclassicalJacobi(t, 1, 1, 1, P)
        D = Derivative(axes(P,1))
        @test (D * Weighted(Q))[0.1,1:5]' ≈ (D * P)[0.1,1:8]'* (P \ Weighted(Q))[1:8,1:5]

        P = SemiclassicalJacobi(t, -1/2, -1/2, -1/2)
        Q = SemiclassicalJacobi(t, 1/2, 1/2, 1/2, P)
        h = 0.00001
        @test (D * Weighted(Q))[0.1,1:5] ≈  (Weighted(Q)[0.1+h,1:5] - Weighted(Q)[0.1,1:5])/h atol=100h
    end

    @testset "HalfWeighted"  begin
        t = 2
        P = SemiclassicalJacobi(t, 1, 1, 1)
        P̃ = Normalized(jacobi(1,1,0..1))
        x = axes(P,1)
        D = Derivative(x)

        @test HalfWeighted{:a}(P)[0.1,1:10] ≈ 0.1*P[0.1,1:10]
        @test HalfWeighted{:b}(P)[0.1,1:10] ≈ (1-0.1)*P[0.1,1:10]
        @test HalfWeighted{:c}(P)[0.1,1:10] ≈ (t-0.1)*P[0.1,1:10]

        t,a,b,c = 2,0.1,0.2,0.3
        P = SemiclassicalJacobi(t, a+1, b, c)
        # Q = SemiclassicalJacobi(t,a,b+1,c+1)
        HP = HalfWeighted{:a}(P)
        h = 0.000001
        @test (D * HP)[0.1,1:10] ≈ (HP[0.1+h,1:10]-HP[0.1,1:10])/h atol=200h


        P = SemiclassicalJacobi(t, a, b+1, c)
        # Q = SemiclassicalJacobi(t,a+1,b,c+1)
        HP = HalfWeighted{:b}(P)
        @test (D * HP)[0.1,1:10] ≈ (HP[0.1+h,1:10]-HP[0.1,1:10])/h atol=1000h

        P = SemiclassicalJacobi(t, a, b, c+1)
        # Q = SemiclassicalJacobi(t,a+1,b+1,c)
        HP = HalfWeighted{:c}(P)
        @test (D * HP)[0.1,1:10] ≈ (HP[0.1+h,1:10]-HP[0.1,1:10])/h atol=2000h
    end

    @testset "Double-HalfWeighted" begin
        t = 2
        P = SemiclassicalJacobi(t, 1, 1, 1)
        x = axes(P,1)
        D = Derivative(x)

        @test HalfWeighted{:ab}(P)[0.1,1:10] ≈ 0.1*(1-0.1)*P[0.1,1:10]
        @test HalfWeighted{:bc}(P)[0.1,1:10] ≈ (1-0.1)*(t-0.1)*P[0.1,1:10]
        @test HalfWeighted{:ac}(P)[0.1,1:10] ≈ 0.1*(t-0.1)*P[0.1,1:10]

        h = 0.000001
        @test (D * HalfWeighted{:ab}(P))[0.1,1:10] ≈ (HalfWeighted{:ab}(P)[0.1+h,1:10]-HalfWeighted{:ab}(P)[0.1,1:10])/h atol=2000h
        @test (D * HalfWeighted{:bc}(P))[0.1,1:10] ≈ (HalfWeighted{:bc}(P)[0.1+h,1:10]-HalfWeighted{:bc}(P)[0.1,1:10])/h atol=2000h
        @test (D * HalfWeighted{:ac}(P))[0.1,1:10] ≈ (HalfWeighted{:ac}(P)[0.1+h,1:10]-HalfWeighted{:ac}(P)[0.1,1:10])/h atol=2000h
    end
end