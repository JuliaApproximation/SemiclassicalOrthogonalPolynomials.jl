using SemiclassicalOrthogonalPolynomials, ClassicalOrthogonalPolynomials, LazyArrays, Test
import ClassicalOrthogonalPolynomials: recurrencecoefficients, _BandedMatrix, _p0, Weighted
import LazyArrays: Accumulate, AccumulateAbstractVector
import SemiclassicalOrthogonalPolynomials: MulAddAccumulate


@testset "Derivative" begin
    @testset "basics" begin
        t = 2
        P = SemiclassicalJacobi(t, -0.5, -0.5, -0.5)
        Q = SemiclassicalJacobi(t,  0.5,  0.5,  0.5, P)
        x = axes(P,1)
        D = Derivative(x)
        D = Q \ (D*P)

        @test D * (P \ exp.(x)) ≈ Q \ exp.(x)
    end

    @testset "Derivation" begin
        t = 2
        P = SemiclassicalJacobi(t, -0.5, -0.5, -0.5)
        Q = SemiclassicalJacobi(t,  0.5,  0.5,  0.5, P)

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
            @test Base.unsafe_getindex(P.P,100,n+1) ≈ (k[n]*100^n + j[n]*100^(n-1)) * P.P[0.1,1] rtol=0.001
            @test Base.unsafe_getindex(Q,100,n+1) ≈ (κ[n]*100^n + ξ[n]*100^(n-1)) rtol=0.001
        end

        n = 1
        @test k[1]*P.P[0.1,1] ≈ L[1,2]
        n = 2
        @test L[n,n+1] ≈ n*k[n]/κ[n-1]*P.P[0.1,1]
        @test L[n-1,n+1] ≈ ((n-1)*j[n] - n*k[n]*ξ[n-1]/κ[n-1])*P.P[0.1,1]
        for n = 3:6
            @test L[n,n+1] ≈ n*k[n]/κ[n-1]*P.P[0.1,1]
            @test L[n-1,n+1] ≈ ((n-1)*j[n]/κ[n-2] - n*k[n]*ξ[n-1]/(κ[n-2]κ[n-1]))*P.P[0.1,1]
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

        @test L[n,n+1]/P.P[0.1,1] ≈ n * dv(n)
        @test L[n-1,n+1]/P.P[0.1,1] ≈ ((n-1)*j[n]/k[n] - n*ξ[n-1]/κ[n-1]) * k[n]/κ[n-2] ≈
            ((n-1)*(ev1(n-1) + B[n]/A[n]) - n*ξ[n-1]/κ[n-1]) * ev3(n-1) ≈
            ((n-1)*(ev1(n-1) + B[n]/A[n]) - n*(α[n-1]*ev2(n-1) + β[n-1]/α[n-1])) * ev3(n-1)

        n = 1
        @test L[n,n+2]/P.P[0.1,1] ≈ (ev1(1) + B[2]/A[2] - 2*(β[1]/α[1]))*ev3(1)
        n = 2
        @test L[n,n+2]/P.P[0.1,1] ≈ (n*(ev1(n) + B[n+1]/A[n+1]) - (n+1)*(α[n]*ev2(n) + β[n]/α[n])) * ev3(n)
        n = 3
        @test L[n,n+2]/P.P[0.1,1] ≈ (n*(ev1(n) + B[n+1]/A[n+1]) - (n+1)*(α[n]*ev2(n) + β[n]/α[n])) * ev3(n)


        d = AccumulateAbstractVector(*, A ./ Vcat(1,α))
        v1 = AccumulateAbstractVector(+, B ./ A)
        v2 = MulAddAccumulate(Vcat(0,0,α[2:∞]) ./ α, Vcat(0,β ./ α) ./ α);
        v3 = AccumulateAbstractVector(*, Vcat(A[1]A[2], A[3:∞] ./ α))

        @test d[1:10] ≈ dv.(1:10)
        @test v1[1:10] ≈ ev1.(1:10)
        @test v2[1] ≈ 0
        @test v2[2:10] ≈ ev2.(2:10)
        @test v3[1:10] ≈ ev3.(1:10)


        @test [L[n,n+1] for n=1:10]/P.P[0.1,1] ≈ ((1:∞) .* d)[1:10]
        @test [L[n,n+2] for n=1:10]/P.P[0.1,1] ≈ (((1:∞) .* (v1 .+ B[2:end]./A[2:end]) .- (2:∞) .* (α .* v2 .+ β ./ α)) .* v3)[1:10]


        D_M = _BandedMatrix(Vcat(((1:∞) .* d)', (((1:∞) .* (v1 .+ B[2:end]./A[2:end]) .- (2:∞) .* (α .* v2 .+ β ./ α)) .* v3)'), ∞, 2,-1)' * _p0(P.P)
        @test D_M[1:10,1:10] ≈ L[1:10,1:10]
        # accumulate_abs(*,
    end

    @testset "annuli Weighted" begin
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
end