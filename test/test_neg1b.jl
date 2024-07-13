using SemiclassicalOrthogonalPolynomials
using Test
using ClassicalOrthogonalPolynomials: ClassicalOrthogonalPolynomials, HalfWeighted, jacobimatrix, expand, Weighted, ∞
using LazyArrays
using ContinuumArrays: coefficients
using BandedMatrices: BandedMatrices, band, _BandedMatrix
using FillArrays

@testset "Jacobi matrix" begin
    for t in (1.2, 2.3, 5.0, 2.0)
        for a in (1.5, -1 / 2, 0, 1, 2, 1 / 2, 3 / 2)
            for c in (0.3, -1 / 2, 0, 1, 2, 1 / 2, 3 / 2)
                P = SemiclassicalJacobi(t, a, -1.0, c)
                X = jacobimatrix(P)
                J = X'
                Pb = SemiclassicalJacobi(t, a, 1.0, c)
                _neg1b_def = (x, n) -> n == 0 ? one(x) : (1 - x) * Pb[x, n]
                for x in LinRange(0, 1, 100)
                    a₀, b₀ = J[1, 1], J[1, 2]
                    @test x * _neg1b_def(x, 0) ≈ a₀ * _neg1b_def(x, 0) + b₀ * _neg1b_def(x, 1)
                    for n in 1:25
                        cₙ, aₙ, bₙ = @view J[n+1, n:n+2]
                        Pₙ, Pₙ₋₁, Pₙ₊₁ = _neg1b_def.(x, (n, n - 1, n + 1))
                        @test x * Pₙ ≈ cₙ * Pₙ₋₁ + aₙ * Pₙ + bₙ * Pₙ₊₁ atol = 1e-4
                    end
                end
            end
        end
    end

    @testset "Getting jacobimatrix for (a, 1, c) given (a, -1, c)" begin
        for a in (1 / 2, -1 / 2, 2.0)
            for c in (1 / 2, -1 / 2, 1.0)
                for t in (2.0, 2.5)
                    P = SemiclassicalJacobi(t, a, -1.0, c)
                    Q = SemiclassicalJacobi(t, a, 1.0, c)
                    QQ = SemiclassicalJacobi(t, a, 1.0, c, P)
                    @test Q.X[1:100, 1:100] ≈ QQ.X[1:100, 1:100]
                end
            end
        end
    end
end

@testset "Evaluation" begin
    for t in (1.2, 2.3, 5.0, 2.0)
        for a in (1.5, -1 / 2, 1 / 2, 1, 2, 3, 3 / 2)
            for c in (0.3, -1 / 2, 1 / 2, 3 / 2, 2, 0, 1)
                P = SemiclassicalJacobi(t, a, -1.0, c)
                Pb = SemiclassicalJacobi(t, a, 1.0, c)
                for x in LinRange(0, 1, 100)
                    for n in 1:26
                        Px = P[x, n]
                        Pbx = n == 1 ? one(x) : (1 - x) * Pb[x, n-1]
                        @test Px ≈ Pbx
                    end
                end
            end
        end
    end
end

@testset "Families" begin
    # Not implemented efficiently currently. 
    # TODO: Fix cholesky_jacobimatrix when Δb == 0 and b == -1 so that building families works (currently, it would return a "Polynomials must be orthonormal" error)
    t = 2.0
    P = SemiclassicalJacobi.(t, -1//2:13//2, -1.0, -1//2:13//2) # // so that we get a UnitRange, else broadcasting into a Family fails
    for (i, p) in enumerate(P)
        @test jacobimatrix(p)[1:100, 1:100] == jacobimatrix(SemiclassicalJacobi(t, (-1//2:13//2)[i], -1.0, (-1//2:13//2)[i]))[1:100, 1:100]
    end
end

@testset "Expansions" begin
    Ps = SemiclassicalJacobi.(2, -1//2:5//2, -1.0, -1//2:5//2)
    Ps2 = SemiclassicalJacobi.(2, 0:3, -1.0, 0:3) # used to be broken for integers
    for Ps in (Ps, Ps2)
        # Why does this take SO long for Ps[4]? Without them this takes 40 s, but with them it takes 10m!
        for P in Ps
            P === Ps[4] && continue
            @show 1
            for (idx, g) in enumerate((x -> exp(x) + sin(x), x -> (1 - x) * cos(x^3), x -> 5.0 + (1 - x)))
                f = expand(P, g)
                for x in LinRange(0, 1, 100)
                    @test f[x] ≈ g(x) atol = 1e-9
                end
                x = axes(P, 1)
                @test P[:, 1:20] \ g.(x) ≈ coefficients(f)[1:20]
                if idx == 2
                    @test coefficients(f)[1] ≈ 0 atol = 1e-9
                elseif idx == 3
                    @test coefficients(f)[1:2] ≈ [5.0, 1.0]
                    @test coefficients(f)[3:1000] ≈ zeros(1000 - 3 + 1)
                end
            end
        end
    end
end

@testset "Connections" begin
    function test_connection(t, a, b, c, Δa, Δb, Δc)
        a1, b1, c1 = (a, b, c) .+ (Δa, Δb, Δc)
        g = x -> exp(x) + sin(x)
        P1 = SemiclassicalJacobi(t, a, b, c)
        P2 = SemiclassicalJacobi(t, a1, b1, c1)
        R21 = P1 \ P2
        R12 = P2 \ P1
        f1 = coefficients(expand(P1, g))
        f2 = coefficients(expand(P2, g))
        @test f2[1:100] ≈ ApplyArray(*, R12, f1)[1:100]
        @test f1[1:100] ≈ ApplyArray(*, R21, f2)[1:100]
    end

    @testset "Changing one parameter at a time" begin
        test_connection(2.0, 1.0, -1.0, 2.0, 1.0, 0.0, 0.0)
        test_connection(2.3, 3.0, -1.0, 2.0, 0.0, 1.0, 0.0)
        test_connection(2.5, 1.0, -1.0, 0.0, 1.0, 0.0, 1.0)
        test_connection(2.0, 1.0, -1.0, 2.0, -1.0, 0.0, 0.0)
        test_connection(2.5, 1.0, -1.0, 1.0, 1.0, 0.0, -1.0)
    end

    @testset "Changing two parameters" begin
        test_connection(2.0, 1.0, -1.0, 2.0, 1.0, 1.0, 0.0)
        test_connection(2.3, 3.0, -1.0, 2.0, 1.0, 0.0, 1.0)
        test_connection(2.5, 1.0, -1.0, 0.0, 1.0, 0.0, 1.0)
        test_connection(2.5, 1.0, -1.0, 0.0, 0.0, 1.0, 1.0)
        test_connection(2.0, 1.0, -1.0, 2.0, -1.0, 1.0, 0.0)
        test_connection(2.3, 3.0, -1.0, 2.0, -1.0, 0.0, -1.0)
        test_connection(2.5, 1.0, -1.0, 1.0, -1.0, 0.0, -1.0)
    end

    @testset "Changing three parameters" begin
        test_connection(2.0, 1.0, -1.0, 2.0, 1.0, 1.0, 1.0)
        test_connection(3.5, 2.0, -1.0, 2.0, -1.0, 1.0, 2.0)
    end

    @testset "Doing nothing" begin
        test_connection(2.0, 1.0, -1.0, 2.0, 0.0, 0.0, 0.0)
    end
end

@testset "Differentiation" begin
    t, a, b, c = 2.0, 1.0, -1.0, 1.0
    Rᵦₐ₁ᵪᵗᵃ⁰ᶜ = Weighted(SemiclassicalJacobi(t, a, 0.0, c)) \ Weighted(SemiclassicalJacobi(t, a, 1.0, c))
    Dₐ₀ᵪᵃ⁺¹¹ᶜ⁺¹ = diff(SemiclassicalJacobi(t, a, 0.0, c))
    Rₐ₊₁₁ᵪ₊₁ᵗᵃ⁺¹⁰ᶜ⁺¹ = ApplyArray(inv, SemiclassicalJacobi(t, a + 1, 1.0, c + 1) \ SemiclassicalJacobi(t, a + 1, 0.0, c + 1))
    Dₐ₋₁ᵪᵃ⁺¹⁰ᶜ⁺¹ = Rₐ₊₁₁ᵪ₊₁ᵗᵃ⁺¹⁰ᶜ⁺¹ * Dₐ₀ᵪᵃ⁺¹¹ᶜ⁺¹.args[2] * Rᵦₐ₁ᵪᵗᵃ⁰ᶜ
    b2 = Vcat(0.0, 0.0, Dₐ₋₁ᵪᵃ⁺¹⁰ᶜ⁺¹[band(1)])
    b1 = Vcat(0.0, Dₐ₋₁ᵪᵃ⁺¹⁰ᶜ⁺¹[band(0)])
    data = Hcat(b2, b1)'
    D = _BandedMatrix(data, ∞, -1, 2)
    @test Hcat(Zeros(∞), Dₐ₋₁ᵪᵃ⁺¹⁰ᶜ⁺¹)[1:100, 1:100] ≈ D[1:100, 1:100]
    P = SemiclassicalJacobi(t, a, b, c)
    DP = diff(P)
    @test DP.args[2][1:100, 1:100] ≈ D[1:100, 1:100]
    @test DP.args[1] == SemiclassicalJacobi(t, a + 1, b + 1, c + 1)

    # Evaluation
    gs = (x -> exp(x) + sin(x), x -> (1 - x) * cos(x^3), x -> 5.0 + (1 - x))
    dgs = (x -> exp(x) + cos(x), x -> 3(x - 1) * x^2 * sin(x^3) - cos(x^3), x -> -1.0)
    for (idx, (g, dg)) in enumerate(zip(gs, dgs))
        f = expand(P, g)
        df = diff(f)
        for x in LinRange(0, 1, 100)
            @show x
            @test df[x] ≈ dg(x) atol = 1e-5
        end
    end

    # Test the matrix itself
    dP = SemiclassicalJacobi(t, a + 1, b + 1, c + 1)
    for (g, dg) in zip(gs, dgs)
        f = expand(P, g)
        df = expand(dP, dg)
        @test (coefficients(diff(P))*coefficients(f))[1:100] ≈ coefficients(df)[1:100]
    end
end

@testset "Weighted expansions and derivatives" begin
    t, a, b, c = 2.0, 3.5, -1.0, 1.0
    P = SemiclassicalJacobi(t, a, b, c)
    aP = HalfWeighted{:a}(P)
    cP = HalfWeighted{:c}(P)
    acP = HalfWeighted{:ac}(P)

    # :a 
    g = let a = a
        x -> x^a * exp(x)
    end # Use let to avoid eltype = Any which causes errors in expand from zero(Any)
    f = expand(aP, g)
    @test all(x -> f[x] ≈ g(x), LinRange(0, 1, 100))
    @test coefficients(f)[1] == g(1)
    df = diff(f)
    dg = x -> x^(a - 1) * exp(x) * (a + x)
    @test all(x -> df[x] ≈ dg(x), LinRange(0, 1, 100))

    # :c 
    g = let c = c, t = t
        x -> (t - x)^c * exp(x)
    end
    f = expand(cP, g)
    @test all(x -> f[x] ≈ g(x), LinRange(0, 1, 100))
    @test coefficients(f)[1] ≈ g(1) / (t - 1)^c
    df = diff(f)
    dg = x -> -exp(x) * (t - x)^(c - 1) * (c - t + x)
    @test all(x -> isapprox(df[x], dg(x), atol=1e-9), LinRange(0, 1, 100))

    # :ac 
    g = let a = a, c = c, t = t
        x -> x^a * (t - x)^c * exp(x)
    end
    f = expand(acP, g)
    @test all(x -> f[x] ≈ g(x), LinRange(0, 1, 100))
    @test coefficients(f)[1] ≈ g(1) / (t - 1)^c
    df = diff(f)
    dg = x -> -exp(x) * x^(a - 1) * (t - x)^(c - 1) * (a * (x - t) + x * (c - t + x))
    @test all(x -> isapprox(df[x], dg(x), atol=1e-9), LinRange(0, 1, 100))

    # :b 
    @test_throws ArgumentError expand(HalfWeighted{:b}(P), g)
end

