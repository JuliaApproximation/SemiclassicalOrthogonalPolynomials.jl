using SemiclassicalOrthogonalPolynomials
using Test
using ClassicalOrthogonalPolynomials
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
                for x in LinRange(0, 1, 10)
                    a₀, b₀ = J[1, 1], J[1, 2]
                    @test x * _neg1b_def(x, 0) ≈ a₀ * _neg1b_def(x, 0) + b₀ * _neg1b_def(x, 1)
                    for n in 1:5
                        cₙ, aₙ, bₙ = @view J[n+1, n:n+2]
                        Pₙ, Pₙ₋₁, Pₙ₊₁ = _neg1b_def.(x, (n, n - 1, n + 1))
                        @test x * Pₙ ≈ cₙ * Pₙ₋₁ + aₙ * Pₙ + bₙ * Pₙ₊₁ atol = 1e-4
                    end
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
        for P in Ps
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

    gs = (x -> exp(x) + sin(x), x -> (1 - x) * cos(x^3), x -> 5.0 + (1 - x))
    dgs = (x -> exp(x) + cos(x), x -> -3x^2 * sin(x^3) * (1 - x), x -> -1.0)
    for (idx, (g, dg)) in enumerate(zip(gs, dgs))
        f = expand(P, g)
        df = expand(P, dg)
        for x in LinRange(0, 1, 100)
            @show x
            @test df[x] ≈ dg(x) atol=1e-5
        end
    end
end