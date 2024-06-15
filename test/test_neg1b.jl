@testset "Jacobi matrix" begin
    for t in (1.2, 2.3, 5.0)
        for a in (1.5, -1 / 2, 1 / 2, 3 / 2)
            for c in (0.3, -1 / 2, 1 / 2, 3 / 2)
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
end

@testset "Evaluation" begin
    for t in (1.2, 2.3, 5.0)
        for a in (1.5, -1 / 2, 1 / 2, 3 / 2)
            for c in (0.3, -1 / 2, 1 / 2, 3 / 2)
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

# Differentiation
t, a, b, c = 2.0, 1.0, -1.0, 1.0
P = SemiclassicalJacobi(t, a, b, c)
Pb = SemiclassicalJacobi(t, a, -b, c)
PL = SemiclassicalJacobi(t, a, -b - 1, c)
Weighted(PL) \ Weighted(Pb)