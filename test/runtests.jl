using SemiclassicalOrthogonalPolynomials, OrthogonalPolynomialsQuasi, ContinuumArrays, BandedMatrices, QuasiArrays, Test

##
# Arc
##

@testset "Arc OPs" begin
    P₊ = jacobi(0,1/2,0..1)
    x = axes(P₊,1)
    y = @.(sqrt(1 - x^2))

    U = LanczosPolynomial(y, P₊)

    P₋ = jacobi(0,-1/2,0..1)
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