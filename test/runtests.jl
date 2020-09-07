using SemiclassicalOrthogonalPolynomials, OrthogonalPolynomialsQuasi, ContinuumArrays, BandedMatrices, Test

##
# Arc
##

@testset "Arc OPs" begin
    P₊ = jacobi(0,1/2,0..1)
    x = axes(P₊,1)
    U = LanczosPolynomial(@.(sqrt(1 - x^2)), P₊)

    P₋ = jacobi(0,-1/2,0..1)
    T = LanczosPolynomial(@.(1/sqrt(1 - x^2)), P₋)

    @test bandwidths(U.P \ T.P) == (0,1)
    @test U.w == U.w
    R = U \ T;
    BandedMatrix(view(R,1:10,1:10), (0,2))
end