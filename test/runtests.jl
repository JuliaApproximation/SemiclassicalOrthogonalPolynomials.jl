using SemiclassicalOrthogonalPolynomials, OrthogonalPolynomialsQuasi, ContinuumArrays, BandedMatrices, Test

##
# Arc
##

x = Inclusion(0..1)
w = sqrt.(1 .- x .^2)
LanczosPolynomial(w)

OrthogonalPolynomialsQuasi.LegendreWeight()[affine(x,axes(wP,1))]

wP = WeightedJacobi(0,1/2); wP = wP[affine(x,axes(wP,1)),:]

