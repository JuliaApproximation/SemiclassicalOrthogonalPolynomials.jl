using SemiclassicalOrthogonalPolynomials, ClassicalOrthogonalPolynomials, LinearAlgebra
import ClassicalOrthogonalPolynomials: associated

ρ = 0.5
T = TwoBandJacobi(ρ,-1/2,-1/2,1/2)
U = associated(T)
x = axes(T,1)
H = inv.(x .- x')

L = U \ (H * Weighted(T))
c = L[:,2:∞] \ (U \ x)
c[1] = -c[2]

u = Weighted(T) * c
u[0.5000001]

[0.1]