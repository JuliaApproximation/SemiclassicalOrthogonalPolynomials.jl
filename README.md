# SemiclassicalOrthogonalPolynomials.jl
A Julia repository for semiclassical orthogonal polynomials


[![Build Status](https://github.com/JuliaApproximation/SemiclassicalOrthogonalPolynomials.jl/workflows/CI/badge.svg)](https://github.com/JuliaApproximation/SemiclassicalOrthogonalPolynomials.jl/actions)
[![codecov](https://codecov.io/gh/JuliaApproximation/SemiclassicalOrthogonalPolynomials.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaApproximation/SemiclassicalOrthogonalPolynomials.jl)

This package implements `SemiclassicalJacobi`, a family of orthogonal 
polynomials orthogonal with respect to the weight `x^a * (1-x)^b * (t-x)^c`. 
This builds on top of [ClassicalOrthogonalPolynomials.jl](https://github.com/JuliaApproximation/ClassicalOrthogonalPolynomials.jl) and usage is similar.

For example, the following gives a half-range Chebyshev polynomial:
```julia
julia> using SemiclassicalOrthogonalPolynomials
julia> T = SemiclassicalJacobi(2, -1/2, 0, -1/2)
SemiclassicalJacobi with weight x^-0.5 * (1-x)^0.0 * (2.0-x)^-0.5 on 0..1

julia> T[0.1,1:10] # evaluate first 10 OPs at 0.1
10-element Array{Float64,1}:
  1.0
 -0.855801766003832
  0.19083013661761547
  0.5574589013555691
 -1.085921276099753
  1.181713489691121
 -0.8056271765695796
  0.10748539771183807
  0.6338334369113602
 -1.1219700834800677

julia> U = SemiclassicalJacobi(2, 1/2, 0, 1/2, T) # last argument reuses computation from T
SemiclassicalJacobi with weight x^0.5 * (1-x)^0.0 * (2.0-x)^0.5 on 0..1
```
