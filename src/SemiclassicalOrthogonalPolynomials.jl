module SemiclassicalOrthogonalPolynomials
using OrthogonalPolynomialsQuasi, FillArrays, LazyArrays, ArrayLayouts, QuasiArrays, InfiniteArrays, ContinuumArrays, LinearAlgebra

import Base: getindex, axes, size

import LazyArrays: resizedata!, paddeddata, CachedVector, CachedMatrix, LazyMatrix, LazyVector, arguments
import OrthogonalPolynomialsQuasi: OrthogonalPolynomial, recurrencecoefficients, jacobimatrix, normalize, recurrencecoefficients
import InfiniteArrays: OneToInf, InfUnitRange
import ContinuumArrays: basis

export LanczosPolynomial, Legendre, Normalized, normalize

include("lanczos.jl")

end