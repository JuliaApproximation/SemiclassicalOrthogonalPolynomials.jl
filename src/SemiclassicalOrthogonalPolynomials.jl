module SemiclassicalOrthogonalPolynomials
using OrthogonalPolynomialsQuasi, FillArrays, LazyArrays, ArrayLayouts, QuasiArrays, InfiniteArrays, ContinuumArrays, LinearAlgebra, BandedMatrices

import Base: getindex, axes, size, \, /, *, +, -

import BandedMatrices: bandwidths
import LazyArrays: resizedata!, paddeddata, CachedVector, CachedMatrix, LazyMatrix, LazyVector, arguments, ApplyLayout
import OrthogonalPolynomialsQuasi: OrthogonalPolynomial, recurrencecoefficients, jacobimatrix, normalize, recurrencecoefficients
import InfiniteArrays: OneToInf, InfUnitRange
import ContinuumArrays: basis

export LanczosPolynomial, Legendre, Normalized, normalize

include("lanczos.jl")

end