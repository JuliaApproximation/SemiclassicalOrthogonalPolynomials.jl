module SemiclassicalOrthogonalPolynomials
using OrthogonalPolynomialsQuasi, FillArrays, LazyArrays, ArrayLayouts, QuasiArrays, InfiniteArrays, ContinuumArrays, LinearAlgebra, BandedMatrices

import Base: getindex, axes, size, \, /, *, +, -

import ArrayLayouts: MemoryLayout
import BandedMatrices: bandwidths
import LazyArrays: resizedata!, paddeddata, CachedVector, CachedMatrix, LazyMatrix, LazyVector, arguments, ApplyLayout, colsupport
import OrthogonalPolynomialsQuasi: OrthogonalPolynomial, recurrencecoefficients, jacobimatrix, normalize, recurrencecoefficients, _p0
import InfiniteArrays: OneToInf, InfUnitRange
import ContinuumArrays: basis

export LanczosPolynomial, Legendre, Normalized, normalize


end