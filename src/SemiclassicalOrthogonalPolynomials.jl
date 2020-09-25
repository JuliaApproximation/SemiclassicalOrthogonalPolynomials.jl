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


# orthogonal w.r.t. (1-x)^a * x^b * (t-x)^c on [0,1]
struct SemiclassicalJacobi{T} <: OrthogonalPolynomial{T}
    t::T
    a::T
    b::T
    c::T

end

# sqrt(1-(1-x)^2) == sqrt(2x-x^2) == sqrt(x)*sqrt(2-x)
# sqrt(1-(1-x)^2) == sqrt(2x-x^2) == sqrt(x)*sqrt(2-x)


end