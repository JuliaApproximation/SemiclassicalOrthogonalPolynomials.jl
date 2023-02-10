# These functions are in the PR as workarounds
# because a handful of minor bugs prevent their direct implementations for now

# returns conversion operator from SemiclassicalJacobi P to SemiclassicalJacobi Q.
function ConversionOperator(P,Q)
    @assert Q.t ≈ P.t
    M = I
    if !iszero(P.a-Q.a)
        M = (P.X)^(Q.a-P.a)
    end
    if !iszero(P.b-Q.b)
        M = M*(I-P.X)^(Q.b-P.b)
    end
    if !iszero(P.c-Q.c)
        M = M*(Q.t*I-P.X)^(Q.c-P.c)
    end
    # the next line is a workaround for a Symtridiagonal / Symmetric bug
    M = Symmetric(BandedMatrix(0 => M.dv, 1 => M.ev))
    K = (cholesky(M).U)
    return ApplyArray(*, K, Diagonal(Fill(1/K[1],∞))) # match our normalization choice P_0(x) = 1
end