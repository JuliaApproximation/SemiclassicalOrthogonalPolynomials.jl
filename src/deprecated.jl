# The old ldiv variants are still supported for now but deprecated and implemented using non-efficient constructors
Base.@deprecate semijacobi_ldiv_direct(Qt, Qa, Qb, Qc, P::SemiclassicalJacobi) semijacobi_ldiv_direct(SemiclassicalJacobi(Qt, Qa, Qb, Qc), P)
Base.@deprecate semijacobi_ldiv(Qt, Qa, Qb, Qc, P::SemiclassicalJacobi) semijacobi_ldiv(SemiclassicalJacobi(Qt, Qa, Qb, Qc), P)
