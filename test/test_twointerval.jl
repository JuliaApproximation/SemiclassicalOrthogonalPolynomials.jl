using SemiclassicalOrthogonalPolynomials, Test

function interlace(a::AbstractVector{S},b::AbstractVector{V}) where {S<:Number,V<:Number}
    na=length(a);nb=length(b)
    T=promote_type(S,V)
    if nb≥na
        ret=zeros(T,2nb)
        ret[1:2:1+2*(na-1)]=a
        ret[2:2:end]=b
        ret
    else
        ret=zeros(T,2na-1)
        ret[1:2:end]=a
        if !isempty(b)
            ret[2:2:2+2*(nb-1)]=b
        end
        ret
    end
end


@testset "Chebyshev T case" begin
    a = b = -1/2
    c = 1/2

    ρ = 0.7
    t = inv(1-ρ^2)

    P = SemiclassicalJacobi(t, a, b, c-1/2)
    Q = SemiclassicalJacobi(t, a, b, c+1/2, P)

    L = P \ (SemiclassicalJacobiWeight(t,0,0,1) .* Q);

    p = (n,x) -> iseven(n) ? P[(1-x^2)/(1-ρ^2),n÷2+1] : x * Q[(1-x^2)/(1-ρ^2),n÷2+1]

    x = 0.8
    @test x * p(0,x) ≈ p(1,x)
    for n = 1:2:7
        m = n ÷ 2 + 1
        @test x * p(n,x) ≈ (1-ρ^2) * (p(n-1,x) * L[m,m] + p(n+1,x) * L[m+1,m])
    end
    for n = 2:2:6
        m = n÷2+1
        @test x * p(n,x) ≈  (p(n-1,x) * L[m,m-1] + p(n+1,x) * L[m,m])/L[1,1]
    end

    n =20
    ev = L.ev[1:n]
    dv = L.dv[1:n]
    J = Tridiagonal(interlace((1-ρ^2) * dv,ev/L[1,1]), zeros(2n+1), interlace(dv/L[1,1], (1-ρ^2) * ev));
    @test x * p.(0:2n-1,x) ≈ J[1:end-1,:] * p.(0:2n,x)
    ρ
    sort(eigvals(Matrix(J[2:end,2:end])); lt=(x,y) -> isless(abs(x),abs(y)))
end

@testset "Chebyshev U case" begin
    a = b = 1/2
    c = -1/2

    ρ = 0.7
    t = inv(1-ρ^2)

    P = SemiclassicalJacobi(t, a, b, c-1/2)
    Q = SemiclassicalJacobi(t, a, b, c+1/2)

    L = P \ (SemiclassicalJacobiWeight(t,0,0,1) .* Q);
    @test P[0.1,1:5]' * L[1:5,1:4] ≈ (t - 0.1) * Q[0.1,1:4]'

    p = (n,x) -> iseven(n) ? P[(1-x^2)/(1-ρ^2),n÷2+1] : x * Q[(1-x^2)/(1-ρ^2),n÷2+1]

    x = 0.8
    @test x * p(0,x) ≈ p(1,x)
    for n = 1:2:7
        m = n ÷ 2 + 1
        @test x * p(n,x) ≈ (1-ρ^2) * (p(n-1,x) * L[m,m] + p(n+1,x) * L[m+1,m])
        # exact formula
        @test x * p(n,x) ≈ (1-ρ^2) * (p(n-1,x) * L[1,1] - p(n+1,x)/4)
    end
    for n = 2:2:6
        m = n÷2+1
        @test x * p(n,x) ≈  (p(n-1,x) * L[m,m-1] + p(n+1,x) * L[m,m])/L[1,1]
        # exact formula
        @test x * p(n,x) ≈  -p(n-1,x)/4L[1,1] + p(n+1,x)
    end


    dv = L.dv[1:10]
    ev = L.ev[1:10]
    J = Tridiagonal(interlace((1-ρ^2) * dv,ev/L[1,1]), zeros(21), interlace(dv/L[1,1], (1-ρ^2) * ev))
    @test x * p.(0:19,x) ≈ J[1:end-1,:] * p.(0:20,x)
    sqrt.(J.dl .* J.du)
end