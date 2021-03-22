using SemiclassicalOrthogonalPolynomials
import SemiclassicalOrthogonalPolynomials: initialα, αdirect, αdirect!, backαcoeff!, αcoefficients!, evalϕn, neg1c_tolegendre, evalQn

@testset "inital α" begin
    t1 = 1.1
    t2 = 1.841
    t3 = 3.91899
    t4 = BigFloat("1.0000000000000000000001")
    # Mathematica
    @test initialα(t1) ≈ 0.4430825224938773
    @test initialα(t2) ≈ 0.1980462516542294
    @test initialα(t3) ≈ 0.0865853392346796
    @test initialα(t4) ≈ 0.9610516212042500
end

@testset "αdirect! consistency" begin
    N = 5
    t1 = 1.1
    t2 = 1.841
    t3 = 3.91899
    t4 = BigFloat("1.0000000000000000000001")
    α1 = zeros(eltype(float(t1)), N)
    α2 = zeros(eltype(float(t2)), N)
    α3 = zeros(eltype(float(t3)), N)
    α4 = zeros(eltype(float(t3)), N)
    αdirect!(α1,t1,1:N)
    αdirect!(α2,t2,1:N)
    αdirect!(α3,t3,1:N)
    αdirect!(α4,t4,1:N)
    @test αdirect.((1:N),t1) ≈ α1
    @test αdirect.((1:N),t2) ≈ α2
    @test αdirect.((1:N),t3) ≈ α3
    @test αdirect.((1:N),t4) ≈ α4
end

@testset "OPs for a=b=0, c=-1 - basic forward and back recurrence" begin
    # set parameters
    N = 30
    t1 = BigFloat("1.841")
    t2 = BigFloat("1.0000000000000000000001")
    # initialize α
    α1f = zeros(eltype(float(t1)), N)
    α1f[1] = initialα(t1)
    α2f = zeros(eltype(float(t2)), N)
    α2f[1] = initialα(t2)
    α1b = zeros(eltype(float(t1)), N)
    α2b = zeros(eltype(float(t2)), N)
    # compute coefficients via forward recurrence
    αcoefficients!(α1f,t1,BigInt.(2:N))
    αcoefficients!(α2f,t2,BigInt.(2:N))
    # compute coefficients via back recurrence
    backαcoeff!(α1b,t1,BigInt.(1:N))
    backαcoeff!(α2b,t2,BigInt.(1:N))
    # Mathematica α1
    @test α1b[4]  ≈ 0.262708732908399743 ≈ α1f[4]  
    @test α1b[6]  ≈ 0.272687692260606064 ≈ α1f[6]  
    @test α1b[10] ≈ 0.281264391787711543 ≈ α1f[10] 
    @test α1b[20] ≈ 0.288083843194443346 ≈ α1f[20]
    # Mathematica α2
    @test α2b[3]  ≈ 0.98621165663772362 ≈ α2f[3]
    @test α2b[5]  ≈ 0.99152243369113373 ≈ α2f[5]
    @test α2b[10] ≈ 0.99562287407137481 ≈ α2f[10]  
    @test α2b[20] ≈ 0.99774034482793111 ≈ α2f[20] 
end

@testset "OPs for a=b=0, c=-1 - high n, high and low t back recurrence" begin
    # set parameters
    N = 10000
    t0 = BigFloat("2.0")
    t1 = BigFloat("371.138")
    t2 = BigFloat("1.0000000000000000000001")
    t3 = BigFloat("500")
    # initialize α
    α0 = zeros(eltype(float(t0)), N)
    α1 = zeros(eltype(float(t1)), N)
    α2 = zeros(eltype(float(t2)), N)
    α3 = zeros(eltype(float(t3)), N)
    # compute coefficients via back recurrence
    backαcoeff!(α0,t0,BigInt.(1:N))
    backαcoeff!(α1,t1,BigInt.(1:N))
    backαcoeff!(α2,t2,BigInt.(1:N))
    backαcoeff!(α3,t3,BigInt.(1:N))
    # Mathematica α1
    @test α0[1] ≈ Float64(initialα(t0))
    @test α1[1] ≈ Float64(initialα(t1))
    @test α2[1] ≈ Float64(initialα(t2))
    @test α3[1] ≈ Float64(initialα(t3))
end

@testset "OPs for a=b=0, c=-1 - evaluation normalized" begin
    t = BigFloat("1.1")
    # Mathematica values
    @test evalQn(0,0.99,t) ≈ 0.5
    @test evalQn(1,0.5,t) ≈ 0.4277471315809677
    @test evalQn(6,0.12,t) ≈ -1.269069450850338
end

@testset "OPs for a=b=0, c=-1 - evaluation non-normalized" begin
    t = 1.1
    x = 0.1
    n = 5
    α = zeros(n+1)'
    α[1] = initialα(2*t-1)
    αcoefficients!(α,2*t-1,2:n)
    # compare versions with and without recomputing α with Mathematica results
    @test evalϕn(0,x,t) == 1
    @test evalϕn(1,x,t) ≈ 1.165935217151491
    @test evalϕn(2,x,t) ≈ 0.806910345733665
end

@testset "OPs for a=b=0, c=-1 - basic evaluation consistency" begin
    # basis
    t = 1.1
    P = Legendre()[affine(Inclusion(0..1), axes(Legendre(),1)), :]
    x = axes(P,1)
    # compute coefficients for basis
    N = 20
    α = zeros(N)'
    α[1] = initialα(2*t-1)
    αcoefficients!(α,2*t-1,2:N)
    # generate B operator
    B = neg1c_tolegendre(t)
    B = B[1:20,1:20]
    # some test functions
    f1(x) = x^2
    f2(x) = (t-x)^2
    f3(x) = exp(t-x)
    # test basic expansion and evaluation via Legendre()
    y = rand(1)[1]
    u1 = qr(B) \ (P[:,1:20] \ f1.(x))
    @test (P[:,1:20]*(B*u1))[y] ≈ f1(y)
    u2 = qr(B) \ (P[:,1:20] \ @.((t-x)^2))
    @test (P[:,1:20]*(B*u2))[y] ≈ f2(y)
    u3 = qr(B) \ (P[:,1:20] \ @.(exp(t-x)))
    @test (P[:,1:20]*(B*u3))[y] ≈ f3(y)
end