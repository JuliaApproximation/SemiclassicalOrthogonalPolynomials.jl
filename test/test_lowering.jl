using SemiclassicalOrthogonalPolynomials, Test
using ClassicalOrthogonalPolynomials, ContinuumArrays, BandedMatrices, QuasiArrays, Test, LazyArrays, LinearAlgebra, InfiniteArrays
import LazyArrays: AbstractCachedVector
import SemiclassicalOrthogonalPolynomials: initialα, αdirect, αdirect!, backαcoeff!, αcoefficients!, evalϕn, neg1c_tolegendre, evalQn, initialαc_gen, αcmillerbackwards, αcfillerbackwards!, lowercjacobimatrix, αcforward!, CLoweringCoefficients, BLoweringCoefficients, ALoweringCoefficients, lowerajacobimatrix, lowerbjacobimatrix, initialαa_gen, initialαb_gen, αaforward!, αbforward!, αbfillerbackwards!, αafillerbackwards!

@testset "Jacobi operator for c-1 from c" begin
    @testset "α[1] consistency" begin
        t=1.1; a=0; b=0; c=0;
        scale = 20;
        P = SemiclassicalJacobi(t,a,b,c)
        v = zeros(10)
        v[1] = initialαc_gen(t,a,b,c)
        αcfillerbackwards!(v,10,10,P,1:10)
        @test initialαc_gen(t,a,b,c) ≈ initialαc_gen(P) ≈ αcmillerbackwards(20, scale, t, a, b, c)[1] ≈ αcmillerbackwards(20, scale, P)[1] ≈ v[1]
        
        t=1.001; a=0; b=0; c=1;
        P = SemiclassicalJacobi(t,a,b,c)
        v[1] = initialαc_gen(t,a,b,c)
        αcfillerbackwards!(v,10,10,P,1:10)
        @test initialαc_gen(t,a,b,c) ≈ initialαc_gen(P) ≈ αcmillerbackwards(20, scale, t, a, b, c)[1] ≈ αcmillerbackwards(20, scale, P)[1] ≈ v[1]
        
        t=1.71; a=3; b=2; c=4;
        P = SemiclassicalJacobi(t,a,b,c)
        v[1] = initialαc_gen(t,a,b,c)
        αcfillerbackwards!(v,10,10,P,1:10)
        @test initialαc_gen(t,a,b,c) ≈ initialαc_gen(P) ≈ αcmillerbackwards(20, scale, t, a, b, c)[1] ≈ αcmillerbackwards(20, scale, P)[1] ≈ v[1]
    end

    @testset "forward recurrence consistency" begin
        # forward recurrence is unstable for high orders
        # but can be used for value comparison at low orders
        t=1.001; a=0; b=0; c=1;
        P = SemiclassicalJacobi(t,a,b,c)
        v = zeros(10)
        w = zeros(10)
        v[1] = initialαc_gen(t,a,b,c)
        αcfillerbackwards!(v, 200, 10, P, 1:10)
        w[1] = initialαc_gen(t,a,b,c)
        αcforward!(w, t, a, b, c, 1:10)
        @test v ≈ w
    end
    
    @testset "cached α" begin
        t = 1.1; a = 2; b = 1; c = 3;
        P = SemiclassicalJacobi(t,a,b,c)
        α = CLoweringCoefficients(P)
        @test α isa AbstractCachedVector
        @test size(α) == (ℵ₀,)
    end

    @testset "compare lowered Jacobi operators" begin
        t = 1.1; a = 2; b = 1; c = 3;
        P = SemiclassicalJacobi(t,a,b,c)
        @test jacobimatrix(SemiclassicalJacobi(t,a,b,c-1))[1:50,1:50] ≈ lowercjacobimatrix(P)[1:50,1:50]
        t = 1.001; a = 0; b = 0; c = 0;
        P = SemiclassicalJacobi(t,a,b,c)
        @test jacobimatrix(SemiclassicalJacobi(t,a,b,c-1))[1:50,1:50] ≈ lowercjacobimatrix(P)[1:50,1:50]
        t = 1.8; a = 4; b = 0; c = 20;
        P = SemiclassicalJacobi(t,a,b,c)
        @test jacobimatrix(SemiclassicalJacobi(t,a,b,c-1))[1:50,1:50] ≈ lowercjacobimatrix(P)[1:50,1:50]
    end
end

@testset "Special case: SemiclassicalJacobi(t,0,0,-1) " begin
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

    @testset "basic forward and back recurrence" begin
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

    @testset "evaluation normalized" begin
        t = BigFloat("1.1")
        # Mathematica values
        @test evalQn(0,0.99,t) ≈ 1
        @test evalQn(1,0.5,t) ≈ -0.6623723753894052
        @test evalQn(6,0.12,t) ≈ -1.965171674178137
    end

    @testset "evaluation non-normalized" begin
        t = 1.1
        x = 0.1
        n = 5
        α = zeros(n+1)'
        α[1] = initialα(2*t-1)
        αcoefficients!(α,2*t-1,2:n)
        # compare versions with and without recomputing α with Mathematica results
        @test evalϕn(0,x,t) == 1
        @test evalϕn(1,x,t) ≈ -1.165935217151491
        @test evalϕn(2,x,t) ≈ 0.806910345733665
    end

    @testset "Expansion" begin
        # basis
        t = 1.00001
        Q = SemiclassicalJacobi(t,0,0,-1)
        x = axes(Q,1)
        # test functions
        f1(x) = x^2
        f2(x) = (t-x)^2
        f3(x) = exp(t-x)
        f4(x) = sinh(t*x)
        # test expansion
        y = rand(1)[1]
        @test (Q[:,1:30]*(Q[:,1:30]\f1.(x)))[y] ≈ f1(y)
        @test (Q[:,1:30]*(Q[:,1:30]\f2.(x)))[y] ≈ f2(y)
        @test (Q[:,1:30]*(Q[:,1:30]\f3.(x)))[y] ≈ f3(y)
        @test (Q[:,1:30]*(Q[:,1:30]\f4.(x)))[y] ≈ f4(y)
    end

    @testset "Expansion (adaptive)" begin
        # basis
        t = 1.23
        Q = SemiclassicalJacobi(t,0,0,-1)
        x = axes(Q,1)
        # test functions
        f1(x) = x^2
        f2(x) = (t-x)^2
        f3(x) = exp(t-x)
        f4(x) = sinh(t*x)
        # test expansion
        y = rand(1)[1]
        @test (Q*(Q\f1.(x)))[y] ≈ f1(y)
        @test (Q*(Q\f2.(x)))[y] ≈ f2(y)
        @test (Q*(Q\f3.(x)))[y] ≈ f3(y)
        @test (Q*(Q\f4.(x)))[y] ≈ f4(y)
    end

    @testset "Multiplication by x" begin
        # basis
        t = 1.000001
        Q = SemiclassicalJacobi(t,0,0,-1)
        x = axes(Q,1)
        X = jacobimatrix(Q)
        # test functions
        f1(x) = x^2
        f2(x) = (t-x)^2
        f3(x) = exp(t-x)
        f4(x) = sinh(t*x)
        # test expansion
        y = rand(1)[1]
        @test (Q*(X*(Q\f1.(x))))[y] ≈ y*f1(y)
        @test (Q*(X*X*(Q\f2.(x))))[y] ≈ y^2*f2(y)
        @test (Q*(X*(Q\f3.(x))))[y] ≈ y*f3(y)
        @test (Q*(X*(Q\f4.(x))))[y] ≈ y*f4(y)
    end
end

@testset "Lowering a and b" begin
    @testset "Jacobi special case" begin
        α = zeros(20)
        P = SemiclassicalJacobi(1.1,1,1,0)
        α[1] = initialαb_gen(P)
        αbforward!(α,P,1:20)
        # Mathematica
        @test α[1] ≈ -0.7453559924999
        @test α[2] ≈ -0.8366600265340
        @test α[13] ≈ -0.9648130376041
        @test α[20] ≈ -0.976440887660561
    end

    @testset "Jacobi operator consistency - lowering a" begin
        @test lowerajacobimatrix(SemiclassicalJacobi(1.1,2,3,1))[1:50,1:50] ≈ jacobimatrix(SemiclassicalJacobi(1.1,1,3,1))[1:50,1:50]
        @test lowerajacobimatrix(SemiclassicalJacobi(1.4,5,1,1))[1:100,1:100] ≈ jacobimatrix(SemiclassicalJacobi(1.4,4,1,1))[1:100,1:100]
        @test lowerajacobimatrix(SemiclassicalJacobi(1.01,10,10,5))[1:100,1:100] ≈ jacobimatrix(SemiclassicalJacobi(1.01,9,10,5))[1:100,1:100]
    end

    @testset "Jacobi operator consistency - lowering b" begin
        @test lowerbjacobimatrix(SemiclassicalJacobi(1.1,2,3,1))[1:50,1:50] ≈ jacobimatrix(SemiclassicalJacobi(1.1,2,2,1))[1:50,1:50]
        @test lowerbjacobimatrix(SemiclassicalJacobi(1.4,5,2,1))[1:100,1:100] ≈ jacobimatrix(SemiclassicalJacobi(1.4,5,1,1))[1:100,1:100]
        @test lowerbjacobimatrix(SemiclassicalJacobi(1.01,10,10,5))[1:50,1:50] ≈ jacobimatrix(SemiclassicalJacobi(1.01,10,9,5))[1:50,1:50]
    end    
end





