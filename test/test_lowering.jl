using SemiclassicalOrthogonalPolynomials
using ClassicalOrthogonalPolynomials, ContinuumArrays, BandedMatrices, QuasiArrays, Test, LazyArrays, LinearAlgebra, InfiniteArrays
import LazyArrays: AbstractCachedVector
import SemiclassicalOrthogonalPolynomials: initialα, αdirect, αdirect!, backαcoeff!, αcoefficients!, evalϕn, neg1c_tolegendre, evalQn, getindex, initialα_gen, LoweredJacobiMatrix, αgenfillerbackwards!, MemoryLayout

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
        t = 1.00001
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
        @test (Q*(X*(Q\x.^2)))[y] ≈ y*f1(y)
        @test (Q*(X*X*(Q\(t.-x).^2)))[y] ≈ y^2*f2(y)
        @test (Q*(X*(Q\exp.(t.-x))))[y] ≈ y*f3(y)
        @test (Q*(X*(Q\sinh.(t.*x))))[y] ≈ y*f4(y)
    end
end

@testset "Compare lowering of a, b and c with raised OPs" begin
    @testset "lowering a" begin
        @test LoweredJacobiMatrix(SemiclassicalJacobi(1.0001,10,10,5,SemiclassicalJacobi(1.0001,0,0,0)),:a)[1:500,1:500] ≈ jacobimatrix(SemiclassicalJacobi(1.0001,9,10,5,SemiclassicalJacobi(1.0001,0,0,0)))[1:500,1:500]
    end
    @testset "lowering b" begin
        @test LoweredJacobiMatrix(SemiclassicalJacobi(3.1,6,6,6,SemiclassicalJacobi(3.1,0,0,0)),:b)[1:500,1:500]≈ jacobimatrix(SemiclassicalJacobi(3.1,6,5,6,SemiclassicalJacobi(3.1,0,0,0)))[1:500,1:500]
        @test LoweredJacobiMatrix(SemiclassicalJacobi(1.11,5,7,4,SemiclassicalJacobi(1.11,0,0,0)),:b)[1200,1200]≈ jacobimatrix(SemiclassicalJacobi(1.11,5,6,4,SemiclassicalJacobi(1.11,0,0,0)))[1200,1200]
    end
    @testset "lowering c" begin
        @test LoweredJacobiMatrix(SemiclassicalJacobi(1.1,2,3,4,SemiclassicalJacobi(1.1,0,0,0)),:c)[1:500,1:500]≈ jacobimatrix(SemiclassicalJacobi(1.1,2,3,3,SemiclassicalJacobi(1.1,0,0,0)))[1:500,1:500]
        @test LoweredJacobiMatrix(SemiclassicalJacobi(1.00001,12,4,7,SemiclassicalJacobi(1.00001,0,0,0)),:c)[1:500,1:500] ≈ jacobimatrix(SemiclassicalJacobi(1.00001,12,4,6,SemiclassicalJacobi(1.00001,0,0,0)))[1:500,1:500]
    end
end