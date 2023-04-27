using SemiclassicalOrthogonalPolynomials
using ClassicalOrthogonalPolynomials, ContinuumArrays, BandedMatrices, QuasiArrays, Test, LazyArrays, LinearAlgebra, InfiniteArrays
import LazyArrays: AbstractCachedVector
import SemiclassicalOrthogonalPolynomials: evalϕn, neg1c_tolegendre, evalQn, getindex, initialα_gen

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
        # compare versions with and without recomputing α with Mathematica results
        @test evalϕn(0,x,t) == 1
        @test evalϕn(1,x,t) ≈ -1.165935217151491
        @test evalϕn(2,x,t) ≈ 0.806910345733665
    end

    @testset "Expansion" begin
        # basis
        t = 1.0001
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
        y = 0.4781
        @test (Q*(X*(Q\x.^2)))[y] ≈ y*f1(y)
        @test (Q*(X*X*(Q\(t.-x).^2)))[y] ≈ y^2*f2(y)
        @test (Q*(X*(Q\exp.(t.-x))))[y] ≈ y*f3(y)
        @test (Q*(X*(Q\sinh.(t.*x))))[y] ≈ y*f4(y)
    end
end