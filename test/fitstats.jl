using StatsBase, Distributions, Test

@testset "fitstats" begin
    
@testset "moments" begin
    @testset "two moments" begin
        m = @inferred Moments(1,0.5)
        @test n_moments(typeof(m)) == 2
        @test n_moments(m) == 2
        #@test n_moments(typeof(M)) == 2
        @test (@inferred mean(m)) == m[1] == 1
        @test (@inferred var(m)) == m[2] == 0.5
        @test (@inferred std(m)) == sqrt(0.5)
        @test_throws Exception skewness(m)
        @test_throws Exception kurtosis(m)
        @test (@inferred convert(AbstractArray, m)) == [1.0, 0.5]
    end;
    @testset "zero moments" begin
        m = @inferred  Moments()
        @test (@inferred n_moments(m)) == 0
        @test_throws Exception mean(m)
        @test_throws Exception m[1]
    end;
end;

@testset "moments of Distribution" begin
    function testMoments(d)
        m = @inferred moments(d, Val(4))
        @test [m[i] for i in 1:n_moments(m)] == [mean(d), var(d), skewness(d), kurtosis(d)]
        m = @inferred moments(d, Val(3))
        @test [m[i] for i in 1:n_moments(m)] == [mean(d), var(d), skewness(d)]
        m = @inferred moments(d, Val(2))
        @test [m[i] for i in 1:n_moments(m)] == [mean(d), var(d)]
        m = @inferred moments(d, Val(1))
        @test [m[i] for i in 1:n_moments(m)] == [mean(d)]
        m = @inferred moments(d, Val(0))
        @test [m[i] for i in 1:(@inferred n_moments(m))] == []
        @test_throws Exception moments(d, Val(-1))
        @test_throws Exception moments(d, Val(:bla))
    end
    @testset "Normal" begin
        d = @inferred Normal(2,5)
        testMoments(d)
    end;
    @testset "Lognormal" begin 
        testMoments((@inferred LogNormal(2,5)))
    end;
    @testset "Logitnormal" begin
        # mean not defined fro LogitNormal
        @test_throws MethodError testMoments((@inferred LogitNormal(2,5)))
    end;
end;

@testset "QuantilePoint" begin
    qp1 = @inferred QuantilePoint(2,0.25);
    qp2 = @inferred QuantilePoint(7.5,0.95);
    @testset "isless" begin
        #println("isless")
        # p and q both increasing/decreasing
        @test (@inferred(QuantilePoint(2,0.4) < QuantilePoint(3,0.5))) == true
        @test (@inferred(QuantilePoint(1,0.6) < QuantilePoint(1,0.5))) == false
        # same q may have different p as same samples can be repeated
        @test (@inferred(QuantilePoint(2,0.4) < QuantilePoint(2,0.5))) == false
        @test (@inferred(QuantilePoint(2,0.6) < QuantilePoint(2,0.5))) == false
        # q decreasing but p increasing or vise-versa
        @test_throws ErrorException QuantilePoint(1,0.6) < QuantilePoint(2,0.5) 
        @test_throws ErrorException QuantilePoint(2,0.4) < QuantilePoint(1,0.5)
        # for same p equal, also q must be equal
        @test_throws ErrorException  QuantilePoint(1,0.5) < QuantilePoint(2,0.5) 
        @test_throws ErrorException  QuantilePoint(2,0.5) < QuantilePoint(1,0.5)
    end;
    @testset "modifying Quantilepoint" begin
        qp3 = QuantilePoint(qp1, q = 3);
        @test qp3.q == 3
        @test qp3.p == qp1.p == 0.25
    end;
    @testset "macros" begin
        @test @qp(0.4,0.7) == QuantilePoint(0.4,0.7)
        @test @qp_ll(0.7) == QuantilePoint(0.7,0.025)
        @test @qp_l(0.7) == QuantilePoint(0.7,0.05)
        @test @qp_m(0.7) == QuantilePoint(0.7,0.5)
        @test @qp_u(0.7) == QuantilePoint(0.7,0.95)
        @test @qp_uu(0.7) == QuantilePoint(0.7,0.975)
        @test @qs_cf90(0.2,0.7) == Set([@qp_l(0.2),@qp_u(0.7)])
        @test @qs_cf95(0.2,0.7) == Set([@qp_ll(0.2),@qp_uu(0.7)])
    end;
end;

@testset "global fit functions" begin
    @test_throws ErrorException fit(Distribution, Moments())
    qpl = @qp_m(3)
    qpu = @qp_u(5)
    @test_throws ErrorException fit(Distribution, qpl, qpu)
    @testset "fit to stats and quantilepoint of unknown distribution" begin
        @test_throws ErrorException fit(Distribution, 3.0, qpu)
        @test_throws ErrorException fit(Distribution, 3.0, qpu, Val(:mean))
        @test_throws ErrorException fit(Distribution, 3.0, qpu, Val(:mode))
        @test_throws ErrorException fit(Distribution, 3.0, qpu, Val(:median))
        @test_throws ErrorException fit(Distribution, 3.0, qpu, Val(:bla))
        @test_throws ErrorException fit(Distribution, Moments(1.0,3.0))
    end;
end;

end; # fitstats
