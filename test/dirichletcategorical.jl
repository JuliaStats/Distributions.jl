# Tests for DirichletCategorical distribution

using  Distributions
using Test, Random, LinearAlgebra


Random.seed!(34567)

rng = MersenneTwister(123)

@testset "Testing DirichletCategorical with $key" for (key, func) in
    Dict("rand(...)" => [rand, rand] ) #,
         #"rand(rng, ...)" => [dist -> rand(rng, dist), (dist, n) -> rand(rng, dist, n)])

# Constructors
d1 = DirichletCategorical([0.1, 0.2, 0.3], [1, 2, 3])
d2 = DirichletCategorical([4, 2, 3], [1, 1, 0])
d3 = DirichletCategorical(5)
d4 = DirichletCategorical([1.3, 4.3, 2.6, 3.9])

@testset "Constructors" for d in (d1, d2, d3, d4)
    @test typeof(d) == DirichletCategorical{Float64,Int64}
end

@testset "Constructor throws" begin
    @test_throws ArgumentError DirichletCategorical([-1, 0], [0, 0])
    @test_throws ArgumentError DirichletCategorical([0, 0], [-1, 0])
    @test_throws ArgumentError DirichletCategorical([0], [0, 0])
    @test_throws ArgumentError DirichletCategorical([0, 0], [1, 1])
    @test_throws ArgumentError DirichletCategorical([-1, 2], [0, 0])
    @test_throws ArgumentError DirichletCategorical([1.0, 1.0], [2, -1])
end

@testset "ncategories" for (d,n) in [(d1, 3), (d2, 3), (d3, 5), (d4, 4)]
    @test ncategories(d) == n
end

@testset "length" for (d,n) in [(d1, 3), (d2, 3), (d3, 5), (d4, 4)]
    @test length(d) == n
end

@testset "params" begin
    @test params(d1) == ([0.1, 0.2, 0.3], [1, 2, 3])
    @test params(d2) == ([4, 2, 3], [1, 1, 0])
    @test params(d3) == (ones(5), zeros(Int, 5))
    @test params(d4) == ([1.3, 4.3, 2.6, 3.9], zeros(Int, 4))
end

@testset "partype" for d in (d1, d2, d3, d4)
    @test partype(d) == Float64
end

@testset "rand" begin
    @testset "single" for (d,n) in [(d1, 1), (d2, 1), (d3, 1), (d4, 1)]
        @test rand(d) == n
    end

    @testset "multiple" for (d,n) in [
            (d1, [1, 2, 1, 1, 2, 2, 3, 3, 3, 1]),
            (d2, [1, 1, 1, 1, 1, 1, 3, 2, 3, 1]),
            (d3, [1, 1, 1, 1, 3, 2, 5, 3, 5, 1]),
            (d4, [1, 2, 2, 1, 2, 2, 4, 3, 4, 1])]
        @test rand(d, 10) == n
    end

    @testset "10k" for (d,n) in [
            (d1, [1613, 3334, 5053]),
            (d2, [4504, 2720, 2776]),
            (d3, [1932, 2024, 1978, 2055, 2011]),
            (d4, [1058, 3526, 2147, 3269])]
        @test begin
            r = rand(d, 10000)
            counts = map(x -> sum(r .== x), 1:maximum(d))
            counts == n
        end
    end

end

@testset "pdf" for (d,x,p) in [
        (d1, 1, 0.16666666666666669),
        (d2, 2, 0.2727272727272727),
        (d3, 3, 0.2),
        (d4, 4, 0.32231404958677684),
        (d4, -1, 0),
        (d4, 7, 0)]
    @test pdf(d, x) == p
end

@testset "logpdf" for (d,x,p) in [
        (d1, 1, -1.7917594692280547),
        (d2, 2, -1.2992829841302609),
        (d3, 3, -1.6094379124341003),
        (d4, 4, -1.1322288994670948),
        (d4, -1, -Inf),
        (d4, 9, -Inf)]
    @test logpdf(d, x) == p
end


@testset "cpdf" for (d,x,p) in [
        (d1, 1, 0.16666666666666669),
        (d2, 2, 0.7272727272727273),
        (d3, 3, 0.6000000000000001),
        (d4, 4, 1.0),
        (d4, -1, 0),
        (d3, 5, 1.0)]
    @test cdf(d, x) == p
end


@testset "quantile" for (d,x,p) in [(d1, 1, 0.1), (d2, 1, 0.4), (d3, 3, 0.6), (d4, 4, 0.8)]
     @test quantile(d, p) == x

end

@testset "quantile throws" begin
     @test_throws DomainError quantile(d1, -1.0)
     @test_throws DomainError quantile(d1, 2.0)
end

@testset "minimum" for d in (d1, d2, d3, d4)
    @test minimum(d) == 1
end

@testset "maximum" for (d,n) in [(d1, 3), (d2, 3), (d3, 5), (d4, 4)]
    @test maximum(d) == n
end

@testset "insupport" for (d,s) in [
        (d1, [0, 1, 1, 1, 0, 0, 0]),
        (d2, [0, 1, 1, 1, 0, 0, 0]),
        (d3, [0, 1, 1, 1, 1, 1, 0]),
        (d4, [0, 1, 1, 1, 1, 0, 0])]
    @test map(x -> insupport(d, x), collect(0:6)) == s
end

@testset "mode" for (d,n) in [(d1, 3), (d2, 1), (d4, 2)]
    @test mode(d) == n
end

@testset "mode throws" begin
    @test_throws ErrorException mode(d3)
end

@testset "update!" begin
    @testset "single" for (d,o,n) in [
            (d1, 1, [2, 2, 3]),
            (d2, 2, [1, 2, 0]),
            (d3, 3, [0, 0, 1, 0, 0]),
            (d4, 4, [0, 0, 0, 1])]
        @test begin
            d = deepcopy(d)
            update!(d, o)

            d.n == n
        end
    end

    @testset "multiple" for (d,o,n) in [
            (d1, [1, 1, 1, 3, 3, 3, 2, 1, 3, 3], [5, 3, 8]),
            (d2, [3, 2, 3, 2, 3, 2, 3, 3, 3, 3], [1, 4, 7]),
            (d3, [3, 4, 5, 4, 4, 4, 5, 5, 4, 4], [0, 0, 1, 6, 3]),
            (d4, [4, 4, 1, 4, 4, 4, 2, 1, 3, 2], [2, 2, 1, 5])]
        @test begin
            d = deepcopy(d)
            update!(d, o)

            d.n == n
        end
    end

    @testset "throws" begin
        @test_throws ArgumentError update!(d1, -1)
        @test_throws ArgumentError update!(d1, 5)
    end

    @testset "rand" begin
        d = DirichletCategorical([10, 0.0001, 0.0001, 0.0001])
        a = rand(d, 100)
        update!(d, ones(Int, 10000) .* 4)
        b = rand(d, 100)

        all(a .== 1) && all(b .== 4)
    end

end

end
