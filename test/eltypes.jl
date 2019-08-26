using Distributions
using Test, Random, ForwardDiff
using ForwardDiff: Dual
using Random
using Random: GLOBAL_RNG
using Unitful

function test_types(D::Type{<:UnivariateDistribution{<:ContinuousSupport}},
                    targs)
    dist = D(targs...)
    @test eltype(dist) ≡ eltype(targs) ≡ typeof(rand(dist)) ≡
        typeof(rand(GLOBAL_RNG, dist)) ≡ eltype(rand(dist, 3))
    @test eltype(dist) ≡ typeof(mean(dist)) ≡ typeof(median(dist)) ≡
        typeof(mode(dist)) ≡ typeof(std(dist))
    @test eltype(targs) ≡
        typeof(quantile(dist, 0.5)) ≡ typeof(cquantile(dist, 0.5)) ≡
        typeof(invlogcdf(dist, -2.0)) ≡ typeof(invlogccdf(dist, -2.0))
    @test typeof(var(dist)) ≡ typeof(mean(dist) * mean(dist))
    @test typeof(skewness(dist)) ≡ typeof(kurtosis(dist)) ≡
        typeof(one(eltype(dist)))
end

function test_types(D::Type{<:UnivariateDistribution{<:CountableSupport}},
                    targs)
    dist = D(targs...)
    @test eltype(dist) ≡ eltype(targs) ≡ typeof(rand(dist)) ≡
        typeof(rand(GLOBAL_RNG, dist)) ≡ eltype(rand(dist, 3))
    @test eltype(dist) ≡ typeof(mode(dist))
    @test eltype(targs) ≡
        typeof(quantile(dist, 0.5)) ≡ typeof(cquantile(dist, 0.5)) ≡
        typeof(invlogcdf(dist, -2.0)) ≡ typeof(invlogccdf(dist, -2.0))
    @test typeof(var(dist)) ≡ typeof(mean(dist) * mean(dist))
    @test eltype(support(dist)) ≡ typeof(one(eltype(dist)))
end

@testset "Support for continuous distributions supporting just Float64" begin
    @testset "Testing $Dist" for (Dist, args) in Dict(
        # Chernoff broken Chernoff => (),
        Cosine => (0.0, 1.0)
    )
        test_types(Dist, Float64.(args))
    end
end

M = typeof(1.0u"m");
@testset "Support for continuous distributions supporting any eltype" begin
    @testset "Testing $Dist" for (Dist, args) in Dict(
        Normal => (0.0, 1.0)
    )
        @testset "Testing $T for $Dist" for T in [Float64, Float16, Dual, M]
            test_types(Dist, T.(args))
        end
    end
end

@testset "Support for discrete distributions supporting only Int64s" begin
    @testset "Testing $Dist" for (Dist, args) in Dict(
        Binomial => (10,)
    )
        @testset "Testing $T for $Dist" for T in [Int64]
            test_types(Dist, T.(args))
        end
    end
end
