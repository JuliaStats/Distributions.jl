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



@testset "Testing support and eltype" begin
    for dist in [Binomial(), Binomial(Int16(2)), Binomial(UInt8(10))]
        @test eltype(dist) == typeof(rand(dist)) == eltype(support(dist))
    end
end
