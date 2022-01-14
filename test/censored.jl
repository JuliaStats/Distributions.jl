# Testing censored distributions

module TestCensored

using Distributions, Test
using Distributions: Censored

@testset "censored" begin
    d0 = Normal(0, 1)
    @test_throws ErrorException censored(d0, 1, -1)

    d = censored(d0, -1, 1.0)
    @test d isa Censored
    @test d.lower === -1.0
    @test d.upper === 1.0

    d = censored(d0, missing, -1)
    @test d isa Censored
    @test ismissing(d.lower)
    @test d.upper == -1

    d = censored(d0, 1, missing)
    @test d isa Censored
    @test ismissing(d.upper)
    @test d.lower == 1

    d = censored(d0, missing, missing)
    @test d === d0
end

@testset "Censored" begin
    @testset "basic" begin
        d = Censored(Normal(0.0, 1.0), -1, 2)
        @test d isa Censored
        @test eltype(d) === Float64
        @test params(d) === (params(Normal(0.0, 1.0))..., -1, 2)
        @test partype(d) === Float64
        @test @inferred extrema(d) == (-1, 2)
        @test @inferred islowerbounded(d)
        @test @inferred isupperbounded(d)
        @test @inferred insupport(d, 0.1)
        @test insupport(d, -1)
        @test insupport(d, 2)
        @test !insupport(d, -1.1)
        @test !insupport(d, 2.1)
        @test sprint(show, "text/plain", d) == "Censored($(Normal(0.0, 1.0)), range=(-1, 2))"

        d = Censored(Cauchy(0, 1), missing, 2)
        @test d isa Censored
        @test eltype(d) === Base.promote_type(eltype(Cauchy(0, 1)), Int)
        @test params(d) == (params(Cauchy(0, 1))..., 2)
        @test partype(d) === Float64
        @test extrema(d) == (-Inf, 2.0)
        @test @inferred !islowerbounded(d)
        @test @inferred isupperbounded(d)
        @test @inferred insupport(d, 0.1)
        @test insupport(d, -3)
        @test insupport(d, 2)
        @test !insupport(d, 2.1)
        @test sprint(show, "text/plain", d) == "Censored($(Cauchy(0.0, 1.0)), range=(missing, 2))"

        d = Censored(Gamma(1, 2), 2, missing)
        @test d isa Censored
        @test eltype(d) === Base.promote_type(eltype(Gamma(1, 2)), Int)
        @test params(d) == (params(Gamma(1, 2))..., 2)
        @test partype(d) === Float64
        @test extrema(d) == (2.0, Inf)
        @test @inferred islowerbounded(d)
        @test @inferred !isupperbounded(d)
        @test @inferred insupport(d, 2.1)
        @test insupport(d, 2.0)
        @test !insupport(d, 1.9)
        @test sprint(show, "text/plain", d) == "Censored($(Gamma(1, 2)), range=(2, missing))"

        d = Censored(Binomial(10, 0.2), -1.5, 9.5)
        @test extrema(d) === (0.0, 9.5)
        @test @inferred islowerbounded(d)
        @test @inferred isupperbounded(d)
        @test @inferred !insupport(d, -1.5)
        @test insupport(d, 0)
        @test insupport(d, 9.5)
        @test !insupport(d, 10)

        @test censored(Censored(Normal(), 1, missing), missing, 2) == Censored(Normal(), 1, 2)
        @test censored(Censored(Normal(), missing, 1), -1, missing) == Censored(Normal(), -1, 1)
        @test censored(Censored(Normal(), 1, 2), 1.5, 2.5) == Censored(Normal(), 1.5, 2.0)
        @test censored(Censored(Normal(), 1, 3), 1.5, 2.5) == Censored(Normal(), 1.5, 2.5)
        @test censored(Censored(Normal(), 1, 2), 0.5, 2.5) == Censored(Normal(), 1.0, 2.0)
        @test censored(Censored(Normal(), 1, 2), 0.5, 1.5) == Censored(Normal(), 1.0, 1.5)

        @test censored(Censored(Normal(), missing, 1), missing, 1) == Censored(Normal(), missing, 1)
        @test censored(Censored(Normal(), missing, 1), missing, 2) == Censored(Normal(), missing, 1)
        @test censored(Censored(Normal(), missing, 1), missing, 1.5) == Censored(Normal(), missing, 1)
        @test censored(Censored(Normal(), missing, 1.5), missing, 1) == Censored(Normal(), missing, 1)

        @test censored(Censored(Normal(), 1, missing), 1, missing) == Censored(Normal(), 1, missing)
        @test censored(Censored(Normal(), 1, missing), 2, missing) == Censored(Normal(), 2, missing)
        @test censored(Censored(Normal(), 1, missing), 1.5, missing) == Censored(Normal(), 1.5, missing)
        @test censored(Censored(Normal(), 1.5, missing), 1, missing) == Censored(Normal(), 1.5, missing)
    end
end

end # module