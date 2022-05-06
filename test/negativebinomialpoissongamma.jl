using Distributions
using Test

@testset "Check the corner case p==1" begin
    d = convert(NegativeBinomialPoissonGamma{Float64}, NegativeBinomial(0.5, 1.0))
    @test logpdf(d, 0) === 0.0
    @test logpdf(d, 1) === -Inf
end

@testset "Interconversion checks" begin
    d = NegativeBinomialPoissonGamma()
    nb = NegativeBinomial()
    nbl = NegativeBinomialLocation()
    nbll = NegativeBinomialLogLocation()
    nbpg = NegativeBinomialPoissonGamma()
    for (d₂, T) ∈ zip((nb, nbl, nbll, nbpg), (NegativeBinomial, NegativeBinomialLocation, NegativeBinomialLogLocation, NegativeBinomialPoissonGamma))
        # Convert to other distribution from NegativeBinomialPoissonGamma
        @test d₂ == convert(T{Float64}, d)
        # Convert from other distribution to NegativeBinomialPoissonGamma
        @test d == convert(NegativeBinomialPoissonGamma{Float64}, d₂)
    end
end

@testset "logpdf checks" begin
    d = NegativeBinomialPoissonGamma()
    nb = NegativeBinomial()
    nbl = NegativeBinomialLocation()
    nbll = NegativeBinomialLogLocation()
    nbpg = NegativeBinomialPoissonGamma()
    for d₂ ∈ (nb, nbl, nbll, nbpg), n ∈ (2, 10, 10^2, 10^5, 10^8)
        @test logpdf(d, n) ≈ logpdf(d₂, n)
    end
end

# # This is equivalent to the expanded code below. The explicit testsets may be
# # more informative if/when tests fail. In any case, it is nice to cleanly present the concept.
for (params, xs) ∈ zip(((), (6,), (1, 0.5), (5, 0.6), (0.5, 0.5)),
                       ((0, 1, 2, 3, 4, 5, 6), (0, 2, 3, 4, 5, 6, 7, 9, 11, 16), (0, 1, 2, 3, 4, 5, 6), (0, 1, 2, 3, 4, 5, 6, 10), (0, 1, 2, 3, 4, 5)))
    @testset "Checks against NegativeBinomial($(join(params, ',')))" begin
        nb = NegativeBinomial()
        d = convert(NegativeBinomialPoissonGamma{Float64}, nb)
        @test minimum(d) == 0
        @test maximum(d) == Inf
        @test insupport(d, -0)
        @test !insupport(d, 1.5)
        for f ∈ (succprob, failprob, mean, var, std, skewness, kurtosis, mode)
            @test f(nb) ≈ f(d)
        end
        for f ∈ (pdf, logpdf, cdf, logcdf, ccdf, logccdf)
            for x ∈ xs
                @test f(nb, x) ≈ f(d, x)
            end
        end
        for q ∈ (0.1, 0.25, 0.5, 0.75, 0.9)
            @test quantile(nb, q) == quantile(d, q)
            @test cquantile(nb, q) == cquantile(d, q)
        end
        for f ∈ (invlogcdf, invlogccdf)
            @test f(nb, -0.1) ≈ f(d, -0.1)
        end
        t = -log(succprob(nb)) / 2
        @test mgf(nb, t) ≈ mgf(d, t)
        @test cf(nb, t) ≈ cf(d, t)
    end
end

# @testset "Checks against NegativeBinomial()" begin
#     nb = NegativeBinomial()
#     d = convert(NegativeBinomialPoissonGamma{Float64}, nb)
#     @test minimum(d) == 0
#     @test maximum(d) == Inf
#     @test insupport(d, -0)
#     @test !insupport(d, 1.5)
#     for f ∈ (succprob, failprob, mean, var, std, skewness, kurtosis, mode)
#         @test f(nb) ≈ f(d)
#     end
#     for f ∈ (pdf, logpdf, cdf, logcdf, ccdf, logccdf)
#         for x ∈ (0, 1, 2, 3, 4, 5, 6)
#             @test f(nb, x) ≈ f(d, x)
#         end
#     end
#     for q ∈ (0.1, 0.25, 0.5, 0.75, 0.9)
#         @test quantile(nb, q) == quantile(d, q)
#         @test cquantile(nb, q) == cquantile(d, q)
#     end
#     for f ∈ (invlogcdf, invlogccdf)
#         @test f(nb, -0.1) ≈ f(d, -0.1)
#     end
#     t = -log(succprob(nb)) / 2
#     @test mgf(nb, t) ≈ mgf(d, t)
#     @test cf(nb, t) ≈ cf(d, t)
# end

# @testset "Checks against NegativeBinomial(6)" begin
#     nb = NegativeBinomial(6)
#     d = convert(NegativeBinomialPoissonGamma{Float64}, nb)
#     @test minimum(d) == 0
#     @test maximum(d) == Inf
#     @test insupport(d, -0)
#     @test !insupport(d, 1.5)
#     for f ∈ (succprob, failprob, mean, var, std, skewness, kurtosis, mode)
#         @test f(nb) ≈ f(d)
#     end
#     for f ∈ (pdf, logpdf, cdf, logcdf, ccdf, logccdf)
#         for x ∈ (0, 2, 3, 4, 5, 6, 7, 9, 11, 16)
#             @test f(nb, x) ≈ f(d, x)
#         end
#     end
#     for q ∈ (0.1, 0.25, 0.5, 0.75, 0.9)
#         @test quantile(nb, q) == quantile(d, q)
#         @test cquantile(nb, q) == cquantile(d, q)
#     end
#     for f ∈ (invlogcdf, invlogccdf)
#         @test f(nb, -0.1) ≈ f(d, -0.1)
#     end
#     t = -log(succprob(nb)) / 2
#     @test mgf(nb, t) ≈ mgf(d, t)
#     @test cf(nb, t) ≈ cf(d, t)
# end

# @testset "Checks against NegativeBinomial(1, 0.5)" begin
#     nb = NegativeBinomial(1, 0.5)
#     d = convert(NegativeBinomialPoissonGamma{Float64}, nb)
#     @test minimum(d) == 0
#     @test maximum(d) == Inf
#     @test insupport(d, -0)
#     @test !insupport(d, 1.5)
#     for f ∈ (succprob, failprob, mean, var, std, skewness, kurtosis, mode)
#         @test f(nb) ≈ f(d)
#     end
#     for f ∈ (pdf, logpdf, cdf, logcdf, ccdf, logccdf)
#         for x ∈ (0, 1, 2, 3, 4, 5, 6)
#             @test f(nb, x) ≈ f(d, x)
#         end
#     end
#     for q ∈ (0.1, 0.25, 0.5, 0.75, 0.9)
#         @test quantile(nb, q) == quantile(d, q)
#         @test cquantile(nb, q) == cquantile(d, q)
#     end
#     for f ∈ (invlogcdf, invlogccdf)
#         @test f(nb, -0.1) ≈ f(d, -0.1)
#     end
#     t = -log(succprob(nb)) / 2
#     @test mgf(nb, t) ≈ mgf(d, t)
#     @test cf(nb, t) ≈ cf(d, t)
# end

# @testset "Checks against NegativeBinomial(5, 0.6)" begin
#     nb = NegativeBinomial(5, 0.6)
#     d = convert(NegativeBinomialPoissonGamma{Float64}, nb)
#     @test minimum(d) == 0
#     @test maximum(d) == Inf
#     @test insupport(d, -0)
#     @test !insupport(d, 1.5)
#     for f ∈ (succprob, failprob, mean, var, std, skewness, kurtosis, mode)
#         @test f(nb) ≈ f(d)
#     end
#     for f ∈ (pdf, logpdf, cdf, logcdf, ccdf, logccdf)
#         for x ∈ (0, 1, 2, 3, 4, 5, 6, 10)
#             @test f(nb, x) ≈ f(d, x)
#         end
#     end
#     for q ∈ (0.1, 0.25, 0.5, 0.75, 0.9)
#         @test quantile(nb, q) == quantile(d, q)
#         @test cquantile(nb, q) == cquantile(d, q)
#     end
#     for f ∈ (invlogcdf, invlogccdf)
#         @test f(nb, -0.1) ≈ f(d, -0.1)
#     end
#     t = -log(succprob(nb)) / 2
#     @test mgf(nb, t) ≈ mgf(d, t)
#     @test cf(nb, t) ≈ cf(d, t)
# end

# @testset "Checks against NegativeBinomial(0.5, 0.5)" begin
#     nb = NegativeBinomial(0.5, 0.5)
#     d = convert(NegativeBinomialPoissonGamma{Float64}, nb)
#     @test minimum(d) == 0
#     @test maximum(d) == Inf
#     @test insupport(d, -0)
#     @test !insupport(d, 1.5)
#     for f ∈ (succprob, failprob, mean, var, std, skewness, kurtosis, mode)
#         @test f(nb) ≈ f(d)
#     end
#     for f ∈ (pdf, logpdf, cdf, logcdf, ccdf, logccdf)
#         for x ∈ (0, 1, 2, 3, 4, 5)
#             @test f(nb, x) ≈ f(d, x)
#         end
#     end
#     for q ∈ (0.1, 0.25, 0.5, 0.75, 0.9)
#         @test quantile(nb, q) == quantile(d, q)
#         @test cquantile(nb, q) == cquantile(d, q)
#     end
#     for f ∈ (invlogcdf, invlogccdf)
#         @test f(nb, -0.1) ≈ f(d, -0.1)
#     end
#     t = -log(succprob(nb)) / 2
#     @test mgf(nb, t) ≈ mgf(d, t)
#     @test cf(nb, t) ≈ cf(d, t)
# end
