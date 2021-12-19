function test_location_scale(
    rng::Union{AbstractRNG, Missing},
    μ::Real, σ::Real, ρ::UnivariateDistribution, dref::UnivariateDistribution,
)
    d = Distributions.AffineDistribution(μ, σ, ρ)
    @test params(d) == (μ,σ,ρ)
    @test eltype(d) === eltype(dref)

    # Different ways to construct the AffineDistribution object
    if dref isa DiscreteDistribution
        # floating point division introduces numerical errors
        # Better: multiply with rational numbers
        d_dict = Dict(
            "original" => d,
            "sugar" => σ * ρ + μ,
        )
    else
        d_dict = Dict(
            "original" => d,
            "sugar" => σ * ρ + μ,
            "composed" =>  μ + ((2 * ρ) * σ - 2) / 2 + 1
        )
    end

    @test d == deepcopy(d)

    #### Support / Domain

    @testset "Support" begin
        function test_support(d, dref)
            @test minimum(d) == minimum(dref)
            @test maximum(d) == maximum(dref)
            @test extrema(d) == (minimum(d), maximum(d))
            @test extrema(support(d)) == extrema(d)
            if support(d.ρ) isa RealInterval
                @test support(d) isa RealInterval
            elseif hasfinitesupport(d.ρ)
                @test support(d) == d.μ .+ d.σ .* support(d.ρ)
            end
        end
        @testset "$k" for (k,dtest) in d_dict
            test_support(dtest, dref)
        end
    end

    #### Promotions and conversions

    @testset "Promotions and conversions" begin
        function test_promotions_and_conversions(d, dref)
            @test typeof(d.µ) === typeof(d.σ)
            @test location(d) ≈ μ atol=1e-15
            @test    scale(d) ≈ σ atol=1e-15
        end
        @testset "$k" for (k,dtest) in d_dict
            test_promotions_and_conversions(dtest, dref)
        end
    end

    #### Statistics

    @testset "Statistics" begin
        function test_statistics(d, dref)
            @test mean(d) ≈ mean(dref)
            @test median(d) ≈ median(dref)
            @test mode(d) ≈ mode(dref)
            @test modes(d) ≈ modes(dref)

            @test var(d) ≈ var(dref)
            @test std(d) ≈ std(dref)

            @test skewness(d) ≈ skewness(dref)
            @test kurtosis(d) ≈ kurtosis(dref)

            @test isplatykurtic(d) == isplatykurtic(dref)
            @test isleptokurtic(d) == isleptokurtic(dref)
            @test ismesokurtic(d) == ismesokurtic(dref)

            @test entropy(d) ≈ entropy(dref)
            @test mgf(d,-0.1) ≈ mgf(dref,-0.1)
        end
        @testset "$k" for (k,dtest) in d_dict
            test_statistics(dtest, dref)
        end
    end

    #### Evaluation & Sampling

    @testset "Evaluation & Sampling" begin
        function test_evaluation_and_sampling(rng, d, dref)
            xs = rand(dref, 5)
            x = first(xs)
            insupport(d, x) == insupport(dref, x)
            # might return `false` for discrete distributions
            insupport(d, -x) == insupport(dref, -x)

            @test pdf(d, x) ≈ pdf(dref, x)
            @test pdf.(d, xs) ≈ pdf.(dref, xs)
            @test logpdf(d, x) ≈ logpdf(dref, x)
            @test logpdf.(d, xs) ≈ logpdf.(dref, xs)
            @test loglikelihood(d, x) ≈ loglikelihood(dref, x)
            @test loglikelihood(d, xs) ≈ loglikelihood(dref, xs)

            @test cdf(d, x) ≈ cdf(dref, x)
            @test logcdf(d, x) ≈ logcdf(dref, x)
            @test ccdf(d, x) ≈ ccdf(dref, x) atol=1e-15
            @test logccdf(d, x) ≈ logccdf(dref, x) atol=1e-15

            @test quantile(d,0.1) ≈ quantile(dref,0.1)
            @test quantile(d,0.5) ≈ quantile(dref,0.5)
            @test quantile(d,0.9) ≈ quantile(dref,0.9)

            @test cquantile(d,0.1) ≈ cquantile(dref,0.1)
            @test cquantile(d,0.5) ≈ cquantile(dref,0.5)
            @test cquantile(d,0.9) ≈ cquantile(dref,0.9)

            @test invlogcdf(d,log(0.2)) ≈ invlogcdf(dref,log(0.2))
            @test invlogcdf(d,log(0.5)) ≈ invlogcdf(dref,log(0.5))
            @test invlogcdf(d,log(0.8)) ≈ invlogcdf(dref,log(0.8))

            @test invlogccdf(d,log(0.2)) ≈ invlogccdf(dref,log(0.2))
            @test invlogccdf(d,log(0.5)) ≈ invlogccdf(dref,log(0.5))
            @test invlogccdf(d,log(0.8)) ≈ invlogccdf(dref,log(0.8))

            r = Array{float(eltype(d))}(undef, 100000)
            if ismissing(rng)
                rand!(d,r)
            else
                rand!(rng,d,r)
            end
            @test mean(r) ≈ mean(dref) atol=0.02
            @test std(r) ≈ std(dref) atol=0.01
            @test cf(d, -0.1) ≈ cf(dref,-0.1)

            if dref isa ContinuousDistribution
                @test gradlogpdf(d, 0.1) ≈ gradlogpdf(dref, 0.1)
            end
        end
        @testset "$k" for (k,dtest) in d_dict
            test_evaluation_and_sampling(rng, dtest, dref)
        end
    end
end

function test_location_scale_normal(
    rng::Union{AbstractRNG, Missing}, μ::Real, σ::Real, μD::Real, σD::Real,
)
    ρ = Normal(μD, σD)
    dref = Normal(μ + σ * μD, σ * σD)
    return test_location_scale(rng, μ, σ, ρ, dref)
end

function test_location_scale_discretenonparametric(
    rng::Union{AbstractRNG, Missing}, μ::Real, σ::Real, support, probs,
)
    ρ = DiscreteNonParametric(support, probs)
    dref = DiscreteNonParametric(μ .+ σ .* support, probs)
    return test_location_scale(rng, μ, σ, ρ, dref)
end

@testset "AffineDistribution" begin
    rng = MersenneTwister(123)

    for _rng in (missing, rng)
        test_location_scale_normal(_rng, 0.3, 0.2, 0.1, 0.2)
        test_location_scale_normal(_rng, -0.3, 0.1, -0.1, 0.3)
        test_location_scale_normal(_rng, 1.3, 0.4, -0.1, 0.5)
    end
    test_location_scale_normal(rng, ForwardDiff.Dual(0.3), 0.2, 0.1, 0.2)

    probs = normalize!(rand(10), 1)
    for _rng in (missing, rng)
        test_location_scale_discretenonparametric(_rng, 1//3, 1//2, 1:10, probs)
        test_location_scale_discretenonparametric(_rng, -1//4, 1//3, (-10):(-1), probs)
        test_location_scale_discretenonparametric(_rng, 6//5, 3//2, 15:24, probs)
    end

    @test_logs Distributions.AffineDistribution(1.0, 1, Normal())

    @test_deprecated ls_norm = LocationScale(1.0, 1, Normal())
    @test ls_norm isa LocationScale{Float64, Continuous, Normal{Float64}}
    @test ls_norm isa Distributions.AffineDistribution{Float64, Continuous, Normal{Float64}}
end
