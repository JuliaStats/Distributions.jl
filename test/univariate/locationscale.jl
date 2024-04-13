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
        @testset "$k" for (k,dtest) in d_dict
            @test minimum(dtest) == minimum(dref)
            @test maximum(dtest) == maximum(dref)
            @test extrema(dtest) == (minimum(dref), maximum(dref))
            @test extrema(support(dtest)) == extrema(dref)
            @test support(dtest) == support(dref)
        end
    end

    #### Promotions and conversions

    @testset "Promotions and conversions" begin
        @testset "$k" for (k,dtest) in d_dict
            if dtest isa Distributions.AffineDistribution
                @test typeof(dtest.μ) === typeof(dtest.σ)
                @test location(dtest) ≈ μ atol=1e-15
                @test    scale(dtest) ≈ σ atol=1e-15
            end
        end
    end

    #### Statistics

    @testset "Statistics" begin
        @testset "$k" for (k,dtest) in d_dict
            @test mean(dtest) ≈ mean(dref)
            @test median(dtest) ≈ median(dref)
            @test mode(dtest) ≈ mode(dref)
            @test modes(dtest) ≈ modes(dref)

            @test var(dtest) ≈ var(dref)
            @test std(dtest) ≈ std(dref)

            @test skewness(dtest) ≈ skewness(dref)
            @test kurtosis(dtest) ≈ kurtosis(dref)

            @test isplatykurtic(dtest) == isplatykurtic(dref)
            @test isleptokurtic(dtest) == isleptokurtic(dref)
            @test ismesokurtic(dtest) == ismesokurtic(dref)

            @test entropy(dtest) ≈ entropy(dref)
            @test mgf(dtest,-0.1) ≈ mgf(dref,-0.1)
        end
    end

    #### Evaluation & Sampling

    @testset "Evaluation & Sampling" begin
        @testset "$k" for (k,dtest) in d_dict
            xs = rand(dref, 5)
            x = first(xs)
            insupport(dtest, x) == insupport(dref, x)
            # might return `false` for discrete distributions
            insupport(dtest, -x) == insupport(dref, -x)

            @test pdf(dtest, x) ≈ pdf(dref, x)
            @test Base.Fix1(pdf, dtest).(xs) ≈ Base.Fix1(pdf, dref).(xs)
            @test logpdf(dtest, x) ≈ logpdf(dref, x)
            @test Base.Fix1(logpdf, dtest).(xs) ≈ Base.Fix1(logpdf, dref).(xs)
            @test loglikelihood(dtest, x) ≈ loglikelihood(dref, x)
            @test loglikelihood(dtest, xs) ≈ loglikelihood(dref, xs)

            @test cdf(dtest, x) ≈ cdf(dref, x)
            @test logcdf(dtest, x) ≈ logcdf(dref, x) atol=1e-14
            @test ccdf(dtest, x) ≈ ccdf(dref, x) atol=1e-14
            @test logccdf(dtest, x) ≈ logccdf(dref, x) atol=1e-14

            @test quantile(dtest, 0.1) ≈ quantile(dref, 0.1)
            @test quantile(dtest, 0.5) ≈ quantile(dref, 0.5)
            @test quantile(dtest, 0.9) ≈ quantile(dref, 0.9)

            @test cquantile(dtest, 0.1) ≈ cquantile(dref, 0.1)
            @test cquantile(dtest, 0.5) ≈ cquantile(dref, 0.5)
            @test cquantile(dtest, 0.9) ≈ cquantile(dref, 0.9)

            @test invlogcdf(dtest, log(0.2)) ≈ invlogcdf(dref, log(0.2))
            @test invlogcdf(dtest, log(0.5)) ≈ invlogcdf(dref, log(0.5))
            @test invlogcdf(dtest, log(0.8)) ≈ invlogcdf(dref, log(0.8))

            @test invlogccdf(dtest, log(0.2)) ≈ invlogccdf(dref, log(0.2))
            @test invlogccdf(dtest, log(0.5)) ≈ invlogccdf(dref, log(0.5))
            @test invlogccdf(dtest, log(0.8)) ≈ invlogccdf(dref, log(0.8))

            r = Array{float(eltype(dtest))}(undef, 100000)
            if ismissing(rng)
                rand!(dtest, r)
            else
                rand!(rng, dtest, r)
            end
            @test mean(r) ≈ mean(dref) atol=0.02
            @test std(r) ≈ std(dref) atol=0.01
            @test cf(dtest, -0.1) ≈ cf(dref,-0.1)

            if dref isa ContinuousDistribution
                @test gradlogpdf(dtest, 0.1) ≈ gradlogpdf(dref, 0.1)
            end
        end
    end
end

function test_location_scale_normal(
    rng::Union{AbstractRNG, Missing}, μ::Real, σ::Real, μD::Real, σD::Real,
)
    ρ = Normal(μD, σD)
    dref = Normal(μ + σ * μD,  abs(σ) * σD)
    @test dref === μ + σ * ρ
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

    @testset "Normal" begin
        for _rng in (missing, rng), sign in (1, -1)
            test_location_scale_normal(_rng, 0.3, sign * 0.2, 0.1, 0.2)
            test_location_scale_normal(_rng, -0.3, sign * 0.1, -0.1, 0.3)
            test_location_scale_normal(_rng, 1.3, sign * 0.4, -0.1, 0.5)
        end
        test_location_scale_normal(rng, ForwardDiff.Dual(0.3), 0.2, 0.1, 0.2)
    end
    @testset "DiscreteNonParametric" begin
        probs = normalize!(rand(10), 1)
        for _rng in (missing, rng), sign in (1, -1)
            test_location_scale_discretenonparametric(_rng, 1//3, sign * 1//2, 1:10, probs)
            test_location_scale_discretenonparametric(_rng, -1//4, sign * 1//3, (-10):(-1), probs)
            test_location_scale_discretenonparametric(_rng, 6//5, sign * 3//2, 15:24, probs)
        end
    end

    @test_logs Distributions.AffineDistribution(1.0, 1, Normal())

    @test_deprecated ls_norm = LocationScale(1.0, 1, Normal())
    @test ls_norm isa LocationScale{Float64, Continuous, Normal{Float64}}
    @test ls_norm isa Distributions.AffineDistribution{Float64, Continuous, Normal{Float64}}
end
