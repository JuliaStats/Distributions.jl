


function test_location_scale_normal(μ::Real, σ::Real, μD::Real, σD::Real,
                                    rng::Union{AbstractRNG, Missing} = missing)
    ρ = Normal(μD,σD)
    d = LocationScale(μ,σ,ρ)
    @test params(d) == (μ,σ,ρ)
    
    d_dict = Dict( # Different ways to construct the LocationScale object
                  "original" => d,
                  "sugar" => σ * ρ + μ,
                  "composed" =>  μ + (ρ * 2σ - 2) / 2 + 1
                 )
    dref = Normal(μ+σ*μD,σ*σD)

    @test d == deepcopy(d)

    #### Support / Domain

    @testset "Support" begin
        function test_support(d, dref)
            @test minimum(d) == minimum(dref)
            @test maximum(d) == maximum(dref)
            @test extrema(d) == (minimum(d), maximum(d))
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
        function test_evaluation_and_sampling(d, dref, rng)
            insupport(d,0.4) == insupport(dref,0.4)
            @test pdf(d,0.1) ≈ pdf(dref,0.1)
            @test pdf.(d,2:4) ≈ pdf.(dref,2:4)
            @test logpdf(d,0.4) ≈ logpdf(dref,0.4)
            @test loglikelihood(d,[0.1,0.2,0.3]) ≈ loglikelihood(dref,[0.1,0.2,0.3])
            @test loglikelihood(d, 0.4) ≈ loglikelihood(dref, 0.4)
            @test cdf(d,μ-0.4) ≈ cdf(dref,μ-0.4)
            @test logcdf(d,μ-0.4) ≈ logcdf(dref,μ-0.4)
            @test ccdf(d,μ-0.4) ≈ ccdf(dref,μ-0.4) atol=1e-15
            @test logccdf(d,μ-0.4) ≈ logccdf(dref,μ-0.4) atol=1e-16
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

            r = Array{partype(d)}(undef, 100000)
            if ismissing(rng)
                rand!(d,r)
            else
                rand!(rng,d,r)
            end
            @test mean(r) ≈ mean(dref) atol=0.01
            @test std(r) ≈ std(dref) atol=0.01
            @test cf(d, -0.1) ≈ cf(dref,-0.1)
            @test gradlogpdf(d, 0.1) ≈ gradlogpdf(dref, 0.1)
        end
        @testset "$k" for (k,dtest) in d_dict
            test_evaluation_and_sampling(dtest, dref, rng)
        end
    end

end

@testset "LocationScale" begin
    rng = MersenneTwister(123)
    test_location_scale_normal(0.3,0.2,0.1,0.2)
    test_location_scale_normal(-0.3,0.1,-0.1,0.3)
    test_location_scale_normal(1.3,0.4,-0.1,0.5)
    test_location_scale_normal(0.3,0.2,0.1,0.2,rng)
    test_location_scale_normal(-0.3,0.1,-0.1,0.3,rng)
    test_location_scale_normal(1.3,0.4,-0.1,0.5,rng)
    test_location_scale_normal(ForwardDiff.Dual(0.3),0.2,0.1,0.2, rng)
end
