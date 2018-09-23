import StatsBase: ProbabilityWeights

d = Generic([40., 80., 120., -60.],
            [.4, .3, .1,  .2])
println("    testing $d")

@test rand(d) ∈ [40., 80., 120., -60.]
@test rand(sampler(d)) ∈ [40., 80., 120., -60.]

@test pdf(d, -100.) == 0.
@test pdf(d, -60.) == .2
@test pdf(d, 100.) == 0.
@test pdf(d, 120.) == .1
@test pdf(d, -100) == 0.
@test pdf(d, -60) == .2

@test cdf(d, -100.) == 0
@test cdf(d, 0.) ≈ .2
@test cdf(d, 100.) ≈ .9
@test cdf(d, 150.) ≈ 1
@test cdf(d, -100) ≈ 0
@test cdf(d, 100) ≈ .9

@test quantile(d, 0) == -60
@test quantile(d, .1) == -60
@test quantile(d, .3) == 40
@test quantile(d, .7) == 80
@test quantile(d, 1) == 120
@test minimum(d) == -60
@test maximum(d) == 120

@test insupport(d, -60)
@test !insupport(d, 20)
@test insupport(d, 80)
@test !insupport(d, 150)
@test insupport(d, -60.)
@test !insupport(d, 20.)

xs = support(d)
ws = ProbabilityWeights(probs(d))
@test mean(d) ≈ mean(xs, ws)
@test var(d) ≈ var(xs, ws, corrected=false)
@test skewness(d) ≈ skewness(xs, ws)
@test kurtosis(d) ≈ kurtosis(xs, ws)
@test entropy(d) ≈ 1.2798542258336676
@test mode(d) == 40
@test modes(d) == [40]
@test mgf(d, 0) ≈ 1.0
@test mgf(d, 0.17) ≈ 7.262034e7
@test cf(d, 0) ≈ 1.0
@test cf(d, 0.17) ≈ 0.3604521478 + 0.6953481124im

# Fitting
xs = [1,2,3,4,5,4,3,2,3,2,1]
ss = suffstats(Generic, xs)
@test ss isa Distributions.GenericStats{Int,Float64,Vector{Int}}
@test ss.support == [1,2,3,4,5]
@test ss.freq == [2., 3., 3., 2., 1.]

d1 = fit_mle(Generic, ss)
@test d1 isa Generic{Int,Float64,Vector{Int}}
@test support(d1) == ss.support
@test probs(d1) ≈ ss.freq ./ 11

d2 = fit_mle(Generic, xs)
@test typeof(d2) == typeof(d1)
@test support(d2) == support(d1)
@test probs(d2) ≈ probs(d1)

d3 = fit(Generic, xs)
@test typeof(d2) == typeof(d1)
@test support(d3) == support(d1)
@test probs(d3) ≈ probs(d1)
