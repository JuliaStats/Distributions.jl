import StatsBase: ProbabilityWeights

d = Generic([40., 80., 120., -60.],
            [.4, .3, .1,  .2])

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
@test var(d) ≈ var(xs, ws)
@test skewness(d) ≈ skewness(xs, ws)
@test kurtosis(d) ≈ kurtosis(xs, ws)
@test entropy(d) ≈ 1.2798542258336676
@test mode(d) == 40
@test modes(d) == [40]
