import StatsBase: ProbabilityWeights
using Random, Distributions
using Test

rng = MersenneTwister(123)

@testset "Testing matrix-variates with $key" for (key, func) in
    Dict("rand(...)" => [rand, rand],
         "rand(rng, ...)" => [dist -> rand(rng, dist), (dist, n) -> rand(rng, dist, n)])

d = DiscreteNonParametric([40., 80., 120., -60.], [.4, .3, .1,  .2])
d_string = DiscreteNonParametric(["forty", "eighty", "one hundred and twenty", "negative sixty"], [.4, .3, .1,  .2])

@test !(d ≈ DiscreteNonParametric([40., 80, 120, -60], [.4, .3, .1, .2], Distributions.NoArgCheck()))
@test d ≈ DiscreteNonParametric([-60., 40., 80, 120], [.2, .4, .3, .1], Distributions.NoArgCheck())

# Invalid probability
@test_throws ArgumentError DiscreteNonParametric([40., 80, 120, -60], [.5, .3, .1, .2])
@test_throws ArgumentError DiscreteNonParametric(["forty", "eighty", "one hundred and twenty", "negative sixty"], [.5, .3, .1, .2])

# Invalid probability, but no arg check
DiscreteNonParametric([40., 80, 120, -60], [.5, .3, .1, .2], Distributions.NoArgCheck())
DiscreteNonParametric(["forty", "eighty", "one hundred and twenty", "negative sixty"], [.5, .3, .1, .2], Distributions.NoArgCheck())

test_range(d)
vs = Distributions.get_evalsamples(d, 0.00001)
vs_string = Distributions.get_evalsamples(d_string, 0.00001)
test_evaluation(d, vs, true)
test_evaluation(d_string, vs_string, true)
test_stats(d, vs)
test_params(d)
test_params(d_string)

@test func[1](d) ∈ [40., 80., 120., -60.]
@test func[1](d_string) ∈ ["forty", "eighty", "one hundred and twenty", "negative sixty"]
@test func[1](d) ∈ [40., 80., 120., -60.]
@test func[1](d_string) ∈ ["forty", "eighty", "one hundred and twenty", "negative sixty"]
@test func[1](sampler(d)) ∈ [40., 80., 120., -60.]
@test func[1](sampler(d_string)) ∈ ["forty", "eighty", "one hundred and twenty", "negative sixty"]

@test pdf(d, -100.) == 0.
@test pdf(d_string, "negative one hundred") == 0.
@test pdf(d, -100) == 0.
@test pdf(d_string, "negative one hundred") == 0.
@test pdf(d, -60.) == .2
@test pdf(d_string, "negative sixty") == .2
@test pdf(d, -60) == .2
@test pdf(d_string, "negative sixty") == .2
@test pdf(d, 100.) == 0.
@test pdf(d_string, "one hundred") == 0.
@test pdf(d, 120.) == .1
@test pdf(d_string, "one hundred and twenty") == .1

@test cdf(d, -100.) == 0.
@test cdf(d, -100) == 0.
@test cdf(d, 0.) ≈ .2
@test cdf(d, 100.) ≈ .9
@test cdf(d, 100) ≈ .9
@test cdf(d, 150.) == 1.

@test ccdf(d, -100.) == 1.
@test ccdf(d, -100) == 1.
@test ccdf(d, 0.) ≈ .8
@test ccdf(d, 100.) ≈ .1
@test ccdf(d, 100) ≈ .1
@test ccdf(d, 150.) == 0

@test quantile(d, 0) == -60
@test quantile(d, .1) == -60
@test quantile(d, .3) == 40
@test quantile(d, .7) == 80
@test quantile(d, 1) == 120
@test minimum(d) == -60
@test maximum(d) == 120

@test insupport(d, -60)
@test insupport(d, -60.)
@test in("negative sixty", support(d_string))
@test !insupport(d, 20)
@test !insupport(d, 20.)
@test !in("twenty", support(d_string))
@test insupport(d, 80)
@test in("eighty", support(d_string))
@test !insupport(d, 150)
@test !in("one hundred and fifty", support(d_string))

xs = support(d)
xs_string = support(d_string)
ws = ProbabilityWeights(probs(d))
ws_string = ProbabilityWeights(probs(d_string))
@test mean(d) ≈ mean(xs, ws)
@test var(d) ≈ var(xs, ws, corrected=false)
@test skewness(d) ≈ skewness(xs, ws)
@test kurtosis(d) ≈ kurtosis(xs, ws)
@test entropy(d) ≈ 1.2798542258336676
@test mode(d) == 40
@test mode(d_string) == "forty"
@test modes(d) == [40]
@test modes(d_string) == ["forty"]
@test mgf(d, 0) ≈ 1.0
@test mgf(d, 0.17) ≈ 7.262034e7
@test cf(d, 0) ≈ 1.0
@test cf(d, 0.17) ≈ 0.3604521478 + 0.6953481124im

# Fitting
xs = [1, 2,3,4,5,4,3,2,3,2,1]
ws = [1.,2,1,0,1,1,2,3,2,6,1]

ss = suffstats(DiscreteNonParametric, xs)
@test ss isa Distributions.DiscreteNonParametricStats{Int,Float64,Vector{Int}}
@test ss.support == [1,2,3,4,5]
@test ss.freq == [2., 3., 3., 2., 1.]

ss2 = suffstats(DiscreteNonParametric, xs, ws)
@test ss2 isa Distributions.DiscreteNonParametricStats{Int,Float64,Vector{Int}}
@test ss2.support == [1,2,3,4,5]
@test ss2.freq == [2., 11., 5., 1., 1.]

d1 = fit_mle(DiscreteNonParametric, ss)
@test d1 isa DiscreteNonParametric{Int,Float64,Vector{Int}}
@test support(d1) == ss.support
@test probs(d1) ≈ ss.freq ./ 11

d2 = fit_mle(DiscreteNonParametric, xs)
@test typeof(d2) == typeof(d1)
@test support(d2) == support(d1)
@test probs(d2) ≈ probs(d1)

d3 = fit(DiscreteNonParametric, xs)
@test typeof(d2) == typeof(d1)
@test support(d3) == support(d1)
@test probs(d3) ≈ probs(d1)

# Numerical stability; see issue #872 and PR #926
p = [1 - eps(Float32), eps(Float32)]
d = Categorical(p)
@test ([rand(d) for _ = 1:100_000]; true)   

end
