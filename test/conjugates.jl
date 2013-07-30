using Base.Test
using Distributions

n = 100
w = rand(100)

# auxiliary tools

function ccount(K, x)
	r = zeros(K)
	for i = 1:length(x)
		r[x[i]] += 1.0
	end
	r
end

function ccount(K, x, w)
	r = zeros(K)
	for i = 1:length(x)
		r[x[i]] += w[i]
	end
	r
end


# Beta - Bernoulli / Binomial

pri = Beta(1.0, 2.0)

x = rand(Bernoulli(0.3), n)
p = posterior(pri, Bernoulli, x)
@test isa(p, Beta)
@test_approx_eq p.alpha pri.alpha + sum(x)
@test_approx_eq p.beta  pri.beta + (n - sum(x))

x = rand(Bernoulli(0.3), n)
p = posterior(pri, Bernoulli, x, w)
@test isa(p, Beta)
@test_approx_eq p.alpha pri.alpha + sum(x .* w)
@test_approx_eq p.beta  pri.beta + (sum(w) - sum(x .* w))

x = rand(Binomial(10, 0.3), n)
p = posterior(pri, Binomial, 10, x)
@test isa(p, Beta)
@test_approx_eq p.alpha pri.alpha + sum(x)
@test_approx_eq p.beta  pri.beta + (10n - sum(x))

x = rand(Binomial(10, 0.3), n)
p = posterior(pri, Binomial, 10, x, w)
@test isa(p, Beta)
@test_approx_eq p.alpha pri.alpha + sum(x .* w)
@test_approx_eq p.beta  pri.beta + (10 * sum(w) - sum(x .* w))

# Dirichlet - Categorical / Multinomial

pri = Dirichlet([1., 2., 3.])

x = rand(Categorical([0.2, 0.3, 0.5]), n)
p = posterior(pri, Categorical, x)
@test isa(p, Dirichlet)
@test_approx_eq p.alpha pri.alpha + ccount(3, x)

x = rand(Categorical([0.2, 0.3, 0.5]), n)
p = posterior(pri, Categorical, x, w)
@test isa(p, Dirichlet)
@test_approx_eq p.alpha pri.alpha + ccount(3, x, w)

x = rand(Multinomial(100, [0.2, 0.3, 0.5]), 1)
p = posterior(pri, Multinomial, x)
@test isa(p, Dirichlet)
@test_approx_eq p.alpha pri.alpha + x

x = rand(Multinomial(10, [0.2, 0.3, 0.5]), n)
p = posterior(pri, Multinomial, x)
@test isa(p, Dirichlet)
@test_approx_eq p.alpha pri.alpha + vec(sum(x, 2))

x = rand(Multinomial(10, [0.2, 0.3, 0.5]), n)
p = posterior(pri, Multinomial, x, w)
@test isa(p, Dirichlet)
@test_approx_eq p.alpha pri.alpha + vec(x * w)


# Gamma - Exponential

pri = Gamma(1.5, 2.0)

x = rand(Exponential(2.0), n)
p = posterior(pri, Exponential, x)
@test isa(p, Gamma)
@test_approx_eq p.shape pri.shape + n
@test_approx_eq rate(p) rate(pri) + sum(x)

x = rand(Exponential(2.0), n)
p = posterior(pri, Exponential, x, w)
@test isa(p, Gamma)
@test_approx_eq p.shape pri.shape + sum(w)
@test_approx_eq rate(p) rate(pri) + sum(x .* w)


# Normal likelihood

# x = rand(Normal(1.7, 3.0), 10_000)
# newprior = posterior(Normal(0.0, 1.0), 3.0, Normal, x)
# @test_approx_eq_eps mean(newprior) 1.7 0.1

# x = rand(Normal(1.7, 3.0), 10_000)
# newprior = posterior(1.7, InvertedGamma(1.0, 1.0), Normal, x)
# @test_approx_eq_eps sqrt(mean(newprior)) 3.0 0.1

