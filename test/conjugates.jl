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


# Beta - Bernoulli

pri = Beta(1.0, 2.0)

x = rand(Bernoulli(0.3), n)
p = posterior(pri, Bernoulli, x)
@test isa(p, Beta)
@test_approx_eq p.alpha pri.alpha + sum(x)
@test_approx_eq p.beta  pri.beta + (n - sum(x))

f = fit_map(pri, Bernoulli, x)
@test isa(f, Bernoulli)
@test_approx_eq f.p1 mode(p)

p = posterior(pri, Bernoulli, x, w)
@test isa(p, Beta)
@test_approx_eq p.alpha pri.alpha + sum(x .* w)
@test_approx_eq p.beta  pri.beta + (sum(w) - sum(x .* w))

f = fit_map(pri, Bernoulli, x, w)
@test isa(f, Bernoulli)
@test_approx_eq f.p1 mode(p)


# posterior_rand & posterior_randmodel

pri = Beta(1.0, 2.0)
x = rand(Bernoulli(0.3), n)
post = posterior(pri, Bernoulli, x)

pv = posterior_rand(pri, Bernoulli, x)
@test isa(pv, Float64)
@test 0. <= pv <= 1.

pv = posterior_rand(pri, Bernoulli, x, w)
@test isa(pv, Float64)
@test 0. <= pv <= 1.

pm = posterior_randmodel(pri, Bernoulli, x)
@test isa(pm, Bernoulli)
@test 0. <= pm.p1 <= 1.

pm = posterior_randmodel(pri, Bernoulli, x, w)
@test isa(pm, Bernoulli)
@test 0. <= pm.p1 <= 1.


# Beta - Binomial

x = rand(Binomial(10, 0.3), n)
p = posterior(pri, Binomial, (10, x))
@test isa(p, Beta)
@test_approx_eq p.alpha pri.alpha + sum(x)
@test_approx_eq p.beta  pri.beta + (10n - sum(x))

f = fit_map(pri, Binomial, (10, x))
@test isa(f, Binomial)
@test f.size == 10
@test_approx_eq f.prob mode(p)

p = posterior(pri, Binomial, (10, x), w)
@test isa(p, Beta)
@test_approx_eq p.alpha pri.alpha + sum(x .* w)
@test_approx_eq p.beta  pri.beta + (10 * sum(w) - sum(x .* w))

f = fit_map(pri, Binomial, (10, x), w)
@test isa(f, Binomial)
@test f.size == 10
@test_approx_eq f.prob mode(p)


# Dirichlet - Categorical

pri = Dirichlet([1., 2., 3.])

x = rand(Categorical([0.2, 0.3, 0.5]), n)
p = posterior(pri, Categorical, x)
@test isa(p, Dirichlet)
@test_approx_eq p.alpha pri.alpha + ccount(3, x)

f = fit_map(pri, Categorical, x)
@test isa(f, Categorical)
@test_approx_eq f.prob mode(p)

p = posterior(pri, Categorical, x, w)
@test isa(p, Dirichlet)
@test_approx_eq p.alpha pri.alpha + ccount(3, x, w)

f = fit_map(pri, Categorical, x, w)
@test isa(f, Categorical)
@test_approx_eq f.prob mode(p)


# Dirichlet - Multinomial

x = rand(Multinomial(100, [0.2, 0.3, 0.5]), 1)
p = posterior(pri, Multinomial, x)
@test isa(p, Dirichlet)
@test_approx_eq p.alpha pri.alpha + x

r = posterior_mode(pri, Multinomial, x)
@test_approx_eq r mode(p)

x = rand(Multinomial(10, [0.2, 0.3, 0.5]), n)
p = posterior(pri, Multinomial, x)
@test isa(p, Dirichlet)
@test_approx_eq p.alpha pri.alpha + vec(sum(x, 2))

r = posterior_mode(pri, Multinomial, x)
@test_approx_eq r mode(p)

p = posterior(pri, Multinomial, x, w)
@test isa(p, Dirichlet)
@test_approx_eq p.alpha pri.alpha + vec(x * w)

r = posterior_mode(pri, Multinomial, x, w)
@test_approx_eq r mode(p)


# Gamma - Exponential

pri = Gamma(1.5, 2.0)

x = rand(Exponential(2.0), n)
p = posterior(pri, Exponential, x)
@test isa(p, Gamma)
@test_approx_eq p.shape pri.shape + n
@test_approx_eq rate(p) rate(pri) + sum(x)

f = fit_map(pri, Exponential, x)
@test isa(f, Exponential)
@test_approx_eq rate(f) mode(p)

p = posterior(pri, Exponential, x, w)
@test isa(p, Gamma)
@test_approx_eq p.shape pri.shape + sum(w)
@test_approx_eq rate(p) rate(pri) + sum(x .* w)

f = fit_map(pri, Exponential, x, w)
@test isa(f, Exponential)
@test_approx_eq rate(f) mode(p)


