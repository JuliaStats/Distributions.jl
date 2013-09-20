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

x = rand(Binomial(10, 0.3), n)
p = posterior(pri, Binomial, 10, x)
@test isa(p, Beta)
@test_approx_eq p.alpha pri.alpha + sum(x)
@test_approx_eq p.beta  pri.beta + (10n - sum(x))

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

f = fit_map(pri, Categorical, x)
@test isa(f, Categorical)
@test_approx_eq f.prob mode(p)

p = posterior(pri, Categorical, x, w)
@test isa(p, Dirichlet)
@test_approx_eq p.alpha pri.alpha + ccount(3, x, w)

f = fit_map(pri, Categorical, x, w)
@test isa(f, Categorical)
@test_approx_eq f.prob mode(p)

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


# posterior_sample

pri = Beta(1.0, 2.0)
x = rand(Bernoulli(0.3), n)
p = posterior(pri, Bernoulli, x)
ps = posterior_sample(p, Bernoulli)

@test isa(ps, Bernoulli)
@test zero(ps.p0) <= ps.p0 <= one(ps.p0)
@test zero(ps.p1) <= ps.p1 <= one(ps.p1)

ps = posterior_sample(pri, Bernoulli, x)
@test isa(ps, Bernoulli)
@test zero(ps.p0) <= ps.p0 <= one(ps.p0)
@test zero(ps.p1) <= ps.p1 <= one(ps.p1)







