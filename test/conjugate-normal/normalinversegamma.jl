using Distributions
using Base.Test


n = 100

mu0 = 2.
v0 = 3.
shape0 = 5.
scale0 = 2.
nig = NormalInverseGamma(mu0, v0, shape0, scale0)

# Random number generation
mu, sig2 = rand(nig)

# Did it generate something valid?
@test insupport(NormalInverseGamma, mu, sig2)

mu = 2.5
sig2 = 3.

# pdf
npdf = pdf(Normal(mu0, sqrt(v0*sig2)), mu)
gpdf = pdf(InverseGamma(shape0, scale0), sig2)
lnpdf = logpdf(Normal(mu0, sqrt(v0*sig2)), mu)
lgpdf = logpdf(InverseGamma(shape0, scale0), sig2)

@test_approx_eq_eps pdf(nig, mu, sig2) (npdf*gpdf) 1e-8
@test_approx_eq_eps logpdf(nig, mu, sig2) (lnpdf+lgpdf) 1e-8

# Posterior
mu_true = 2.
sig2_true = 3.
x = rand(Normal(mu_true, sig2_true), n)

pri = NormalInverseGamma(mu0, v0, shape0, scale0)

post = posterior(pri, Normal, x)
@test isa(post, NormalInverseGamma)

@test_approx_eq post.mu (mu0/v0 + n*mean(x))/(1./v0 + n)
@test_approx_eq post.v0 1./(1./v0 + n)
@test_approx_eq post.shape shape0 + 0.5*n
@test_approx_eq post.scale scale0 + 0.5*(n-1)*var(x) + n./v0./(n + 1./v0)*0.5*(mean(x)-mu0).^2

# posterior_sample

ps = posterior_sample(pri, Normal, x)

@test isa(ps, Normal)
@test insupport(ps,ps.μ) && ps.σ > zero(ps.σ)

