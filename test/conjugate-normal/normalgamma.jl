using Distributions
using Base.Test

let

n = 100

mu0 = 2.
nu0 = 3.
shape0 = 5.
rate0 = 2.
ng = NormalGamma(mu0, nu0, shape0, rate0)

# Random number generation
mu, tau2 = rand(ng)

# Did it generate something valid?
@test insupport(NormalGamma, mu, tau2)

mu = 2.5
tau2 = 3.

# pdf
npdf = pdf(Normal(mu0, 1./sqrt(nu0*tau2)), mu)
gpdf = pdf(Gamma(shape0, 1./rate0), tau2)
lnpdf = logpdf(Normal(mu0, 1./sqrt(nu0*tau2)), mu)
lgpdf = logpdf(Gamma(shape0, 1./rate0), tau2)

@test_approx_eq_eps pdf(ng, mu, tau2) (npdf*gpdf) 1e-8
@test_approx_eq_eps logpdf(ng, mu, tau2) (lnpdf+lgpdf) 1e-8

# Posterior
mu_true = 2.
tau2_true = 3.
x = rand(Normal(mu_true, 1./tau2_true), n)

pri = NormalGamma(mu0, nu0, shape0, rate0)

post = posterior(pri, Normal, x)
@test isa(post, NormalGamma)

@test_approx_eq post.mu (nu0*mu0 + n*mean(x))./(nu0 + n)
@test_approx_eq post.nu nu0 + n
@test_approx_eq post.shape shape0 + 0.5*n
@test_approx_eq post.rate rate0 + 0.5*(n-1)*var(x) + n*nu0/(n + nu0)*0.5*(mean(x)-mu0).^2

# posterior_sample

ps = posterior_sample(pri, Normal, x)

@test isa(ps, Normal)
@test insupport(ps, ps.μ) && ps.σ > zero(ps.σ)


end
