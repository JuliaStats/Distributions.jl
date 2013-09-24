# Distributions with compound variates
#
#	NormalGamma
#	NormalInverseGamma
#	NormalWishart
#	NormalInverseWishart
#
# and their applications for conjugates & posteriors
#

using Base.Test
using Distributions

### NormalGamma

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


### NormalInverseGamma

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


### NormalWishart

n = 100

mu0 = [2., 3.]
kappa0 = 3.
nu0 = 4.
T0 = eye(2)
T0[1,2] = T0[2,1] = .5
nw = NormalWishart(mu0, kappa0, T0, nu0)

# Random number generation
mu, Lam = rand(nw)

# Did it generate something valid?
@test insupport(NormalWishart, mu, Lam)

mu = [1.5, 2.5]
T = 0.75*eye(2)
T[1,2] = T[2,1] = 0.6

# pdf
npdf = pdf(MultivariateNormal(mu0, inv(kappa0*T)), mu)
wpdf = pdf(Wishart(nu0, T0), T)
lnpdf = logpdf(MultivariateNormal(mu0, inv(kappa0*T)), mu)
lwpdf = logpdf(Wishart(nu0, T0), T)

@test_approx_eq_eps pdf(nw, mu, T) (npdf*wpdf) 1e-8
@test_approx_eq_eps logpdf(nw, mu, T) (lnpdf+lwpdf) 1e-8


### NormalInverseWishart

n = 100

mu0 = [2., 3.]
kappa0 = 3.
nu0 = 4.
T0 = eye(2)
T0[1,2] = T0[2,1] = .5
niw = NormalInverseWishart(mu0, kappa0, T0, nu0)

# Random number generation
mu, Sig = rand(niw)

# Did it generate something valid?
@test insupport(NormalInverseWishart, mu, Sig)

mu = [1.5, 2.5]
T = 0.75*eye(2)
T[1,2] = T[2,1] = 0.6

# pdf
npdf = pdf(MultivariateNormal(mu0, 1/kappa0*T), mu)
wpdf = pdf(InverseWishart(nu0, T0), T)
lnpdf = logpdf(MultivariateNormal(mu0, 1/kappa0*T), mu)
lwpdf = logpdf(InverseWishart(nu0, T0), T)

@test_approx_eq_eps pdf(niw, mu, T) (npdf*wpdf) 1e-8
@test_approx_eq_eps logpdf(niw, mu, T) (lnpdf+lwpdf) 1e-8




