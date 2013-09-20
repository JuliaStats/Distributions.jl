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

mu_true = [2., 2.]
Lam_true = eye(2)
Lam_true[1,2] = Lam_true[2,1] = 0.25

X = rand(MultivariateNormal(mu_true, inv(Lam_true)), n)
Xbar = mean(X,2)
Xm = X .- Xbar

pri = NormalWishart(mu0, kappa0, T0, nu0)

post = posterior(pri, MvNormal, X)

@test_approx_eq post.mu (kappa0*mu0 + n*Xbar)./(kappa0 + n)
@test_approx_eq post.kappa kappa0 + n
@test_approx_eq post.nu nu0 + n
@test_approx_eq (post.Tchol[:U]'*post.Tchol[:U]) T0 + A_mul_Bt(Xm, Xm) + kappa0*n/(kappa0+n)*(Xbar-mu0)*(Xbar-mu0)'

# posterior_sample

ps = posterior_sample(pri, MvNormal, X)

@test isa(ps, MultivariateNormal)
@test insupport(ps, ps.μ)
@test insupport(InverseWishart, ps.Σ.mat)  # InverseWishart on purpose


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

mu_true = [2., 2.]
Sig_true = eye(2)
Sig_true[1,2] = Sig_true[2,1] = 0.25

X = rand(MultivariateNormal(mu_true, Sig_true), n)
Xbar = mean(X,2)
Xm = X .- mean(X,2)

pri = NormalInverseWishart(mu0, kappa0, T0, nu0)

post = posterior(pri, MvNormal, X)

@test_approx_eq post.mu (kappa0*mu0 + n*Xbar)./(kappa0 + n)
@test_approx_eq post.kappa kappa0 + n
@test_approx_eq post.nu nu0 + n
@test_approx_eq (post.Lamchol[:U]'*post.Lamchol[:U]) T0 + A_mul_Bt(Xm, Xm) + kappa0*n/(kappa0+n)*(Xbar-mu0)*(Xbar-mu0)'

# posterior_sample

ps = posterior_sample(pri, MultivariateNormal, X)

@test isa(ps, MultivariateNormal)
@test insupport(ps, ps.μ) && insupport(InverseWishart, ps.Σ.mat)


