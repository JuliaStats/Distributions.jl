
using Distributions
using Base.Test

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
