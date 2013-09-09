using NumericExtensions
using Distributions
using Base.Test
using Base.LinAlg


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
Xm = vbroadcast(Subtract(), X, mean(X,2), 1)
Xbar = mean(X,2)

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

