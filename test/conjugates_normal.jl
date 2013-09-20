# Conjugates for normal distribution

using Base.Test
using Distributions


n = 100
w = rand(100)

# normal - normal (known sigma)

pri = Normal(1.0, 5.0)

x = rand(Normal(2.0, 3.0), n)
p = posterior((pri, 3.0), Normal, x)
@test isa(p, Normal)
@test_approx_eq mean(p)  (mean(pri) / var(pri) + sum(x) / 9.0) / (1.0 / var(pri) + n / 9.0)
@test_approx_eq var(p) inv(1.0 / var(pri) + n / 9.0)

r = posterior_mode((pri, 3.0), Normal, x)
@test_approx_eq r mode(p)

f = fit_map((pri, 3.0), Normal, x)
@test isa(f, Normal)
@test f.μ == r
@test f.σ == 3.0

p = posterior((pri, 3.0), Normal, x, w)
@test isa(p, Normal)
@test_approx_eq mean(p)  (mean(pri) / var(pri) + dot(x, w) / 9.0) / (1.0 / var(pri) + sum(w) / 9.0)
@test_approx_eq var(p) inv(1.0 / var(pri) + sum(w) / 9.0)

r = posterior_mode((pri, 3.0), Normal, x, w)
@test_approx_eq r mode(p)

f = fit_map((pri, 3.0), Normal, x, w)
@test isa(f, Normal)
@test f.μ == r
@test f.σ == 3.0

# inversegamma - normal (known mu)

pri = InverseGamma(1.5, 0.5) # β = 2.0

x = rand(Normal(2.0, 3.0), n)
p = posterior((2.0, pri), Normal, x)
@test isa(p, InverseGamma)
@test_approx_eq p.shape pri.shape + n / 2
@test_approx_eq p.scale pri.scale + sum(abs2(x - 2.0)) / 2

p = posterior((2.0, pri), Normal, x, w)
@test isa(p, InverseGamma)
@test_approx_eq p.shape pri.shape + sum(w) / 2
@test_approx_eq p.scale pri.scale + dot(w, abs2(x - 2.0)) / 2


# MvNormal -- Normal (known covariance)

n = 3
p = 4
X = reshape(Float64[1:12], p, n)
w = rand(n)
Xw = X * diagm(w)

# Convoluted way to put 1's on diag
Sigma = eye(p)
Sigma += 0.25
Sigma -= 0.25*eye(p)

ss = suffstats(MvNormalKnownSigma(Sigma), X)
ssw = suffstats(MvNormalKnownSigma(Sigma), X, w)

s_t = sum(X, 2)
ws_t = sum(Xw, 2)
tw_t = length(w)
wtw_t = sum(w)

@test_approx_eq ss.s s_t
@test_approx_eq ss.tw tw_t

@test_approx_eq ssw.s ws_t
@test_approx_eq ssw.tw wtw_t

# Posterior
n = 100
mu_true = [2., 3.]
Sig_true = eye(2)
Sig_true[1,2] = Sig_true[2,1] = 0.25
mu0 = [2.5, 2.5]
Sig0 = eye(2)
Sig0[1,2] = Sig0[2,1] = 0.5
X = rand(MultivariateNormal(mu_true, Sig_true), n)

pri = MultivariateNormal(mu0, Sig0)

post = posterior((pri, Sig_true), MultivariateNormal, X)
@test isa(post, MultivariateNormal)

@test_approx_eq post.μ inv(inv(Sig0) + n*inv(Sig_true))*(n*inv(Sig_true)*mean(X,2) + inv(Sig0)*mu0)
@test_approx_eq post.Σ.mat inv(inv(Sig0) + n*inv(Sig_true))

# posterior_sample

ps = posterior_sample((pri, Sig_true), MultivariateNormal, X)

@test isa(ps, MultivariateNormal)
@test insupport(ps, ps.μ)
@test insupport(InverseWishart, ps.Σ.mat)

