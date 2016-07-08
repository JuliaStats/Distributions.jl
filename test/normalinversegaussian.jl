using Distributions
using Base.Test

d = NormalInverseGaussian(1.7, 1.8, 1.2, 2.3)

@test_approx_eq_eps params(d)[1] 1.7 0.000000001
@test_approx_eq_eps params(d)[2] 1.8 0.000000001
@test_approx_eq_eps params(d)[3] 1.2 0.000000001
@test_approx_eq_eps params(d)[4] 2.3 0.000000001

# The solution was computed using this R code:
# dnig(4.2, mu=1.7, alpha=1.8, beta=1.2, delta=2.3)
@test_approx_eq_eps pdf(d, 4.2) 0.2021958 0.0000001

# The solution was computed using this R code:
# dnig(4.8, mu=1.7, alpha=1.8, beta=1.2, delta=2.3, log=TRUE)
@test_approx_eq_eps logpdf(d, 4.8) -1.909973 0.000001

# The solution was computed using this R code:
# mean(rnig(100000000, mu=1.7, alpha=1.8, beta=1.2, delta=2.3))
@test_approx_eq_eps mean(d) 3.757509 0.001

# The solution was computed using this R code:
# var(rnig(100000000, mu=1.7, alpha=1.8, beta=1.2, delta=2.3))
@test_approx_eq_eps var(d) 3.085488 0.001

# The solution was computed using this R code:
# skewness(rnig(100000000, mu=1.7, alpha=1.8, beta=1.2, delta=2.3))
@test_approx_eq_eps skewness(d) 1.138959 0.001

@test NormalInverseGaussian(1, 1, 2, 2) == NormalInverseGaussian(1., 1, 2, 2)
@test NormalInverseGaussian(1, 1, 2, 2) == NormalInverseGaussian(1., 1., 2., 2.)
@test typeof(convert(NormalInverseGaussian{Float64}, 1.7f0, 1.8f0, 1.2f0, 2.3f0)) == NormalInverseGaussian{Float64}
@test typeof(convert(NormalInverseGaussian{Float64}, NormalInverseGaussian(1.7f0, 1.8f0, 1.2f0, 2.3f0)))  == NormalInverseGaussian{Float64}
