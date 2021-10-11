"""
    pdfsquaredL2norm(d::Distribution)

Return the square of the L2 norm of the probability density function ``f(x)`` of the distribution `d`:

```math
\\int_{S} f(x)^{2} \\mathrm{d} x
```

where ``S`` is the support of ``f(x)``.
"""
pdfsquaredL2norm

pdfsquaredL2norm(d::Bernoulli) = @evalpoly d.p 1 -2 2

function pdfsquaredL2norm(d::Beta)
    α, β = params(d)
    z = beta(2 * α - 1, 2 * β - 1) / beta(α, β) ^ 2
    # L2 norm of the pdf converges only for α > 0.5 and β > 0.5
    return α > 0.5 && β > 0.5 ? z : oftype(z, Inf)
end

pdfsquaredL2norm(d::DiscreteNonParametric) = dot(probs(d), probs(d))

pdfsquaredL2norm(d::Cauchy) = inv2π / d.σ

function pdfsquaredL2norm(d::Chi)
    ν = d.ν
    z = (2 ^ (1 - ν) * gamma((2 * ν - 1) / 2)) / gamma(ν / 2) ^ 2
    # L2 norm of the pdf converges only for ν > 0.5
    return d.ν > 0.5 ? z : oftype(z, Inf)
end

function pdfsquaredL2norm(d::Chisq)
    ν = d.ν
    z = gamma(d.ν - 1) / (gamma(d.ν / 2) ^ 2 * 2 ^ d.ν)
    # L2 norm of the pdf converges only for ν > 1
    return ν > 1 ? z : oftype(z, Inf)
end

pdfsquaredL2norm(d::DiscreteUniform) = 1 / (d.b - d.a + 1)

pdfsquaredL2norm(d::Exponential) = 1 / (2 * d.θ)

function pdfsquaredL2norm(d::Gamma)
    α, θ = params(d)
    z = (2^(1 - 2 * α) * gamma(2 * α - 1)) / (gamma(α) ^ 2 * θ)
    # L2 norm of the pdf converges only for α > 0.5
    return α > 0.5 ? z : oftype(z, Inf)
end

pdfsquaredL2norm(d::Geometric) = d.p ^ 2 / (2 * d.p - d.p ^ 2)

pdfsquaredL2norm(d::Logistic) = 1 / (6 * d.θ)

pdfsquaredL2norm(d::Normal) = inv(sqrt4π * d.σ)

# The identity is obvious if you look at the definition of the modified Bessel function of
# first kind I_0.  Starting from the L2-norm of the Poisson distribution this can be proven
# by observing this is the Laguerre exponential l−e of x², which is related to said Bessel
# function, see <https://doi.org/10.1140/epjst/e2018-00073-1> (preprint:
# <https://arxiv.org/abs/1707.01135>).
pdfsquaredL2norm(d::Poisson) = besseli(0, 2 * d.λ) * exp(-2 * d.λ)

pdfsquaredL2norm(d::Uniform) = 1 / (d.b - d.a)
