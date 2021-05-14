"""
    pdfsquaredL2norm(d::Distribution)

Return the square of the L2 norm of the probability density function ``f(x)`` of the distribution `d`:

```math
\\big(\\int_{S} f(x)^{2} \\mathrm{d} x \\big)^{1/2}
```

where ``S`` is the support of ``f(x)``.
"""
pdfsquaredL2norm

function pdfsquaredL2norm(d::Beta)
    α, β = params(d)
    z = beta(2 * α - 1, 2 * β - 1) / beta(α, β) ^ 2
    # L2 norm of the pdf converges only for α > 0.5 and β > 0.5
    return α > 0.5 && β > 0.5 ? z : oftype(z, Inf)
end

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

pdfsquaredL2norm(d::Exponential) = 1 / (2 * d.θ)

function pdfsquaredL2norm(d::Gamma{T}) where {T}
    α, θ = params(d)
    z = (2^(1 - 2 * α) * gamma(2 * α - 1)) / (gamma(α) ^ 2 * θ)
    # L2 norm of the pdf converges only for α > 0.5
    return α > 0.5 ? z : oftype(z, Inf)
end

pdfsquaredL2norm(d::Logistic) = 1 / (6 * d.θ)

pdfsquaredL2norm(d::Normal) = inv(sqrt4π * d.σ)

pdfsquaredL2norm(d::Uniform) = 1 / (d.b - d.a)
