export pdfL2norm

"""
    pdfL2norm(d::Distribution)

Return the L2 norm of the probability distribution function `f(x)` of the distribution `d`:

```math
\\int_{S} f(x)^{2} \\mathrm{d} x
```

where `S` is the support of `f(x)`.
"""
pdfL2norm

pdfL2norm(d::Distribution) = throw(ArgumentError("L2 norm not implemented for $(d)"))

function pdfL2norm(d::Beta{T}) where T
    if d.α > 0.5 && d.β > 0.5
        return beta(2 * d.α - 1, 2 * d.β - 1) / beta(d.α, d.β) ^ 2
    else
        return T(Inf)
    end
end

pdfL2norm(d::Cauchy) =  1 / (d.σ * 2 * π)

function pdfL2norm(d::Chi{T}) where {T}
    if d.ν > 0.5
        return (2 ^ (1 - d.ν) * gamma((2 * d.ν - 1) / 2)) / gamma(d.ν / 2) ^ 2
    else
        return T(Inf)
    end
end

function pdfL2norm(d::Chisq{T}) where {T}
    if d.ν > 1
        return gamma(d.ν - 1) / (gamma(d.ν / 2) ^ 2 * 2 ^ d.ν)
    else
        return T(Inf)
    end
end

pdfL2norm(d::Exponential) = 1 / (2 * d.θ)

function pdfL2norm(d::Gamma{T}) where {T}
    if d.α > 0.5
        return (2^(1 - 2 * d.α) * gamma(2 * d.α - 1)) / (gamma(d.α) ^ 2 * d.θ)
    else
        return T(Inf)
    end
end

pdfL2norm(d::Logistic) = 1 / (6 * d.θ)

pdfL2norm(d::Normal{T}) where {T} = 1 / (2 * sqrt(T(π)) * d.σ)

pdfL2norm(d::Uniform) = 1 / (d.b - d.a)
