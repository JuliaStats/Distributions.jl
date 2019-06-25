"""
    convolve(d1::T, d2::T) where T<:Distribution -> Distribution

Convole two distributions of the same type where the convolution has a closed form, e.g.

* Bernoulli
* Binomial
* NegativeBinomial
* Geometric
* Poisson
* Normal
* Cauchy
* Chisq
* Exponential
* Gamma
* Multivariate Normal
"""
convolve(d1, d2) = @throws(MethodError("`convolve` is not defined for $d1 and $d2"))

# discrete univariate
function convolve(d1::Bernoulli, d2::Bernoulli)
    _check_params(d1.p, d2.p)
    return Binomial(2, d1.p)
end

function convolve(d1::Binomial, d2::Binomial)
    _check_params(d1.p, d2.p)
    return Binomial(d1.n + d2.n, d1.p)
end

function convolve(d1::NegativeBinomial, d2::NegativeBinomial)
    _check_params(d1.p, d2.p)
    return NegativeBinomial(d1.n + d2.n, d1.p)
end

function convolve(d1::Geometric, d2::Geometric)
    _check_params(d1.p, d2.p)
    return NegativeBinomial(2, d1.p)
end

function convolve(d1::Poisson, d2::Poisson)
    return Poisson(d1.λ + d2.λ)
end


# continuous univariate
convolve(d1::Normal, d2::Normal) = Normal(d1.μ + d2.μ, √(d1.σ^2 + d2.σ^2))
convolve(d1::Cauchy{T}, d2::Cauchy{T}) where T = Cauchy(d1.μ + d2.μ, d1.σ + d2.σ)
convolve(d1::Chisq, d2::Chisq) = Chisq(d1.ν + d2.ν)

function convolve(d1::Exponential, d2::Exponential)
    _check_params(d1.θ, d2.θ)
    return Gamma(2, d1.θ)
end

function convolve(d1::Gamma, d2::Gamma)
    _check_params(d1.θ, d2.θ)
    return Gamma(d1.α + d2.α, d1.θ)
end

# continuous multivariate
convolve(d1::T, d2::T) where T<:Union{IsoNormal, ZeroMeanIsoNormal}
    _check_params(length(d1), length(d2))
    return MvNormal(d1.μ + d2.μ, d1.Σ.value + d2.Σ.value)
end

convolve(d1::T, d2::T) where T<:Union{DiagNormal, ZeroMeanDiagNormal}
    _check_params(length(d1), length(d2))
    return MvNormal(d1.μ + d2.μ, d1.Σ.diag + d2.Σ.diag)
end

convolve(d1::T, d2::T) where T<:Union{FullNormal, ZeroMeanFullNormal}
    _check_params(length(d1), length(d2))
    return MvNormal(d1.μ + d2.μ, d1.Σ.mat + d2.Σ.mat)
end

convolve(d1::MvNormal, d2::MvNormal)
    _check_params(length(d1), length(d2))
    return MvNormal(d1.μ + d2.μ, Matrix(d1.Σ) + Matrix(d2.Σ.mat))
end



function _check_params(p1, p2)
    if p1 != p2
        throw(ArgumentError("$(p1)!=$(p2): distribution parameters must be equal"))
    end
end
