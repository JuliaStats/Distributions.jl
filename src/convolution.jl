"""
    convolve(d1::T, d2::T) where T<:Distribution -> Distribution

Convolve two distributions of the same type to yield the distribution corresponding to the
sum of independent random variables drawn from the underlying distributions.

The function is only defined in the cases where the convolution has a closed form as
defined here https://en.wikipedia.org/wiki/List_of_convolutions_of_probability_distributions

* Bernoulli
* Binomial
* Negative Binomial
* Geometric
* Poisson
* Normal
* Cauchy
* Chisq
* Exponential
* Gamma
* Multi-variate Normal
"""
#convolve(d1, d2) = throw(ArgumentError("`convolve` is not defined for $(typeof(d1)) and $(typeof(d2))"))

# discrete univariate
function convolve(d1::Bernoulli, d2::Bernoulli)
    _check_convolution_args(d1.p, d2.p)
    return Binomial(2, d1.p)
end

function convolve(d1::Binomial, d2::Binomial)
    _check_convolution_args(d1.p, d2.p)
    return Binomial(d1.n + d2.n, d1.p)
end

function convolve(d1::NegativeBinomial, d2::NegativeBinomial)
    _check_convolution_args(d1.p, d2.p)
    return NegativeBinomial(d1.r + d2.r, d1.p)
end

function convolve(d1::Geometric, d2::Geometric)
    _check_convolution_args(d1.p, d2.p)
    return NegativeBinomial(2, d1.p)
end

convolve(d1::Poisson, d2::Poisson) =  Poisson(d1.λ + d2.λ)


# continuous univariate
convolve(d1::Normal, d2::Normal) = Normal(d1.μ + d2.μ, hypot(d1.σ, d2.σ))
convolve(d1::Cauchy{T}, d2::Cauchy{T}) where T = Cauchy(d1.μ + d2.μ, d1.σ + d2.σ)
convolve(d1::Chisq, d2::Chisq) = Chisq(d1.ν + d2.ν)

function convolve(d1::Exponential, d2::Exponential)
    _check_convolution_args(d1.θ, d2.θ)
    return Gamma(2, d1.θ)
end

function convolve(d1::Gamma, d2::Gamma)
    _check_convolution_args(d1.θ, d2.θ)
    return Gamma(d1.α + d2.α, d1.θ)
end

# continuous multivariate
function convolve(
    d1::Union{IsoNormal, ZeroMeanIsoNormal},
    d2::Union{IsoNormal, ZeroMeanIsoNormal},
    )
    _check_convolution_shape(d1, d2)
    return MvNormal(d1.μ .+ d2.μ, d1.Σ.value + d2.Σ.value)
end

function convolve(
    d1::Union{DiagNormal, ZeroMeanDiagNormal},
    d2::Union{DiagNormal, ZeroMeanDiagNormal},
    )
    _check_convolution_shape(d1, d2)
    return MvNormal(d1.μ .+ d2.μ, d1.Σ.diag + d2.Σ.diag)
end

function convolve(
    d1::Union{FullNormal, ZeroMeanFullNormal},
    d2::Union{FullNormal, ZeroMeanFullNormal},
    )
    _check_convolution_shape(d1, d2)
    return MvNormal(d1.μ .+ d2.μ, d1.Σ.mat + d2.Σ.mat)
end

function convolve(d1::MvNormal, d2::MvNormal)
    _check_convolution_shape(d1, d2)
    return MvNormal(d1.μ .+ d2.μ, Matrix(d1.Σ) + Matrix(d2.Σ))
end


function _check_convolution_args(p1, p2)
    p1 == p2 || throw(ArgumentError("$(p1)!=$(p2): distribution parameters must be equal"))
end

function _check_convolution_shape(d1, d2)
    length(d1) == length(d2) || throw(ArgumentError("$d1 and $d2 are not the same size"))
end
