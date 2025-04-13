"""
    convolve(d1::Distribution, d2::Distribution)

Convolve two distributions and return the distribution corresponding to the sum of
independent random variables drawn from the underlying distributions.

Currently, the function is only defined in cases where the convolution has a closed form.
More precisely, the function is defined if the distributions of `d1` and `d2` are the same
and one of
* [`Bernoulli`](@ref)
* [`Binomial`](@ref)
* [`NegativeBinomial`](@ref)
* [`Geometric`](@ref)
* [`Poisson`](@ref)
* [`DiscreteNonParametric`](@ref)
* [`Normal`](@ref)
* [`Cauchy`](@ref)
* [`Chisq`](@ref)
* [`Exponential`](@ref)
* [`Gamma`](@ref)
* [`MvNormal`](@ref)

External links: [List of convolutions of probability distributions on Wikipedia](https://en.wikipedia.org/wiki/List_of_convolutions_of_probability_distributions)
"""
convolve(::Distribution, ::Distribution)

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


function convolve(d1::DiscreteNonParametric, d2::DiscreteNonParametric)
    support_conv = collect(Set(s1 + s2 for s1 in support(d1), s2 in support(d2)))
    sort!(support_conv) #for fast index finding below
    probs1 = probs(d1)
    probs2 = probs(d2)
    p_conv = zeros(Base.promote_eltype(probs1, probs2), length(support_conv)) 
    for (s1, p1) in zip(support(d1), probs(d1)), (s2, p2) in zip(support(d2), probs(d2))
            idx = searchsortedfirst(support_conv, s1+s2)
            p_conv[idx] += p1*p2
    end
    DiscreteNonParametric(support_conv, p_conv,check_args=false) 
end

# continuous univariate
convolve(d1::Normal, d2::Normal) = Normal(d1.μ + d2.μ, hypot(d1.σ, d2.σ))
convolve(d1::Cauchy, d2::Cauchy) = Cauchy(d1.μ + d2.μ, d1.σ + d2.σ)
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
function convolve(d1::MvNormal, d2::MvNormal)
    _check_convolution_shape(d1, d2)
    return MvNormal(d1.μ + d2.μ, d1.Σ + d2.Σ)
end

function _check_convolution_args(p1, p2)
    p1 ≈ p2 || throw(ArgumentError(
    "$(p1) !≈ $(p2): distribution parameters must be approximately equal",
    ))
end

function _check_convolution_shape(d1, d2)
    length(d1) == length(d2) || throw(ArgumentError("$d1 and $d2 are not the same size"))
end
