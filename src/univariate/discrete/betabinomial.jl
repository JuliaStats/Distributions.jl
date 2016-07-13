doc"""
    BetaBinomial(n,α,β)

A *Beta-binomial distribution* is the compound distribution of the [`Binomial`](:func:`Binomial`) distribution where the probability of success `p` is distributed according to the [`Beta`](:func:`Beta`). It has three parameters: `n`, the number of trials and two shape parameters `α`, `β`

$P(X = k) = {n \choose k} B(k + \alpha, n - k + \beta) / B(\alpha, \beta),  \quad \text{ for } k = 0,1,2, \ldots, n.$

```julia
BetaBinomial(n, a, b)      # BetaBinomial distribution with n trials and shape parameters a, b

params(d)       # Get the parameters, i.e. (n, a, b)
ntrials(d)      # Get the number of trials, i.e. n
```

External links:

* [Beta-binomial distribution on Wikipedia](https://en.wikipedia.org/wiki/Beta-binomial_distribution)
"""

immutable BetaBinomial{T<:Real} <: DiscreteUnivariateDistribution
    n::Int
    α::T
    β::T

    function BetaBinomial(n::Int, α::T, β::T)
        @check_args(BetaBinomial, n >= zero(n) && α >= zero(α) && β >= zero(β))
        new(n, α, β)
    end
end

BetaBinomial{T<:Real}(n::Int, α::T, β::T) = BetaBinomial{T}(n, α, β)
BetaBinomial(n::Int, α::Real, β::Real) = BetaBinomial(n, promote(α, β)...)
BetaBinomial(n::Int, α::Integer, β::Integer) = BetaBinomial(n, Float64(α), Float64(β))

@distr_support BetaBinomial 0 d.n
insupport(d::BetaBinomial, x::Real) = 0 <= x <= d.n

#### Conversions
function convert{T <: Real, S <: Real}(::Type{BetaBinomial{T}}, n::Int, α::S, β::S)
    BetaBinomial(n, T(α), T(β))
end
function convert{T <: Real, S <: Real}(::Type{BetaBinomial{T}}, d::BetaBinomial{S})
    BetaBinomial(d.n, T(d.α), T(d.β))
end

#### Parameters

ntrials(d::BetaBinomial) = d.n

params(d::BetaBinomial) = (d.n, d.α, d.β)
@inline partype{T<:Real}(d::BetaBinomial{T}) = T

#### Properties

mean(d::BetaBinomial) = (d.n * d.α) / (d.α + d.β)
function var(d::BetaBinomial)
    n, α, β = d.n, d.α, d.β
    numerator = n * α * β * (α + β + n)
    denominator = (α + β)^2 * (α + β + 1)
    return numerator / denominator
end

function skewness(d::BetaBinomial)
    n, α, β = d.n, d.α, d.β
    t1 = (α + β + 2n) * (β - α) / (α + β + 2)
    t2 = sqrt((1 + α +β) / (n * α * β * (n + α + β)))
    return t1 * t2
end

function kurtosis(d::BetaBinomial)
    n, α, β = d.n, d.α, d.β
    alpha_beta_sum = α + β
    alpha_beta_product = α * β
    numerator = ((alpha_beta_sum)^2) * (1 + alpha_beta_sum)
    denominator = (n * alpha_beta_product) * (alpha_beta_sum + 2) * (alpha_beta_sum + 3) * (alpha_beta_sum + n)
    left = numerator / denominator
    right = (alpha_beta_sum) * (alpha_beta_sum - 1 + 6n) + 3*alpha_beta_product * (n - 2) + 6n^2
    right -= (3*alpha_beta_product * n * (6 - n)) / alpha_beta_sum
    right -= (18*alpha_beta_product * n^2) / (alpha_beta_sum)^2
    return (left * right) - 3
end

function pdf(d::BetaBinomial, k::Int)
    n, α, β = d.n, d.α, d.β
    choose = Float64(binomial(n, k))
    numerator = beta(k + α, n - k + β)
    denominator = beta(α, β)
    return choose * (numerator / denominator)
end

function pdf(d::BetaBinomial)
    n, α, β = d.n, d.α, d.β
    k = 0:n
    binoms = Float64[binomial(n, i) for i in k]
    fixed_beta = beta(α, β)
    return binoms .* beta(k + α, n - k + β) / fixed_beta
end

entropy(d::BetaBinomial) = entropy(Categorical(pdf(d)))
median(d::BetaBinomial) = median(Categorical(pdf(d))) - 1
mode{T<:Real}(d::BetaBinomial{T}) = indmax(pdf(d)) - one(T)
modes(d::BetaBinomial) = [x - 1 for x in modes(Categorical(pdf(d)))]

quantile(d::BetaBinomial, p::Float64) = quantile(Categorical(pdf(d)), p) - 1
