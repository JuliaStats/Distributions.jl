"""
    BetaBinomial(n,α,β)

A *Beta-binomial distribution* is the compound distribution of the [`Binomial`](@ref) distribution where the probability of success `p` is distributed according to the [`Beta`](@ref). It has three parameters: `n`, the number of trials and two shape parameters `α`, `β`

```math
P(X = k) = {n \\choose k} B(k + \\alpha, n - k + \\beta) / B(\\alpha, \\beta),  \\quad \\text{ for } k = 0,1,2, \\ldots, n.
```

```julia
BetaBinomial(n, a, b)      # BetaBinomial distribution with n trials and shape parameters a, b

params(d)       # Get the parameters, i.e. (n, a, b)
ntrials(d)      # Get the number of trials, i.e. n
```

External links:

* [Beta-binomial distribution on Wikipedia](https://en.wikipedia.org/wiki/Beta-binomial_distribution)
"""
struct BetaBinomial{T<:Real} <: DiscreteUnivariateDistribution
    n::Int
    α::T
    β::T

    BetaBinomial{T}(n::Integer, α::T, β::T) where {T <: Real} = new{T}(n, α, β)
end

function BetaBinomial(n::Integer, α::T, β::T; check_args=true) where {T <: Real}
    check_args && @check_args(BetaBinomial, n >= zero(n) && α >= zero(α) && β >= zero(β))
    return BetaBinomial{T}(n, α, β)
end

BetaBinomial(n::Integer, α::Real, β::Real) = BetaBinomial(n, promote(α, β)...)
BetaBinomial(n::Integer, α::Integer, β::Integer) = BetaBinomial(n, float(α), float(β))

@distr_support BetaBinomial 0 d.n
insupport(d::BetaBinomial, x::Real) = 0 <= x <= d.n

#### Conversions
function convert(::Type{BetaBinomial{T}}, n::Int, α::S, β::S) where {T <: Real, S <: Real}
    BetaBinomial(n, T(α), T(β))
end
function convert(::Type{BetaBinomial{T}}, d::BetaBinomial{S}) where {T <: Real, S <: Real}
    BetaBinomial(d.n, T(d.α), T(d.β), check_args=false)
end

#### Parameters

ntrials(d::BetaBinomial) = d.n

params(d::BetaBinomial) = (d.n, d.α, d.β)
partype(::BetaBinomial{T}) where {T} = T

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

function pdf(d::BetaBinomial{T}, k::Int) where T
    n, α, β = d.n, d.α, d.β
    insupport(d, k) || return zero(T)
    chooseinv = (n + 1) * beta(k + 1, n - k + 1)
    numerator = beta(k + α, n - k + β)
    denominator = beta(α, β)
    return numerator / (denominator * chooseinv)
end

function logpdf(d::BetaBinomial{T}, k::Int) where T
    n, α, β = d.n, d.α, d.β
    logbinom = - log1p(n) - logbeta(k + 1, n - k + 1)
    lognum   = logbeta(k + α, n - k + β)
    logdenom = logbeta(α, β)
    logbinom + lognum - logdenom
end

entropy(d::BetaBinomial) = entropy(Categorical(pdf.(Ref(d),support(d))))
median(d::BetaBinomial) = median(Categorical(pdf.(Ref(d),support(d)))) - 1
mode(d::BetaBinomial) = argmax(pdf.(Ref(d),support(d))) - 1
modes(d::BetaBinomial) = modes(Categorical(pdf.(Ref(d),support(d)))) .- 1

quantile(d::BetaBinomial, p::Float64) = quantile(Categorical(pdf.(Ref(d), support(d))), p) - 1

#### Sampling

rand(rng::AbstractRNG, d::BetaBinomial) =
    rand(rng, Binomial(d.n, rand(rng, Beta(d.α, d.β))))
