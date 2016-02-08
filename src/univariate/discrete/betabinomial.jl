immutable BetaBinomial <: DiscreteUnivariateDistribution
    n::Int
    α::Float64
    β::Float64

    function BetaBinomial(n::Real, α::Real, β::Real)
        @check_args(BetaBinomial, n >= zero(n) && α >= zero(α) && β >= zero(β))
        new(n, α, β)
    end
end

@distr_support BetaBinomial 0 d.n
insupport(d::BetaBinomial, x::Real) = 0 <= x <= d.n

#### Parameters

ntrials(d::BetaBinomial) = d.n

params(d::BetaBinomial) = (d.n, d.α, d.β)

#### Properties

mean(d::BetaBinomial) = (d.n * d.α) / (d.α + d.β)
function var(d::BetaBinomial)
    n, α, β = d.n, d.α, d.β
    numerator = n * α * β * (α + β + n)
    denominator = (α + β)^2.0 * (α + β + 1.0)
    return numerator / denominator
end

function skewness(d::BetaBinomial)
    n, α, β = d.n, d.α, d.β
    t1 = (α + β + 2 * n) * (β - α) / (α + β + 2)
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
    right = (alpha_beta_sum) * (alpha_beta_sum - 1 + 6*n) + 3 * alpha_beta_product * (n - 2) + 6 * n^2
    right -= (3 * alpha_beta_product * n * (6 - n)) / alpha_beta_sum
    right -= (18 * alpha_beta_product * n^2) / (alpha_beta_sum)^2
    return (left * right) - 3
end

function pdf(d::BetaBinomial, k::Int)
    n, α, β = d.n, d.α, d.β
    @compat choose = Float64(binomial(n, k))
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
mode(d::BetaBinomial) = indmax(pdf(d)) - 1
modes(d::BetaBinomial) = [x - 1 for x in modes(Categorical(pdf(d)))]

quantile(d::BetaBinomial, p::Float64) = quantile(Categorical(pdf(d)), p) - 1
