"""
    KSOneSided <: ContinuousUnivariateDistribution

Distribution of the one-sided Kolmogorov-Smirnoff statistic.

# Constructors

    KSOneSided(n=)

Construct a `KSOneSided` distribution object of `n` observations.

# Details

Distribution of the one-sided Kolmogorov-Smirnov test statistic:

```math
D^+_n = \\sup_x (\\hat{F}_n(x) -F(x))
```
"""
struct KSOneSided <: ContinuousUnivariateDistribution
    n::Int
end

@kwdispatch KSOneSided()
@kwmethod KSOneSided(;n) = KSOneSided(n)

@distr_support KSOneSided 0.0 1.0


#### Evaluation

# formula of Birnbaum and Tingey (1951)
function ccdf(d::KSOneSided, x::Float64)
    if x >= 1
        return 0.0
    elseif x <= 0
        return 1.0
    end
    n = d.n
    s = 0.0
    for j = 0:floor(Int,n-n*x)
        p = x+j/n
        s += pdf(Binomial(n,p),j) / p
    end
    s*x
end

cdf(d::KSOneSided, x::Float64) = 1 - ccdf(d,x)
