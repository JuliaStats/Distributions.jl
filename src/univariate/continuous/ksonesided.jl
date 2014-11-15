# Distribution of the one-sided Kolmogorov-Smirnov test statistic:
#    D^+_n = \sup_x (\hat{F}_n(x) -F(x))
immutable KSOneSided <: ContinuousUnivariateDistribution
    n::Int
end

@continuous_distr_support KSOneSided 0.0 1.0

# formula of Birnbaum and Tingey (1951)
function ccdf(d::KSOneSided,x::Real)
    if x >= 1.0
        return 0.0
    elseif x <= 0.0
        return 1.0
    end
    n = d.n
    s = 0.0
    for j = 0:ifloor(n-n*x)
        p = x+j/n
        s += pdf(Binomial(n,p),j) / p
    end
    s*x
end
cdf(d::KSOneSided,x::Real) = 1.0 - ccdf(d,x)

