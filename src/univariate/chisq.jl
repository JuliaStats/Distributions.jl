immutable Chisq <: ContinuousUnivariateDistribution
    df::Float64 # non-integer degrees of freedom are meaningful
    function Chisq(d::Real)
        d > zero(d) || error("df must be positive")
        new(float64(d))
    end
end

insupport(::Chisq, x::Real) = zero(x) <= x < Inf
insupport(::Type{Chisq}, x::Real) = zero(x) <= x < Inf


mean(d::Chisq) = d.df

mode(d::Chisq) = d.df > 2.0 ? d.df - 2.0 : 0.0
modes(d::Chisq) = [mode(d)]

var(d::Chisq) = 2.0 * d.df
skewness(d::Chisq) = sqrt(8.0 / d.df)
kurtosis(d::Chisq) = 12.0 / d.df


function entropy(d::Chisq)
    x = d.df / 2.0 + log(2.0) + lgamma(0.5*d.df)
    x + (1.0 - d.df / 2.0) * digamma(0.5*d.df)
end


pdf(d::Chisq, x::Real) = pdf(Gamma(0.5*d.df), x)
logpdf(d::Chisq, x::Real) = logpdf(Gamma(0.5*d.df), x)

cdf(d::Chisq, x::Real) = cdf(Gamma(0.5*d.df), x)
ccdf(d::Chisq, x::Real) = ccdf(Gamma(0.5*d.df), x)
logcdf(d::Chisq, x::Real) = logcdf(Gamma(0.5*d.df), x)
logccdf(d::Chisq, x::Real) = logccdf(Gamma(0.5*d.df), x)

quantile(d::Chisq, p::Real) = quantile(Gamma(0.5*d.df), p)
cquantile(d::Chisq, p::Real) = cquantile(Gamma(0.5*d.df), p)
invlogcdf(d::Chisq, lp::Real) = invlogcdf(Gamma(0.5*d.df), lp)
invlogccdf(d::Chisq, lp::Real) = invlogccdf(Gamma(0.5*d.df), lp)


function mgf(d::Chisq, t::Real)
    k = d.df
    (1.0 - 2.0 * t)^(-k / 2.0)
end

cf(d::Chisq, t::Real) = (1.0 - 2.0 * im * t)^(-0.5*d.df)


# rand - the distribution chi^2(df) is 2 * gamma(df / 2)
# for integer n, a chi^2(n) is the sum of n squared standard normals
function rand(d::Chisq)
    d.df == 1 ? randn()^2 : 2.0 * rand(Gamma(0.5*d.df))
end

function rand!(d::Chisq, A::Array{Float64})
    if d.df == 1
        for i in 1:length(A)
            A[i] = randn()^2
        end
        return A
    end
    if d.df >= 2
        dpar = 0.5*d.df - 1.0 / 3.0
    else
        error("require degrees of freedom df >= 2")
    end
    cpar = 1.0 / sqrt(9.0 * dpar)
    for i in 1:length(A)
        A[i] = 2.0 * randg2(dpar, cpar)
    end
    A
end

