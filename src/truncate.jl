abstract TruncatedContinuousUnivariateDistribution <: ContinuousUnivariateDistribution
abstract TruncatedDiscreteUnivariateDistribution <: DiscreteUnivariateDistribution
typealias TruncatedUnivariateDistribution Union(TruncatedContinuousUnivariateDistribution, TruncatedDiscreteUnivariateDistribution)

macro truncate(dname::Any)
    new_dname = esc(symbol(string("Truncated", string(dname))))
    # TODO: Are we not supposed to run eval() in a macro?
    if eval(dname) <: ContinuousUnivariateDistribution
        dtype = esc(TruncatedContinuousUnivariateDistribution)
    else
        dtype = esc(TruncatedDiscreteUnivariateDistribution)
    end
    dname = esc(dname)
    quote
        immutable $new_dname <: $dtype
            untruncated::$dname
            lower::Float64
            upper::Float64
            nc::Float64 # Normalization constant
            function ($new_dname)(d::$dname, l::Real, u::Real, nc::Real)
                if l >= u
                    error("upper must be > lower")
                end
                new(d, float64(l), float64(u), float64(nc))
            end
        end
        function ($new_dname)(d::$dname, l::Real, u::Real)
            return ($new_dname)(d, l, u, cdf(d, u) - cdf(d, l))
        end
    end
end

function insupport(d::TruncatedUnivariateDistribution, x::Number)
    return x >= d.lower && x <= d.upper && insupport(d.untruncated, x)
end

function pdf(d::TruncatedUnivariateDistribution, x::Real)
    if !insupport(d, x)
        return 0.0
    else
        return pdf(d.untruncated, x) / d.nc
    end
end

function logpdf(d::TruncatedUnivariateDistribution, x::Real)
    if !insupport(d, x)
        return -Inf
    else
        return logpdf(d.untruncated, x) - log(d.nc)
    end
end

function cdf(d::TruncatedUnivariateDistribution, x::Real)
    if x < d.lower
        return 0.0
    elseif x > d.upper
        return 1.0
    else
        return (cdf(d.untruncated, x) - cdf(d.untruncated, d.lower)) / d.nc
    end
end

function quantile(d::TruncatedUnivariateDistribution, p::Real)
    top = cdf(d.untruncated, d.upper)
    bottom = cdf(d.untruncated, d.lower)
    return quantile(d.untruncated, bottom + p * (top - bottom))
end

function rand(d::TruncatedUnivariateDistribution)
    while true
        r = rand(d.untruncated)
        if d.lower <= r <= d.upper
            return r
        end
    end
end
