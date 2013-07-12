immutable Exponential <: ContinuousUnivariateDistribution
    scale::Float64 # note: scale not rate
    function Exponential(sc::Real)
        if sc > 0.0
            new(float64(sc))
        else
            error("scale must be positive")
        end
    end
end

Exponential() = Exponential(1.0)

cdf(d::Exponential, q::Real) = q <= 0.0 ? 0.0 : -expm1(-q / d.scale)

function logcdf(d::Exponential, q::Real)
    if q <= 0.0
        return -Inf
    else
        qs = -q / d.scale
        if qs > log(0.5)
            return log(-expm1(qs))
        else
            return log1p(-exp(qs))
        end
    end
end

function ccdf(d::Exponential, q::Real)
    q <= 0.0 ? 1.0 : exp(-q / d.scale)
end

function logccdf(d::Exponential, q::Real)
    q <= 0.0 ? 0.0 : -q / d.scale
end

function invlogcdf(d::Exponential, lp::Real)
    if lp <= 0.0
        -d.scale * (lp > log(0.5) ? log(-expm1(lp)) : log1p(-exp(lp)))
    else
        return NaN
    end
end

function invlogccdf(d::Exponential, lp::Real)
    lp <= 0.0 ? -d.scale * lp : NaN
end

entropy(d::Exponential) = 1.0 - log(1.0 / d.scale)

insupport(d::Exponential, x::Number) = isreal(x) && isfinite(x) && 0.0 <= x

kurtosis(d::Exponential) = 6.0

mean(d::Exponential) = d.scale

median(d::Exponential) = d.scale * log(2.0)

function mgf(d::Exponential, t::Real)
    s = d.scale
    return (1.0 - t * s)^(-1)
end

function cf(d::Exponential, t::Real)
    s = d.scale
    return (1.0 - t * im * s)^(-1)
end

modes(d::Exponential) = [0.0]

function pdf(d::Exponential, x::Real)
    x < 0.0 ? 0.0 : exp(-x / d.scale) / d.scale
end

function logpdf(d::Exponential, x::Real)
    x < 0.0 ? -Inf : -x / d.scale - log(d.scale)
end

function quantile(d::Exponential, p::Real)
    0.0 <= p <= 1.0 ? -d.scale * log1p(-p) : NaN
end

function cquantile(d::Exponential, p::Real)
    0.0 <= p <= 1.0 ? -d.scale * log(p) : NaN
end

rand(d::Exponential) = d.scale * Random.randmtzig_exprnd()

function rand!(d::Exponential, A::Array{Float64})
    Random.randmtzig_fill_exprnd!(A)
    for i in 1:length(A)
        A[i] *= d.scale
    end
    return A
end

skewness(d::Exponential) = 2.0

var(d::Exponential) = d.scale * d.scale

function fit_mle(::Type{Exponential}, x::Array)
    for i in 1:length(x)
        if !insupport(Exponential(), x[i])
            error("Exponential observations must be non-negative values")
        end
    end
    return Exponential(mean(x))
end

fit(::Type{Exponential}, x::Array) = fit_mle(Exponential, x)


