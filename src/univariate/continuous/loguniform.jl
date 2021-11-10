"""
    LogUniform(a,b)

A positive random variable `X` is log-uniformly with parameters `a` and `b` if the logarithm of `X` is `Uniform(log(a), log(b))`.
The *log uniform* distribution is also known as *reciprocal distribution*.
```julia
LogUniform(1,10)
```
External links

* [Log uniform distribution on Wikipedia](https://en.wikipedia.org/wiki/Reciprocal_distribution)
"""
struct LogUniform{T<:Real} <: ContinuousUnivariateDistribution
    a::T
    b::T
    LogUniform{T}(a::T, b::T) where {T <: Real} = new{T}(a, b)
end

function LogUniform(a::T, b::T; check_args=true) where {T <: Real}
    check_args && @check_args(LogUniform, 0 < a < b)
    LogUniform{T}(a, b)
end

LogUniform(a::Real, b::Real) = LogUniform(promote(a, b)...)

function convert(::Type{LogUniform{T}}, d::LogUniform) where {T}
    LogUniform(T(d.a), T(d.b))::LogUniform{T}
end
Base.minimum(d::LogUniform) = d.a
Base.maximum(d::LogUniform) = d.b

#### Parameters
params(d::LogUniform) = (d.a, d.b)
partype(::LogUniform{T}) where {T<:Real} = T

#### Statistics

function mean(d::LogUniform)
    a = d.a; b = d.b
    (b - a) / log(b/a)
end
function var(d::LogUniform)
    a, b = params(d)
    log_ba = log(b/a)
    (b^2 - a^2) / (2*log_ba) - ((b-a)/ log_ba)^2
end
mode(d::LogUniform)   = d.a
modes(d::LogUniform)  = partype(d)[]

#### Evaluation
function pdf(d::LogUniform, x::Real)
    a, b = params(d)
    res = inv(x * log(b / a))
    return insupport(d, x) ? res : zero(res)
end
function cdf(d::LogUniform, x::Real)
    a, b = params(d)
    _x = clamp(x, a, b)
    return log(_x / a) / log(b / a)
end
logpdf(d::LogUniform, x::Real) = log(pdf(d,x))

function quantile(d::LogUniform{T}, p::U) where {T,U<:Real}
    a,b = params(d)
    exp(p * log(b/a)) * a
end

truncated(d::LogUniform, lo, hi) = truncated_LogUniform(d, promote(lo, hi)...)
function truncated(d::LogUniform, lo::T, hi::T) where {T<:Integer}
    # this method is needed to fix ambiguities
    truncated_LogUniform(d, promote(lo, hi)...)
end
function truncated_LogUniform(d::LogUniform, lo::T, hi::T) where {T}
    a,b = params(d)
    LogUniform(max(a, lo), min(b, hi))
end
