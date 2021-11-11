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

convert(::Type{LogUniform{T}}, d::LogUniform) where {T<:Real} = LogUniform(T(d.a), T(d.b))
Base.minimum(d::LogUniform) = d.a
Base.maximum(d::LogUniform) = d.b

#### Parameters
params(d::LogUniform) = (d.a, d.b)
partype(::LogUniform{T}) where {T<:Real} = T

#### Statistics

function mean(d::LogUniform)
    a, b = params(d)
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
    x1, a, b = promote(x, params(d)...) # ensure e.g. pdf(LogUniform(1,2), 1f0)::Float32
    res = inv(x1 * log(b / a))
    return insupport(d, x1) ? res : zero(res)
end
function cdf(d::LogUniform, x::Real)
    x1, a, b = promote(x, params(d)...) # ensure e.g. cdf(LogUniform(1,2), 1f0)::Float32
    x1 = clamp(x1, a, b)
    return log(x1 / a) / log(b / a)
end
logpdf(d::LogUniform, x::Real) = log(pdf(d,x))

function quantile(d::LogUniform, p::Real)
    p1,a,b = promote(p, params(d)...) # ensure e.g. quantile(LogUniform(1,2), 1f0)::Float32
    exp(p1 * log(b/a)) * a
end
