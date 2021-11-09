struct LogUniform{T<:Real} <: ContinuousUnivariateDistribution
    a::T
    b::T
    LogUniform{T}(a::T, b::T) where {T <: Real} = new{T}(a, b)
end

function LogUniform(a::T, b::T; check_args=true) where {T <: Real}
    check_args && @check_args(LogUniform, 0 < a < b)
    LogUniform{T}(a, b)
end

function LogUniform(a::Real, b::Real)
    a1,b1 = promote(float(a), float(b))
    LogUniform(a1,b2)
end

function convert(::Type{LogUniform{T}}, d::LogUniform) where {T}
    LogUniform(T(d.a), T(d.b))::LogUniform{T}
end


#### Parameters
params(d::LogUniform) = (d.a, d.b)
partype(::LogUniform{T}) where {T<:Real} = T

#### Statistics

function mean(d::LogUniform)
    a = d.a; b = d.b
    (b - a) / log(b/a)
end
function var(d::LogUniform)
    a = d.a; b = d.b
    log_ba = log(b/a)
    (b^2 - a^2) / (2*log_ba) - ((b-a)/ log_ba)^2
end

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

#### Sampling
function rand(rng::AbstractRNG, d::LogUniform)
    u = Uniform(log(d.a), log(d.b))
    exp(rand(rng, u))
end
function rand!(rng::AbstractRNG, d::LogUniform, out::AbstractArray)
    u = Uniform(log(d.a), log(d.b))
    rand!(rng, u, out)
    out .= exp.(out)
    out
end
