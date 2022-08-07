using Random
"""
    InverseUniform(b⁻¹, a⁻¹)

A positive random variable `X` is inverse uniformly distributed with parameters `b⁻¹`, `a⁻¹`
if `X⁻¹` is `Uniform(a,b)` distributed.
```julia
InverseUniform(0.1,2)
```
"""
struct InverseUniform{T} <: ContinuousUnivariateDistribution
    binv::T
    ainv::T
    InverseUniform{T}(binv::T, ainv::T) where {T} = new{T}(binv,ainv)
end

function InverseUniform(binv::T, ainv::T; check_args::Bool=true) where {T} 
    @check_args InverseUniform (0 < binv < ainv)
    InverseUniform{T}(binv, ainv)
end
InverseUniform(binv, ainv; check_args=true) = InverseUniform(promote(binv, ainv)...; check_args=check_args)

Base.minimum(d::InverseUniform) = d.binv
Base.maximum(d::InverseUniform) = d.ainv

#### Parameters
params(d::InverseUniform) = (d.binv, d.ainv)
partype(::InverseUniform{T}) where {T<:Real} = T

#### Statistics
function mean(d::InverseUniform)
    a = inv(d.ainv)
    b = inv(d.binv)
    (log(b) - log(a)) / (b-a)
end
function var(d::InverseUniform)
    d.ainv*d.binv - mean(d)^2
end
function median(d::InverseUniform)
    a = inv(d.ainv)
    b = inv(d.binv)
    2 / (a + b)
end
function rand(rng::AbstractRNG, d::InverseUniform)
    a = inv(d.ainv)
    b = inv(d.binv)
    y = rand(rng) * (b - a) + a
    x = inv(y)
    return x
end

#### Evaluation
function pdf(d::InverseUniform, x::Real)
    a = inv(d.ainv)
    b = inv(d.binv)
    ret = inv(x^2 * (b-a))
    if insupport(d, x)
        ret
    elseif isnan(x)
        typeof(ret)(x)
    else
        zero(ret)
    end
end
logpdf(d::InverseUniform, x::Real) = log(pdf(d,x))
function cdf(d::InverseUniform, x::Real)
    a = inv(d.ainv)
    b = inv(d.binv)
    ret = (b - inv(x)) / (b-a)
    if x < d.binv
        zero(ret)
    elseif x < d.ainv
        ret
    elseif isnan(x)
        typeof(ret)(x)
    else
        one(ret)
    end
end
function quantile(d::InverseUniform, x::Real)
    a = inv(d.ainv)
    b = inv(d.binv)
    inv((a-b)*x + b)
end
mode(d::InverseUniform) = d.binv
