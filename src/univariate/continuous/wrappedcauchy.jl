"""
    WrappedCauchy(r)

The Wrapped Cauchy distribution with scale factor `r` has probability density function

```math
f(x; r) = \\frac{1-r^2}{2\\pi(1+r^2-2r\\cos(x-\\mu))}, \\quad x \\in [-\\pi, \\pi].
```

```julia
WrappedCauchy(μ,r)   # Wrapped Cauchy distribution centered on μ with scale factor r

WrappedCauchy(r)   # Wrapped Cauchy distribution centered on 0 with scale factor r

params(d)       # Get the location and scale parameters, i.e. (μ, r)
```

External links

* [Wrapped Cauchy distribution on Wikipedia](https://en.wikipedia.org/wiki/Wrapped_Cauchy_distribution)
"""
struct WrappedCauchy{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    r::T
    WrappedCauchy{T}(μ::T, r::T) where {T <: Real} = new{T}(μ, r)
end


function WrappedCauchy(μ::T, r::T; check_args::Bool=true) where {T <: Real}
    @check_args WrappedCauchy (μ, -π < μ < π) (r, zero(r) < r < one(r))

    return WrappedCauchy{T}(μ, r)
end

WrappedCauchy(μ::Real, r::Real; check_args::Bool=true) = WrappedCauchy(promote(μ, r)...; check_args=check_args)
function WrappedCauchy(r::Real; check_args::Bool=true)
    @check_args WrappedCauchy (r, zero(r) < r < one(r))
    return WrappedCauchy(zero(r), r; check_args=false)
end

@distr_support WrappedCauchy -oftype(d.r, π) oftype(d.r, π)


params(d::WrappedCauchy) = (d.μ, d.r)
partype(::WrappedCauchy{T}) where {T} = T

location(d::WrappedCauchy) = d.μ
scale(d::WrappedCauchy) = d.r

#### Statistics

mean(d::WrappedCauchy) = d.μ

var(d::WrappedCauchy) = one(d.r) - d.r

skewness(d::WrappedCauchy) = zero(d.r)

median(d::WrappedCauchy) = d.μ

mode(d::WrappedCauchy) = d.μ

entropy(d::WrappedCauchy) = log2π + log1p(-d.r^2)


cf(d::WrappedCauchy, t::Real) = cis(t * d.μ - abs(t) * log(d.r) * im)

#### Evaluation

function pdf(d::WrappedCauchy, x::Real)
    μ, r = params(d)
    res = inv2π * ((1 - r^2) / (1 + r^2 - 2 * r * cos(x - μ)))
    return insupport(d, x) ? res : zero(res)
end

function logpdf(d::WrappedCauchy, x::Real)
    μ, r = params(d)
    res = - log1p(2 * r * (r - cos(x-μ)) / (1 - r^2)) - log2π
    return insupport(d, x) ? res : oftype(res, -Inf)
end

function cdf(d::WrappedCauchy, x::Real)
    μ, r = params(d)
    min_d, max_d = extrema(d)
    c = (one(r) + r) / (one(r) - r)
    res = (atan(c * tan((x - μ) / 2)) - atan(c * cot(μ / 2))) / π
    return if x ≤ min_d
        zero(res)
    elseif x ≥ max_d
        one(res)
    elseif res < 0 # if mod2pi(x - μ) > π
        1 + res
    else
        res
    end
end

#### Sampling

function rand(rng::AbstractRNG, d::WrappedCauchy)
    return mod2pi(d.μ + log(d.r) * tan(π * (rand(rng) - 0.5))) - π

end

#### Fitting

function _WC_mle_update(x::AbstractArray{T}, μ1, μ2) where {T <: Real}
    n = length(x)
    sc = zero(T)
    sw = zero(T)
    ss = zero(T)
    for i = 1:n
        @inbounds xi = x[i]
        wi = 1/(1 - μ1*cos(xi) - μ2*sin(xi))
        ss += wi*sin(xi)
        sc += wi*cos(xi)
        sw += wi
    end
    return sc / sw, ss / sw
end


function fit_mle(::Type{<:WrappedCauchy},
    x::AbstractArray{T};
    mu0::Float64=NaN, r0::Float64=NaN, maxiter::Int=1000, tol::Float64=1e-16
) where {T <: Real}

    μ::Float64 = isnan(mu0) ? angle(mean(exp.(im*x))) : mu0
    r::Float64 = isnan(mu0) ? 0.5 : r0
    converged = false

    μ1 = 2*r*cos(μ)/(1+r^2)
    μ2 = 2*r*sin(μ)/(1+r^2)

    t = 0
    while !converged && t < maxiter
        t += 1
        μ1_old, μ2_old = μ1, μ2
        μ1, μ2 = _WC_mle_update(x, μ1, μ2)
        converged = (abs(μ1 - μ1_old) <= tol && abs(μ2 - μ2_old) <= tol)
    end
    μ = mod2pi(atan(μ2,μ1)) - π
    r = (1 - sqrt(1-μ1^2-μ2^2))/sqrt(μ1^2+μ2^2)

    WrappedCauchy{T}(μ, r)
end