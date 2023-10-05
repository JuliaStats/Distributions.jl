"""
    WrappedCauchy(c)

The Wrapped Cauchy distribution with concentration parameter `c` has probability density function

```math
f(x; c, \\mu) = \\frac{1-c^2}{2\\pi(1+c^2-2c\\cos(x-\\mu))}, \\quad x \\in [0, 2\\pi].
```

```julia
WrappedCauchy(μ,c)   # Wrapped Cauchy distribution centered on μ with concentration parameter c

WrappedCauchy(c)   # Wrapped Cauchy distribution centered on 0 with concentration parameter c

params(d)       # Get the location and scale parameters, i.e. (μ, r)
```

External links

* [Wrapped Cauchy distribution on Wikipedia](https://en.wikipedia.org/wiki/Wrapped_Cauchy_distribution)
"""
struct WrappedCauchy{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    c::T
    WrappedCauchy{T}(μ::T, c::T) where {T <: Real} = new{T}(μ, c)
end


function WrappedCauchy(μ::T, c::T; check_args::Bool=true) where {T <: Real}
    @check_args WrappedCauchy (μ, 0 ≤ μ ≤ 2π) (c, zero(c) < c < one(c))
    return WrappedCauchy{T}(μ, c)
end

WrappedCauchy(μ::Real, c::Real; check_args::Bool=true) = WrappedCauchy(promote(μ, c)...; check_args=check_args)

function WrappedCauchy(c::Real; check_args::Bool=true)
    @check_args WrappedCauchy (c, zero(c) < c < one(c))
    return WrappedCauchy(zero(c), c; check_args=false)
end

@distr_support WrappedCauchy 0 oftype(d.μ, 2π)


params(d::WrappedCauchy) = (d.μ, d.c)
partype(::WrappedCauchy{T}) where {T} = T

location(d::WrappedCauchy) = d.μ
scale(d::WrappedCauchy) = d.c

#### Statistics

mean(d::WrappedCauchy) = d.μ

var(d::WrappedCauchy) = one(d.c) - d.c

skewness(d::WrappedCauchy) = zero(d.c)

median(d::WrappedCauchy) = d.μ

mode(d::WrappedCauchy) = d.μ

entropy(d::WrappedCauchy) = log2π + log1p(-d.c^2)


cf(d::WrappedCauchy, t::Real) = cis(t * d.μ - abs(t) * log(d.c) * im)

#### Evaluation

function pdf(d::WrappedCauchy, x::Real)
    μ, c = params(d)
    res = inv2π * ((1 - c^2) / (1 + c^2 - 2 * c * cos(x - μ)))
    return insupport(d, x) ? res : zero(res)
end

function logpdf(d::WrappedCauchy, x::Real)
    μ, c = params(d)
    res = - log1p(2 * c * (c - cos(x-μ)) / (1 - c^2)) - log2π
    return insupport(d, x) ? res : oftype(res, -Inf)
end

function cdf(d::WrappedCauchy, x::Real)
    μ, c = params(d)
    min_d, max_d = extrema(d)
    a = (one(c) + c) / (one(c) - c)
    res = (atan(a * tan((x - μ) / 2)) - atan(a * cot((μ+π) / 2))) / π
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
    return mod2pi(d.μ + log(d.c) * tan(π * (rand(rng) - 0.5)))
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

function fit(::Type{<:WrappedCauchy}, x::AbstractArray{T}) where {T <: Real}
    n = length(x)
    sn = mean(sin.(x))
    cs = mean(cos.(x))
    μ = mod2pi(atan(sn, cs))
    c2_bar = sn^2 + cs^2
    c = sqrt(n / (n - one(T)) * (c2_bar - one(T) / n))
    return WrappedCauchy{T}(μ, c)
end


function fit_mle(::Type{<:WrappedCauchy},
    x::AbstractArray{T};
    mu0::Float64=NaN, c0::Float64=NaN, maxiter::Int=1000, tol::Float64=1e-16
) where {T <: Real}
    est0 = fit(WrappedCauchy, x)
    μ::Float64 = isnan(mu0) ? est0.μ : mu0
    c::Float64 = isnan(c0) ? est0.c : r0
    converged = false

    # reparameterize
    μ1 = 2 * c * cos(μ) / (one(T) + c^2)
    μ2 = 2 * c * sin(μ) / (one(T) + c^2)

    t = 0
    while !converged && t < maxiter
        t += 1
        μ1_old, μ2_old = μ1, μ2
        μ1, μ2 = _WC_mle_update(x, μ1, μ2)
        converged = (abs(μ1 - μ1_old) <= tol && abs(μ2 - μ2_old) <= tol)
    end
    μ = mod2pi(atan(μ2, μ1))
    c = (one(T) - sqrt(one(T) - μ1^2 - μ2^2)) / sqrt(μ1^2 + μ2^2)

    WrappedCauchy{T}(μ, c)
end