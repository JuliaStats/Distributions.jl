doc"""
    VonMises(μ, κ)

The *von Mises distribution* with mean `μ` and concentration `κ` has probability density function

$f(x; \mu, \kappa) = \frac{1}{2 \pi I_0(\kappa)} \exp \left( \kappa \cos (x - \mu) \right)$

```julia
VonMises()       # von Mises distribution with zero mean and unit concentration
VonMises(κ)      # von Mises distribution with zero mean and concentration κ
VonMises(μ, κ)   # von Mises distribution with mean μ and concentration κ
```

External links

* [von Mises distribution on Wikipedia](http://en.wikipedia.org/wiki/Von_Mises_distribution)

"""
immutable VonMises <: ContinuousUnivariateDistribution
    μ::Float64      # mean
    κ::Float64      # concentration
    I0κ::Float64    # I0(κ), where I0 is the modified Bessel function of order 0

    function VonMises(μ::Real, κ::Real)
        @check_args(VonMises, κ > zero(κ))
        new(μ, κ, besseli(0, κ))
    end
    VonMises(κ::Real) = VonMises(0.0, κ)
    VonMises() = new(0.0, 1.0)
end

show(io::IO, d::VonMises) = show(io, d, (:μ, :κ))

@distr_support VonMises d.μ - π d.μ + π


#### Parameters

params(d::VonMises) = (d.μ, d.κ)


#### Statistics

mean(d::VonMises) = d.μ
median(d::VonMises) = d.μ
mode(d::VonMises) = d.μ
circvar(d::VonMises) = 1.0 - besseli(1, d.κ) / d.I0κ
entropy(d::VonMises) = log(twoπ * d.I0κ) - d.κ * (besseli(1, d.κ) / d.I0κ)

cf(d::VonMises, t::Real) = (besseli(abs(t), d.κ) / d.I0κ) * cis(t * d.μ)


#### Evaluation

pdf(d::VonMises, x::Real) = 
    float64(x) - d.μ <= -π ? 0.0 : ( float64(x) - d.μ >= π ? 0.0 : exp(d.κ * cos(float64(x) - d.μ)) / (twoπ * d.I0κ) )
logpdf(d::VonMises, x::Real) = 
    float64(x) - d.μ <= -π ? -Inf : ( float64(x) - d.μ >= π ? -Inf : d.κ * cos(float64(x) - d.μ) - log(d.I0κ) - log2π )

cdf(d::VonMises, x::Real) = 
    float64(x) - d.μ <= -π ? 0.0 : ( float64(x) - d.μ >= π ? 1.0 :  _vmcdf(d.κ, d.I0κ, float64(x) - d.μ, 1.0e-15) )

function _vmcdf(κ::Float64, I0κ::Float64, x::Float64, tol::Float64)
    j = 1
    cj = besseli(j, κ) / j
    s = cj * sin(j * x)
    while abs(cj) > tol
        j += 1
        cj = besseli(j, κ) / j
        s += cj * sin(j * x)
    end
    return (x + 2.0 * s / I0κ) / twoπ + 0.5
end


#### Sampling

sampler(d::VonMises) = VonMisesSampler(d.μ, d.κ)
