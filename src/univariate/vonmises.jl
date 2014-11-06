# von Mises distribution
#
#  Implemented based on Wikipedia
#

immutable VonMises <: ContinuousUnivariateDistribution
    μ::Float64      # mean
    κ::Float64      # concentration
    I0κ::Float64    # I0(κ), where I0 is the modified Bessel function of order 0

    function VonMises(μ::Real, κ::Real)
        κ > zero(κ) || error("kappa must be positive")
        new(float64(μ), float64(κ), besseli(0, κ))
    end

    VonMises(κ::Real) = VonMises(0.0, float64(κ))
    VonMises() = VonMises(0.0, 1.0)
end

show(io::IO, d::VonMises) = show(io, d, (:μ, :κ))

### Properties

minimum(d::VonMises) = d.μ - π
maximum(d::VonMises) = d.μ + π
islowerbounded(d::VonMises) = true
isupperbounded(d::VonMises) = true
insupport(d::VonMises, x::Real) = -π <= (x - d.μ) <= π

mean(d::VonMises) = d.μ
median(d::VonMises) = d.μ
mode(d::VonMises) = d.μ
circvar(d::VonMises) = 1.0 - besseli(1, d.κ) / I0κ
entropy(d::VonMises) = log(twoπ * d.I0κ) - d.κ * (besseli(1, d.κ) / d.I0κ)

cf(d::VonMises, t::Real) = (besseli(abs(t), d.k) / I0κ) * exp(im * t * d.μ)


### Functions

pdf(d::VonMises, x::Real) = exp(d.κ * cos(float64(x) - d.μ)) / (twoπ * d.I0κ)
logpdf(d::VonMises, x::Real) = d.κ * cos(float64(x) - d.μ) - log(d.I0κ) - log2π

cdf(d::VonMises, x::Real) = _vmcdf(d.κ, d.I0κ, float64(x) - d.μ, 1.0e-15)

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

## Sampling
sampler(d::VonMises) = VonMisesSampler(d.μ, d.κ)
