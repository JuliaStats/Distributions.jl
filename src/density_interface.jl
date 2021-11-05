@inline DensityInterface.hasdensity(::Distribution) = true

DensityInterface.logdensityof(d::Distribution, x) = logpdf(d, x)
