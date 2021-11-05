@inline DensityInterface.hasdensity(::Distribution) = true

DensityInterface.logdensityof(d::Distribution, x) = loglikelihood(d, x)

DensityInterface.densityof(d::Distribution, x) = pdf(d, x)
