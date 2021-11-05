@inline DensityInterface.hasdensity(::Distribution) = true

DensityInterface.logdensityof(d::Distribution, x) = loglikelihood(d, x)

# Don't specialize `DensityInterface.densityof(d::Distribution, x)`
# until something like `likelihood(d, x)` is available, `pdf(d, x)` can have
# different behavior.
