# Support for DensityInterface

`Distributions` supports [`DensityInterface`](https://github.com/JuliaMath/DensityInterface.jl) for distributions.

For *single* variate values `x`, `DensityInterface.logdensityof(d::Distribution, x)` is equivalent to `logpdf(d, x)` and `DensityInterface.densityof(d::Distribution, x)` is equivalent to `pdf(d, x)`.

You can use
```julia
DensityInterface.logdensityof(Distributions.IIDDensity(d), x)
```
and
```julia
DensityInterface.densityof(Distributions.IIDDensity(d), x)
```
to evaluate the log-density and density of variate value(s) `x` that are assumed to be identically and independently
distributed according to distribution `d`. Here `x` can be either a single variate value or at a set of variate values.
`DensityInferface.logdensityof(Distributions.IIDDensity(d), x)` falls back to `loglikelihood(d, x)`.

```@docs
Distributions.IIDDensity
```
