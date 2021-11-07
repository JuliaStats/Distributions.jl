# Support for DensityInterface

`Distributions` supports [`DensityInterface`](https://github.com/JuliaMath/DensityInterface.jl) for distributions: `DensityInterface.logdensityof(d, x)` is equivalent to `logpdf(d, x)` and `DensityInterface.densityof(d::Distribution, x)` is equivalent to `pdf(d, x)` for *single* variate values `x`.

To get the log-density value at either a single variate value or at a set of variate values (implying a product distribution over `d`), use `logdensityof(IIDDensity(d), x)`, this is equivalent to `loglikelihood(d, x)`.

```@docs
IIDDensity
```
