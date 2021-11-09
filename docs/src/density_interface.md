# Support for DensityInterface

`Distributions` supports [`DensityInterface`](https://github.com/JuliaMath/DensityInterface.jl) for distributions.

For *single* variate values `x`, `DensityInterface.logdensityof(d::Distribution, x)` is equivalent to `logpdf(d, x)` and `DensityInterface.densityof(d::Distribution, x)` is equivalent to `pdf(d, x)`.
