@inline DensityInterface.hasdensity(::Distribution) = true

for (di_func, d_func) in ((:logdensityof, :logpdf), (:densityof, :pdf))
    @eval begin
        DensityInterface.$di_func(d::Distribution, x) = $d_func(d, x)

        function DensityInterface.$di_func(d::UnivariateDistribution, x::AbstractArray)
            throw(ArgumentError("$(DensityInterface.$di_func) doesn't support multiple samples as an argument"))
        end

        function DensityInterface.$di_func(d::MultivariateDistribution, x::AbstractMatrix)
            throw(ArgumentError("$(DensityInterface.$di_func) doesn't support multiple samples as an argument"))
        end

        function DensityInterface.$di_func(d::MatrixDistribution, x::AbstractArray{<:AbstractMatrix{<:Real}})
            throw(ArgumentError("$(DensityInterface.$di_func) doesn't support multiple samples as an argument"))
        end
    end
end


"""
    IIDDensity(distribution::Distribution)

Represents the probability density of an implicit product distribution of
variates that are identically and independently distributed according to
`distribution`.

Use `DensityInterface.logdensityof(d, x)` to compute the logarithmic density
value at `x`. `x` may be a single variate of `d` or a whole set of variates
of `d`.

If `x` is a single variate of `d`, the density is the PDF of `d`.

If `x` is a set of variates (given as a higher-dimensional array or
and array of arrays), the density is the PDF of an implicit product
distribution over `d`, the size of the product is implied by the size of
the set.

`DensityInterface.logdensityof(d, x)` is equivalent to `loglikelihood(d, x)`.
"""
struct IIDDensity{D<:Distribution}
    distribution::D
end

@inline DensityInterface.hasdensity(d::IIDDensity) = true

DensityInterface.logdensityof(d::IIDDensity, x) = loglikelihood(d.distribution, x)
