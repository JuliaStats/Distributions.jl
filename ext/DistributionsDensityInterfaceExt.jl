module DistributionsDensityInterfaceExt

using Distributions
import DensityInterface

@inline DensityInterface.DensityKind(::Distribution) = DensityInterface.HasDensity()

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

end # module
