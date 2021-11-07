@inline DensityInterface.hasdensity(::Distribution) = true

DensityInterface.logdensityof(d::Distribution, x) = loglikelihood(d, x)


DensityInterface.logdensityof(d::Distribution{ArrayLikeVariate{N}}, x::AbstractArray{<:Real,N}) where N = loglikelihood(d, x)

function DensityInterface.logdensityof(d::Distribution{ArrayLikeVariate{N}}, x::AbstractArray{<:Real}) where N
    throw(ArgumentError("logdensityof doesn't support multiple samples in a single function call"))
end

function DensityInterface.logdensityof(d::Distribution{ArrayLikeVariate{N}}, x::AbstractVector{<:AbstractArray{<:Real}}) where N
    throw(ArgumentError("logdensityof doesn't support multiple samples in a single function call"))
end


# Don't specialize `DensityInterface.densityof(d::Distribution, x)`
# until something like `likelihood(d, x)` is available, `pdf(d, x)` can have
# different behavior.
