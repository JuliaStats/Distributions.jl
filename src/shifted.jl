immutable Shifted{D<:UnivariateDistribution, S<:ValueSupport, T} <: UnivariateDistribution{S}
    unshifted::D        # the original distribution (unshifted)
    offset::T           # the shift offset
end

### Constructors

function Shifted(d::UnivariateDistribution, offset::Real)
    Shifted{typeof(d),value_support(typeof(d)), typeof(offset)}(d, offset)
end

### range and support

islowerbounded(d::Shifted) = islowerbounded(d.unshifted)
isupperbounded(d::Shifted) = isupperbounded(d.unshifted)

minimum(d::Shifted) = minimum(d.unshifted)+d.offset
maximum(d::Shifted) = maximum(d.unshifted)+d.offset

@compat insupport{D<:UnivariateDistribution}(d::Shifted{D,Union{Discrete,Continuous}}, x::Real) = insupport(d.unshifted, x-d.offset)


### evaluation

pdf(d::Shifted, x::Float64) = pdf(d.unshifted, x-d.offset)

logpdf(d::Shifted, x::Float64) = logpdf(d.unshifted, x-d.offset)

cdf(d::Shifted, x::Float64) = cdf(d.unshifted, x-d.offset)

logcdf(d::Shifted, x::Float64) = logcdf(d.unshifted, x-d.offset)

ccdf(d::Shifted, x::Float64) = ccdf(d.unshifted, x-d.offset)

logccdf(d::Shifted, x::Float64) = logccdf(d.unshifted, x-d.offset)


quantile(d::Shifted, p::Float64) = quantile(d.unshifted, p) + s

pdf(d::Shifted, x::Int) = pdf(d.unshifted, x-d.offset)

logpdf(d::Shifted, x::Int) = logpdf(d.unshifted, x-d.offset)

cdf(d::Shifted, x::Int) = cdf(d.unshifted, x-d.offset)

logcdf(d::Shifted, x::Int) = logcdf(d.unshifted, x-d.offset)

ccdf(d::Shifted, x::Int) = ccdf(d.unshifted, x-d.offset)

logccdf(d::Shifted, x::Int) = logccdf(d.unshifted, x-d.offset)

rand(d::Shifted) = rand(d.unshifted) + d.offset


## show

function show(io::IO, d::Shifted)
    print(io, "Shifted(")
    d0 = d.unshifted
    uml, namevals = _use_multline_show(d0)
    uml ? show_multline(io, d0, namevals) :
          show_oneline(io, d0, namevals)
    print(io, ", offset=$(d.offset))")
    uml && println(io)
end

_use_multline_show(d::Shifted) = _use_multline_show(d.unshifted)
