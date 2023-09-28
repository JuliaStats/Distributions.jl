"""
    MvLogitNormal{<:AbstractMvNormal}

The [multivariate logit-normal distribution](https://en.wikipedia.org/wiki/Logit-normal_distribution#Multivariate_generalization)
is a multivariate generalization of [`LogitNormal`](@ref) capable of handling correlations
between variables.

If ``\\mathbf{y} \\sim \\mathrm{MvNormal}(\\boldsymbol{\\mu}, \\boldsymbol{\\Sigma})`` is a
length ``d-1`` vector, then
```math
\\mathbf{x} = \\operatorname{softmax}\\left(\\begin{bmatrix}\\mathbf{y} \\\\ 0 \\end{bmatrix}\\right) \\sim \\mathrm{MvLogitNormal}(\\boldsymbol{\\mu}, \\boldsymbol{\\Sigma})
```
is a length ``d`` probability vector.

```julia
MvLogitNormal(μ, Σ)                 # MvLogitNormal with y ~ MvNormal(μ, Σ)
MvLogitNormal(MvNormal(μ, Σ))       # same as above
MvLogitNormal(MvNormalCanon(μ, J))  # MvLogitNormal with y ~ MvNormalCanon(μ, J)
```

# Fields

- `normal::AbstractMvNormal`: contains the ``d-1``-dimensional distribution of ``y``
"""
struct MvLogitNormal{D<:AbstractMvNormal} <: ContinuousMultivariateDistribution
    normal::D
    MvLogitNormal{D}(normal::D) where {D<:AbstractMvNormal} = new{D}(normal)
end
MvLogitNormal(d::AbstractMvNormal) = MvLogitNormal{typeof(d)}(d)
MvLogitNormal(args...) = MvLogitNormal(MvNormal(args...))

function Base.show(io::IO, d::MvLogitNormal; indent::String="  ")
    print(io, distrname(d))
    println(io, "(")
    normstr = strip(sprint(show, d.normal; context=IOContext(io)))
    normstr = replace(normstr, "\n" => "\n$indent")
    print(io, indent)
    println(io, normstr)
    println(io, ")")
end

# Conversions

function convert(::Type{MvLogitNormal{D}}, d::MvLogitNormal) where {D}
    return MvLogitNormal(convert(D, d.normal))
end
Base.convert(::Type{MvLogitNormal{D}}, d::MvLogitNormal{D}) where {D} = d

meanform(d::MvLogitNormal{<:MvNormalCanon}) = MvLogitNormal(meanform(d.normal))
canonform(d::MvLogitNormal{<:MvNormal}) = MvLogitNormal(canonform(d.normal))

# Properties

length(d::MvLogitNormal) = length(d.normal) + 1
Base.eltype(::Type{<:MvLogitNormal{D}}) where {D} = eltype(D)
Base.eltype(d::MvLogitNormal) = eltype(d.normal)
params(d::MvLogitNormal) = params(d.normal)
@inline partype(d::MvLogitNormal) = partype(d.normal)

location(d::MvLogitNormal) = mean(d.normal)
minimum(d::MvLogitNormal) = fill(zero(eltype(d)), length(d))
maximum(d::MvLogitNormal) = fill(oneunit(eltype(d)), length(d))

function insupport(d::MvLogitNormal, x::AbstractVector{<:Real})
    return length(d) == length(x) && all(≥(0), x) && sum(x) ≈ 1
end

# Evaluation

function _logpdf(d::MvLogitNormal, x::AbstractVector{<:Real})
    if !insupport(d, x)
        return oftype(logpdf(d.normal, _inv_softmax1(abs.(x))), -Inf)
    else
        return logpdf(d.normal, _inv_softmax1(x)) - sum(log, x)
    end
end

function gradlogpdf(d::MvLogitNormal, x::AbstractVector{<:Real})
    y = _inv_softmax1(x)
    ∂y = gradlogpdf(d.normal, y)
    ∂x = (vcat(∂y, -sum(∂y)) .- 1) ./ x
    return ∂x
end

# Statistics

kldivergence(p::MvLogitNormal, q::MvLogitNormal) = kldivergence(p.normal, q.normal)

# Sampling

function _rand!(rng::AbstractRNG, d::MvLogitNormal, x::AbstractVecOrMat{<:Real})
    y = @views _drop1(x)
    rand!(rng, d.normal, y)
    _softmax1!(x, y)
    return x
end

# Fitting

function fit_mle(::Type{MvLogitNormal{D}}, x::AbstractMatrix{<:Real}; kwargs...) where {D}
    y = similar(x, size(x, 1) - 1, size(x, 2))
    map(_inv_softmax1!, eachcol(y), eachcol(x))
    normal = fit_mle(D, y; kwargs...)
    return MvLogitNormal(normal)
end
function fit_mle(::Type{MvLogitNormal}, x::AbstractMatrix{<:Real}; kwargs...)
    return fit_mle(MvLogitNormal{MvNormal}, x; kwargs...)
end

# Utility

function _softmax1!(x::AbstractVector, y::AbstractVector)
    u = max(0, maximum(y))
    _drop1(x) .= exp.(y .- u)
    x[end] = exp(-u)
    LinearAlgebra.normalize!(x, 1)
    return x
end
function _softmax1!(x::AbstractMatrix, y::AbstractMatrix)
    map(_softmax1!, eachcol(x), eachcol(y))
    return x
end

_drop1(x::AbstractVector) = @views x[firstindex(x, 1):(end - 1)]
_drop1(x::AbstractMatrix) = @views x[firstindex(x, 1):(end - 1), :]

_last1(x::AbstractVector) = x[end]
_last1(x::AbstractMatrix) = @views x[end, :]

function _inv_softmax1!(y::AbstractVecOrMat, x::AbstractVecOrMat)
    x₋ = _drop1(x)
    xd = _last1(x)
    @. y = log(x₋) - log(xd)
    return y
end
function _inv_softmax1(x::AbstractVecOrMat)
    y = similar(_drop1(x))
    _inv_softmax1!(y, x)
    return y
end
