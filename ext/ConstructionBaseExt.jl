module ConstructionBaseExt

using Distributions
using Distributions: Censored, GenericMvTDist, ProductDistribution, namedparams
import ConstructionBase

# Properties of a distribution are its parameters, not its fields (which may contain derived quantities)
# Distributions without `namedparams` (e.g. defined in other packages) fall back to their fields
function ConstructionBase.getproperties(d::Distribution)
    if hasmethod(namedparams, Tuple{typeof(d)})
        return namedparams(d)
    else
        return ConstructionBase.getfields(d)
    end
end

function ConstructionBase.setproperties(d::Distribution, patch::NamedTuple)
    props = ConstructionBase.getproperties(d)
    invalid = Base.structdiff(patch, props)
    if !isempty(invalid)
        throw(ArgumentError(lazy"$(keys(invalid)) are not properties of $(nameof(typeof(d))); its properties are $(keys(props))"))
    end
    return _constructorof(typeof(d))(values(merge(props, patch))...)
end

# Constructors from the fields, for types whose inner constructor computes derived fields
# As in the `constructorof` docstring, the derived fields must be consistent with the other fields
function _checkfields(d, fields...)
    if map(k -> getproperty(d, k), fieldnames(typeof(d))) != fields
        throw(ArgumentError(lazy"inconsistent fields of $(nameof(typeof(d)))"))
    end
    return d
end
ConstructionBase.constructorof(::Type{<:Dirichlet}) = (alpha, alpha0, lmnB) -> _checkfields(Dirichlet(alpha), alpha, alpha0, lmnB)
ConstructionBase.constructorof(::Type{<:DirichletMultinomial}) = (n, α, α0) -> _checkfields(DirichletMultinomial(n, α), n, α, α0)
ConstructionBase.constructorof(::Type{<:DiscreteUniform}) = (a, b, pv) -> _checkfields(DiscreteUniform(a, b), a, b, pv)
ConstructionBase.constructorof(::Type{<:GenericMvTDist}) = (df, dim, μ, Σ) -> _checkfields(GenericMvTDist(df, μ, Σ), df, dim, μ, Σ)
ConstructionBase.constructorof(::Type{<:InverseGamma}) = (invd, θ) -> _checkfields(InverseGamma(shape(invd), θ), invd, θ)
ConstructionBase.constructorof(::Type{<:NormalCanon}) = (η, λ, μ) -> _checkfields(NormalCanon(η, λ), η, λ, μ)
ConstructionBase.constructorof(::Type{<:NormalInverseGaussian}) = (μ, α, β, δ, γ) -> _checkfields(NormalInverseGaussian(μ, α, β, δ), μ, α, β, δ, γ)
# `pmf` is computed lazily
ConstructionBase.constructorof(::Type{<:PoissonBinomial}) = (p, pmf) -> pmf === nothing ? PoissonBinomial(p) : _checkfields(PoissonBinomial(p), p, pmf)
ConstructionBase.constructorof(::Type{<:ProductDistribution}) = (dists, size) -> _checkfields(ProductDistribution(dists), dists, size)
ConstructionBase.constructorof(::Type{<:Soliton}) = (K, M, δ, atol, degrees, CDF) -> _checkfields(Soliton(K, M, δ, atol), K, M, δ, atol, degrees, CDF)
ConstructionBase.constructorof(::Type{<:UnivariateGMM}) = (K, means, stds, prior) -> _checkfields(UnivariateGMM(means, stds, prior), K, means, stds, prior)
ConstructionBase.constructorof(::Type{<:VonMisesFisher}) = (μ, κ, logCκ) -> _checkfields(VonMisesFisher(μ, κ), μ, κ, logCκ)

# Constructors from the properties
_constructorof(T::Type) = ConstructionBase.constructorof(T)
# `Categorical` is an alias of `DiscreteNonParametric` with its own parameters
for D in (Categorical, Dirichlet, DirichletMultinomial, DiscreteUniform, GenericMvTDist, InverseGamma,
          NormalCanon, NormalInverseGaussian, PoissonBinomial, ProductDistribution, Soliton, UnivariateGMM,
          VonMisesFisher)
    @eval _constructorof(::Type{<:$D}) = $D
end
_constructorof(::Type{<:Truncated}) = truncated
_constructorof(::Type{<:Censored}) = censored

end # module
