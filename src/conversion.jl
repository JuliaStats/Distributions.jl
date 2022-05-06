
# Discrete univariate

convert(::Type{Binomial}, d::Bernoulli) = Binomial(1, d.p)

# to NegativeBinomial
function convert(::Type{NegativeBinomial{T}}, d::NegativeBinomialLocation) where {T<:Real}
    NegativeBinomial{T}(T(inv(d.ϕ)), T(inv(d.μ * d.ϕ + 1)))
end
function convert(::Type{NegativeBinomial{T}}, d::NegativeBinomialLogLocation) where {T<:Real}
    NegativeBinomial{T}(T(inv(d.ϕ)), T(inv(exp(d.η) * d.ϕ + 1)))
end
function convert(::Type{NegativeBinomial{T}}, d::NegativeBinomialPoissonGamma) where {T<:Real}
    NegativeBinomial{T}(T(d.α), T(inv(d.β + 1)))
end

# to NegativeBinomialLocation
function convert(::Type{NegativeBinomialLocation{T}}, d::NegativeBinomial) where {T<:Real}
    NegativeBinomialLocation{T}(T((1 - d.p) / (d.p * d.r)), T(inv(d.r)))
end
function convert(::Type{NegativeBinomialLocation{T}}, d::NegativeBinomialLogLocation) where {T<:Real}
    NegativeBinomialLocation{T}(T(exp(d.η)), T(d.ϕ))
end
function convert(::Type{NegativeBinomialLocation{T}}, d::NegativeBinomialPoissonGamma) where {T<:Real}
    NegativeBinomialLocation{T}(T(d.α * d.β), T(inv(d.α)))
end

# to NegativeBinomialLogLocation
function convert(::Type{NegativeBinomialLogLocation{T}}, d::NegativeBinomial) where {T<:Real}
    NegativeBinomialLogLocation{T}(T(log1p(-d.p) - log(d.p) - log(d.r)), T(inv(d.r)))
end
function convert(::Type{NegativeBinomialLogLocation{T}}, d::NegativeBinomialLocation) where {T<:Real}
    NegativeBinomialLogLocation{T}(T(log(d.μ)), T(d.ϕ))
end
function convert(::Type{NegativeBinomialLogLocation{T}}, d::NegativeBinomialPoissonGamma) where {T<:Real}
    NegativeBinomialLogLocation{T}(T(log(d.α) + log(d.β)), T(inv(d.α)))
end

# to NegativeBinomialPoissonGamma
function convert(::Type{NegativeBinomialPoissonGamma{T}}, d::NegativeBinomial) where {T<:Real}
    NegativeBinomialPoissonGamma{T}(T(d.r), T((1 - d.p) / d.p))
end
function convert(::Type{NegativeBinomialPoissonGamma{T}}, d::NegativeBinomialLocation) where {T<:Real}
    NegativeBinomialPoissonGamma{T}(T(inv(d.ϕ)), T(d.μ * d.ϕ))
end
function convert(::Type{NegativeBinomialPoissonGamma{T}}, d::NegativeBinomialLogLocation) where {T<:Real}
    NegativeBinomialPoissonGamma{T}(T(inv(d.ϕ)), T(exp(d.η) * d.ϕ))
end

# Continuous univariate

convert(::Type{Gamma}, d::Exponential) = Gamma(1.0, d.θ)
convert(::Type{Gamma}, d::Erlang) = Gamma(d.α, d.θ)
