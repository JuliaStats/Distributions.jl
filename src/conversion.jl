
# Discrete univariate

convert(::Type{Binomial}, d::Bernoulli) = Binomial(1, d.p)

# to NegativeBinomial
function convert(::Type{NegativeBinomial{T}}, d::NegativeBinomial2) where {T<:Real}
    NegativeBinomial{T}(T(d.ϕ), T(d.ϕ / (d.μ + d.ϕ)))
end
function convert(::Type{NegativeBinomial{T}}, d::NegativeBinomial2Log) where {T<:Real}
    NegativeBinomial{T}(T(d.ϕ), T(d.ϕ / (exp(d.η) + d.ϕ)))
end
function convert(::Type{NegativeBinomial{T}}, d::NegativeBinomial3) where {T<:Real}
    NegativeBinomial{T}(T(d.α), T(one(T) / (d.β + one(T))))
end

# to NegativeBinomial2
function convert(::Type{NegativeBinomial2{T}}, d::NegativeBinomial) where {T<:Real}
    NegativeBinomial2{T}(T(d.r * (1 - d.p) / d.p), T(d.r))
end
function convert(::Type{NegativeBinomial2{T}}, d::NegativeBinomial2Log) where {T<:Real}
    NegativeBinomial2{T}(T(exp(d.η)), T(d.ϕ))
end
function convert(::Type{NegativeBinomial2{T}}, d::NegativeBinomial3) where {T<:Real}
    NegativeBinomial2{T}(T(d.α * d.β), T(d.α))
end

# to NegativeBinomial2Log
function convert(::Type{NegativeBinomial2Log{T}}, d::NegativeBinomial) where {T<:Real}
    NegativeBinomial2Log{T}(T(log(d.r) + log1p(-d.p) - log(d.p)), T(d.r))
end
function convert(::Type{NegativeBinomial2Log{T}}, d::NegativeBinomial2) where {T<:Real}
    NegativeBinomial2Log{T}(T(log(d.μ)), T(d.ϕ))
end
function convert(::Type{NegativeBinomial2Log{T}}, d::NegativeBinomial3) where {T<:Real}
    NegativeBinomial2Log{T}(T(log(d.α) + log(d.β)), T(d.α))
end

# to NegativeBinomial3
function convert(::Type{NegativeBinomial3{T}}, d::NegativeBinomial) where {T<:Real}
    NegativeBinomial3{T}(T(d.r), T((1 - d.p) / d.p))
end
function convert(::Type{NegativeBinomial3{T}}, d::NegativeBinomial2) where {T<:Real}
    NegativeBinomial3{T}(T(d.ϕ), T(d.μ / d.ϕ))
end
function convert(::Type{NegativeBinomial3{T}}, d::NegativeBinomial2Log) where {T<:Real}
    NegativeBinomial3{T}(T(d.ϕ), T(exp(d.η) / d.ϕ))
end

# Continuous univariate

convert(::Type{Gamma}, d::Exponential) = Gamma(1.0, d.θ)
convert(::Type{Gamma}, d::Erlang) = Gamma(d.α, d.θ)
