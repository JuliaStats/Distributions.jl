immutable NormalInverseGaussian <: ContinuousUnivariateDistribution
  μ::Float64
  α::Float64
  β::Float64
  δ::Float64

  function NormalInverseGaussian(μ::Real, α::Real, β::Real, δ::Real)
    new(μ, α, β, δ)
  end
end

@distr_support NormalInverseGaussian -Inf Inf

params(d::NormalInverseGaussian) = (d.μ, d.α, d.β, d.δ)

mean(d::NormalInverseGaussian) = d.μ + d.δ * d.β / sqrt(d.α^2 - d.β^2)
var(d::NormalInverseGaussian) = d.δ * d.α^2 / sqrt(d.α^2 - d.β^2)^3
skewness(d::NormalInverseGaussian) = 3d.β / (d.α * sqrt(d.δ * sqrt(d.α^2 - d.β^2)))

function pdf(d::NormalInverseGaussian, x::Float64)
	μ, α, β, δ = params(d)
	α * δ * besselk(1, α*sqrt(δ^2+(x-μ)^2)) / (π*sqrt(δ^2+(x-μ)^2)) * exp(δ*sqrt(α^2-β^2) + β*(x-μ))
end

function logpdf(d::NormalInverseGaussian, x::Float64)
  μ, α, β, δ = params(d)
  log(α*δ) + log(besselk(1, α*sqrt(δ^2+(x-μ)^2))) - log(π*sqrt(δ^2+(x-μ)^2)) + δ*sqrt(α^2-β^2) + β*(x-μ)
end
