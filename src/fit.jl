function digamma(x::Float64) # Use Int instead?
  ccall(dlsym(_jl_libRmath,  :digamma), Float64, (Float64, ), x)
end

function trigamma(x::Float64) # Use Int instead?
  ccall(dlsym(_jl_libRmath,  :trigamma), Float64, (Float64, ), x)
end

function fit(::Type{Bernoulli}, x::Array)
  for i in 1:length(x)
    if !insupport(Bernoulli(), x[i])
      error("Bernoulli observations must be 0/1 values")
    end
  end
  return Bernoulli(mean(x))
end

function fit(::Type{Beta}, x::Array)
  for i in 1:length(x)
    if !insupport(Beta(), x[i])
      error("Bernoulli observations must be in [0,1]")
    end
  end
  x_bar = mean(x)
  v_bar = var(x)
  a = x_bar * (((x_bar * (1. - x_bar)) / v_bar) - 1.)
  b = (1. - x_bar) * (((x_bar * (1. - x_bar)) / v_bar) - 1.)
  return Beta(a, b)
end

function fit(::Type{Binomial}, x::Real, n::Real)
  if x > n || x < 0
    error("For binomial observations, x must lie in [0, n]")
  end
  return Binomial(int(n), x / n)
end

# Categorical

# Cauchy

# Chisq

# Dirichlet

function fit(::Type{DiscreteUniform}, x::Array)
  DiscreteUniform(min(x), max(x))
end

function fit(::Type{Exponential}, x::Array)
  for i in 1:length(x)
    if !insupport(Exponential(), x[i])
      error("Exponential observations must be non-negative values")
    end
  end
  return Exponential(mean(x))
end

# FDist

function fit(::Type{Gamma}, x::Array)
  a_old = 0.5 / (log(mean(x)) - mean(log(x)))
  a_new = 0.5 / (log(mean(x)) - mean(log(x)))

  z = 1.0 / a_old + (mean(log(x)) - log(mean(x)) + log(a_old) - digamma(a_old)) / (a_old^2 * (1.0 / a_old - trigamma(a_old)))
  a_new, a_old = (1.0 / z, a_new)

  iterations = 1

  while abs(a_old - a_new) > 10e-16 && iterations <= 10_000
    z = 1.0 / a_old + (mean(log(x)) - log(mean(x)) + log(a_old) - digamma(a_old)) / (a_old^2 * (1.0 / a_old - trigamma(a_old)))
    a_new, a_old = (1.0 / z, a_new)
  end

  if iterations >= 10_000
    error("Parameter estimation failed to converge in 10,000 iterations")
  end

  Gamma(a_new, mean(x) / a_new)
end

function fit(::Type{Geometric}, x::Array)
  Geometric(1.0 / mean(convert(Array{Int}, x)))
end

# HyperGeometric

# Logistic

function fit(::Type{Laplace}, x::Array)
  a = median(x)
  deviations = 0.0
  for i in 1:length(x)
    deviations += abs(x[i] - a)
  end
  b = deviations / length(x)
  Laplace(a, b)
end

# logNormal

# function fit(Multinomial)
# end

# Assumes observations are rows
# Positive-definite errors
# function fit{T <: Real}(x::Matrix{T}, d::MultivariateNormal)
#   MultivariateNormal(reshape(mean(x, 1), size(x, 2)), cov(x))
# end

# fit([1.0 -0.1; -2.0 0.1], MultivariateNormal())

# NegativeBinomial

# NoncentralBeta

# NoncentralChisq

# NoncentralF

# NoncentralT

# Normal
# Not strict MLE
function fit(::Type{Normal}, x::Array)
  Normal(mean(x), std(x))
end

# Poisson
function fit(::Type{Poisson}, x::Array)
  for i in 1:length(x)
    if !insupport(Poisson(), x[i])
      error("Poisson observations must be non-negative integers")
    end
  end
  Poisson(mean(x))
end

# TDist

# Uniform
function fit(::Type{Uniform}, x::Array)
  Uniform(min(x), max(x))
end

# Weibull
