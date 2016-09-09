# Composite Multivariate distribution

## Generic Composite multivariate distribution class

module CompositeDistributions

if !isdefined(:Distributions) using Distributions end

import Base.length, Base.mean, Base.show
import Distributions.mean, Distributions.mode, Distributions.var, Distributions.cov
import Distributions.entropy, Distributions.insupport
import Distributions._logpdf, Distributions._pdf!
import Distributions._rand!

export AbstractCompositeContinuousDist, ContinuousMultivariateDistribution
export GenericCompositeContinuousDist, CompositeDist
export length, params, mean, mode, var, cov, entropy
export insupport, _logpdf, gradlogpdf, _rand!


abstract AbstractCompositeContinuousDist <: ContinuousMultivariateDistribution
#abstract AbstractCompositeDiscreteDist <: DiscreteMultivariateDistribution  # An idea, but not implemented yet

immutable GenericCompositeContinuousDist <: AbstractCompositeContinuousDist
    dist::Vector{ContinuousDistribution}
    indices::Vector{UnitRange{Int64}}

    function GenericCompositeContinuousDist(dist::Vector{ContinuousDistribution})
      length(dist) > 0 || error("number of distributions must be positive")
      #@compat new(dist)
      indices = Array(UnitRange{Int64},length(dist))
      idx_start = 1; idx_stop = 0
      for i in 1:length(dist)
         idx_stop += length(dist[i])
         indices[i] = idx_start:idx_stop
         idx_start += length(dist[i])
      end
      new(dist,indices)
    end
end

typealias CompositeDist  GenericCompositeContinuousDist

### Parameters
length(d::GenericCompositeContinuousDist, idx::Integer) = length(d.dist[idx])
length(d::GenericCompositeContinuousDist)  = d.indices[end].stop

params(d::GenericCompositeContinuousDist, idx::Integer) = params(d.dist[idx])
function params(d::GenericCompositeContinuousDist) 
  n = length(d)
  p = Array(Float64,n)
  for i in 1:length(d.dist)
     p[d.indices[i]] = params(d,i)
  end
  return p
end

#=  Seems like a good idea, but would require set_params for all distributions used
function set_params(d::GenericCompositeContinuousDist, x::DenseVector{T}) 
  n = length(d)
  for i in 1:length(d.dist)
     set_params(d.dist[i], params(d,i) )
  end
end
=#

### Basic statistics

mean(d::GenericCompositeContinuousDist, idx::Integer) = mean(d.dist[idx])
function mean(d::GenericCompositeContinuousDist) 
  n = length(d)
  mu = Array(Float64,n)  
  for i in 1:length(d.dist)
     mu[d.indices[i]] = mean(d,i)
  end
  return mu
end

mode(d::GenericCompositeContinuousDist, idx::Integer) = mode(d.dist[idx])
function mode(d::GenericCompositeContinuousDist) 
  n = length(d)
  mo = Array(Float64,n)
  for i in 1:length(d.dist)
     mo[d.indices[i]] = mode(d,i)
  end
  return mo
end

var(d::GenericCompositeContinuousDist, idx::Integer) = var(d.dist[idx])
function var(d::GenericCompositeContinuousDist) 
  n = length(d)
  v = Array(Float64,n)
  for i in 1:length(d.dist)
     v[d.indices[i]] = var(d,i)
  end
  return v
end

cov(d::GenericCompositeContinuousDist, idx::Integer) = cov(d.dist[idx])
function cov(d::GenericCompositeContinuousDist) 
  n = length(d)
  covar = zeros(Float64,(n,n) )
  for i in 1:length(d.dist)
     if length(d.dist[i]) == 1
        covar[d.indices[i]] = var(d,i)
     else
        covar[d.indices[i],d.indices[i]] = cov(d,i)
     end
  end
  return covar
end

entropy(d::GenericCompositeContinuousDist, idx::Integer) = entropy(d.dist[idx])
function entropy(d::GenericCompositeContinuousDist)
  mapreduce(entropy,+,d.dist)
  #=sum = 0
  for i in 1:length(d.dist)
    sum += entropy(d,i)
  end
  return sum =#
end

### Evaluation 

function index(d::GenericCompositeContinuousDist, idx::Integer)
  # note that this function is intentionally not type stable.  
  # Needs to provide scalar index for univariate distributions or range of indices for multivariate distributions
  #if d.indices[idx].start == d.indices[idx].stop
  if isa(d.dist[idx], UnivariateDistribution)
     return d.indices[idx].start
  else
     return d.indices[idx]
  end
end

function insupport{T<:Real}(d::GenericCompositeContinuousDist, x::AbstractVector{T})  
  if ! (length(d) == length(x)) return false end
  for i in 1:length(d.dist)
    if !insupport(d.dist[i],x[index(d,i)]) return false end
  end
  return true
end

function _logpdf{T<:Real}(d::GenericCompositeContinuousDist, x::AbstractArray{T,1})
  sum = zero(T) 
  for i in 1:length(d.dist)
    #sum += _logpdf(d.dist[i],x[index(d,i)])
    sum += logpdf(d.dist[i],x[index(d,i)])
  end
  return sum      
end

function gradlogpdf{T<:Real}(d::GenericCompositeContinuousDist, x::DenseVector{T})
    z = Array(T,length(d))
    for i = 1:length(d.dist)
        z[index(d,i)] = gradlogpdf(d.dist[i],view(x,index(d,i)))
    end
    return z
end

### Sampling

function _rand!{T<:Real}(d::GenericCompositeContinuousDist, x::DenseVector{T})
    for i = 1:length(d.dist)
        _rand!(d.dist[i],view(x,index(d,i)))
    end
    return x
end

function _rand!{T<:Real}(d::GenericCompositeContinuousDist, x::DenseMatrix{T})
    for i = 1:length(d.dist)
        _rand!(d.dist[i],view(x,index(d,i),:))   # Check got dimensions right
    end
    return x
end

### Show

distrname(d::GenericCompositeContinuousDist) = "GenericCompositeContinuous"
function Base.show(io::IO, d::GenericCompositeContinuousDist) 
  print(io,distrname(d) * ":\n")
  for i in 1:length(d.dist)
     show(io, d.dist[i] )
     print(io,"\n")
  end
end

end # module
