# Sample from arbitrary arrays
 
################################################################
#
#  A variety of algorithms for sampling without replacement
#
#  They are suited for different cases.
#
#  Particularly,
#  - Fisher-Yates sampler is suited for general cases
#    where n is not overly large
#
#  - Self avoiding sampler is suited for cases where k << n
#
################################################################

function pick2!(a::AbstractArray, x::AbstractArray)
    # Pick a pair of values without replacement

    n0 = length(a)
    i1 = randi(n0)
    i2 = randi(n0 - 1)
    if i2 == i1
        i2 = n0
    end

    x[1] = a[i1]
    x[2] = a[i2]
end

## A sampler that implements without-replacement sampling 
## via Fisher-Yates shuffling
##
immutable FisherYatesSampler
    n::Int 
    seq::Vector{Int}   # Internal sequence for shuffling

    FisherYatesSampler(n::Int) = new(n, [1:n])
end

function rand!(s::FisherYatesSampler, a::AbstractArray, x::AbstractArray)
    # draw samples without-replacement to x

    n::Int = s.n
    k::Int = length(x)
    if k > n
        throw(ArgumentError("Cannot draw more than n samples without replacement."))
    end

    seq::Vector{Int} = s.seq
    for i = 1:k
        j = randi(i, n)
        sj = seq[j]
        x[i] = a[sj]
        seq[j] = seq[i]
        seq[i] = sj
    end
    x
end

fisher_yates_sample!(a::AbstractArray, x::AbstractArray) = rand!(FisherYatesSampler(length(a)), a, x)

function self_avoid_sample!{T}(a::AbstractArray{T}, x::AbstractArray)
    # This algorithm is suitable when length(x) << length(a)

    s = Set{T}()
    # sizehint(s, length(x))
    rgen = RandIntSampler(length(a))

    # first one    
    idx = rand(rgen)
    x[1] = a[idx]
    push!(s, idx)

    # remaining
    for i = 2:length(x)
        idx = rand(rgen)
        while in(s, idx)
            idx = rand(rgen)
        end
        x[i] = a[idx]
        push!(s, idx)
    end
    x
end

# Ordered sampling without replacement
# Author: Mike Innes

function rand_first_index(n, k)
  r = rand()
  p = k/n
  i = 1
  while p < r
    i += 1
    p += (1-p)k/(n-(i-1))
  end
  return i
end

function ordered_sample_norep!(xs::AbstractArray, target::AbstractArray)
  n = length(xs)
  k = length(target)
  i = 0
  for j in 1:k
    step = rand_first_index(n, k)
    n -= step
    i += step
    target[j] = xs[i]
    k -= 1
  end
  return target
end

function ordered_sample_rep!(xs::AbstractArray, target::AbstractArray)
  n = length(xs)
  n_left = n
  k = length(target)
  j = 0

  for i = 1:n
    k > 0 || break
    num = i == n ? k : rand(Binomial(k, 1/n_left))
    for _ = 1:num
      j += 1
      target[j] = xs[i]
    end
    k -= num
    n_left -= 1
  end
  return target
end

###########################################################
#
#   Interface functions
#
###########################################################

sample(a::AbstractArray) = a[randi(length(a))]

function sample!(a::AbstractArray, x::AbstractArray; replace=true, ordered=false)
    n = length(a)
    k = length(x)

    if !isempty(x)
        if ordered
          replace ? ordered_sample_rep!(a, x) : ordered_sample_norep!(a, x)
        else
            if replace   # with replacement
                s = RandIntSampler(n)
                for i = 1:k
                    x[i] = a[rand(s)]
                end

            else  # without replacement
                if k > n
                    throw(ArgumentError("n exceeds the length of x"))
                end

                if k == 1
                    x[1] = sample(a)
                    
                elseif k == 2
                    pick2!(a, x)

                elseif n < k * max(k, 100) 
                    fisher_yates_sample!(a, x)
                    
                else
                    self_avoid_sample!(a, x)
                end
            end
        end
    end
    x
end

function sample{T}(a::AbstractArray{T}, n::Integer; replace=true, ordered=false)
    sample!(a, Array(T, n); replace=replace, ordered=ordered)
end

function sample{T}(a::AbstractArray{T}, dims::Dims; replace=true, ordered=false)
    sample!(a, Array(T, dims); replace=replace, ordered=ordered)
end


################################################################
#
#  Weighted sampling
#
################################################################

function wsample(ws::AbstractArray; wsum::Number = sum(ws))
  t = rand() * wsum
  i = 0
  p = 0.
  while p < t
    i += 1
    p += ws[i]
  end
  return i
end

wsample(xs::AbstractArray, ws::AbstractArray; wsum::Number = sum(ws)) =
  xs[wsample(ws, wsum=wsum)]

# Author: Mike Innes
function ordered_wsample!(xs::AbstractArray, ws::AbstractArray, target::AbstractArray; wsum::Number = sum(ws))
  n = length(xs)
  k = length(target)
  j = 0

  length(ws) == n || throw(ArgumentError("Inconsistent argument dimensions."))

  for i = 1:n
    k > 0 || break
    num = i == n ? k : rand(Binomial(k, ws[i]/wsum))
    for _ = 1:num
      j += 1
      target[j] = xs[i]
    end
    k -= num
    wsum -= ws[i]
  end
  return target
end

function wsample!(xs::AbstractArray, ws::AbstractArray, target::AbstractArray; wsum::Number = sum(ws), ordered::Bool = false)
  k = length(target)
  ordered && return ordered_wsample!(xs, ws, target, wsum = wsum)
  k > 100 && return ordered_wsample!(xs, ws, target, wsum = wsum) |> shuffle!

  for i = 1:k
    target[i] = wsample(xs)
  end
  return target
end

wsample(xs::AbstractArray, ws::AbstractArray, k; wsum::Number = sum(ws), ordered::Bool = false) =
  wsample!(xs, ws, similar(xs, k), wsum = wsum, ordered = ordered)
