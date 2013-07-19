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

function pick2!(a::AbstractArray, x::Array)
    # Pick a pair of values without replacement

    n0 = 1 : length(a)
    i1 = randi(n0)
    i2 = randi(n0 - 1)
    if i2 == i1
        i2 = n0
    end

    x1[1] = a[i1]
    x2[1] = a[i2]
end

## A sampler that implements without-replacement sampling 
## via Fisher-Yates shuffling
##
immutable FisherYatesSampler
    n::Int 
    seq::Vector{Int}   # Internal sequence for shuffling

    FisherYatesSampler(n::Int) = new(n, [1:n])
end

function rand!(s::FisherYatesSampler, a::AbstractArray, x::Array)
    # draw samples without-replacement to x

    k = length(x)
    if k > s.n
        throw(ArgumentError("Cannot draw more than n samples without replacement."))
    end

    seq::Vector{Int} = s.seq
    for i = 1:k
        j = randi(i, k)
        sj = seq[j]
        x[i] = a[sj]
        seq[j] = seq[i]
        seq[i] = sj
    end
    x
end

fisher_yates_sample!(a::AbstractArray, x::Array) = rand!(FisherYatesSampler(length(a)), a, x)

function self_avoid_sample!{T}(a::AbstractArray{T}, x::Array)
    # This algorithm is suitable when length(x) << length(a)

    s = Set{T}()
    sizehint(s, length(x))
    rg = 1:length(a)

    # first one    
    idx = rand(rg)
    x[1] = a[idx]
    add!(s, idx)

    # remaining
    for i = 2:length(x)
        idx = rand(rg)
        while !contains(s, idx)
            idx = rand(rg)
        end
        x[i] = a[idx]
        add!(s, idx)
    end
    x
end

function sample_without_replacement!(a::AbstractArray, x::Array)
    n = length(a)
    k = length(x)
    if k > n
        throw(ArgumentError("n exceeds the length of x"))
    end

    if n == 1
        x[1] = a[randi(length(a))]
        
    elseif n == 2
        pick2!(a, x)

    elseif n * max(n, 100) < n0
        fisher_yates_sample!(a, x)
        
    else
        self_avoid_sample!(a, x)
    end
    x
end

# Interface function

function sample!(a::AbstractArray, x::Array; replace=true)
	rep ? rand!(a, x) : sample_with_rep!(a, x)
	return x
end

function sample{T}(a::AbstractArray{T}, n::Integer; replace=true)
    sample!(a, Array(T, n); replace=replace)
end

function sample{T}(a::AbstractArray{T}, dims::Dims; replace=true)
    sample!(a, Array(T, dims); replace=replace)
end






