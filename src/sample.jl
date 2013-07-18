# Sample from arbitrary arrays

#####
#
#  Sampling from an array with even probabilities
#
#####

### Internal implementation of sampling with replacement

sample_one{T<:Integer}(a::Range1{T}) = rand(a)
sample_one(a::AbstractArray) = a[rand(1:length(a))]


function sample_pair_without_rep!(a::AbstractArray, x::Array)
    # Pick a pair of values without replacement

    n0 = 1 : length(a)
    i1 = rand(1:n0)
    i2 = rand(1:n0-1)
    if i2 == i1
        i2 = n0
    end

    x1[1] = a[i1]
    x2[1] = a[i2]
end

function sample_without_rep_by_set!{T}(a::AbstractArray{T}, x::Array)
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
end

function sample_without_rep_by_shuffle!(a::AbstractArray, x::Array)
    # Efficient for general cases

    inds = [1:length(a)]
    randshuffle!(inds, length(x))
    for i = 1:length(x)
        x[i] = a[inds[i]]
    end
end

# Debate: shall we expose this ?
function sample_without_rep!(a::AbstractArray, x::Array)
    n0 = length(a)
    n = length(x)
    if n > n0
        throw(ArgumentError("n exceeds the length of x"))
    end

    if n == 1
        x[1] = sample_one(a)
        
    elseif n == 2
        sample_pair_without_rep!(a, x)

    elseif n * max(n, 100) < n0
        sample_without_rep_by_set!(a, x)
        
    else
        sample_without_rep_by_shuffle!(a, x)
    end
end

# Interface function

function sample!(a::AbstractArray, x::Array; rep=true)
	rep ? rand!(a, x) : sample_with_rep!(a, x)
	return x
end

sample{T}(a::AbstractArray{T}, n::Integer; rep=true) = sample!(a, Array(T, n); rep=rep)
sample{T}(a::AbstractArray{T}, dims::Dims; rep=true) = sample!(a, Array(T, dims); rep=rep)


