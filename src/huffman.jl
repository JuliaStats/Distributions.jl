# Constructs a Huffman coding tree for efficiently generating discrete random
# variables by relative weights.  

# Maximum granularity is 1/float64(typemax(Uint64)) = 5.4-20, but also
# supports non-base2 divisions via rejection sampling.

import Base.isless, Base.show, Base.getindex, Base.rand

abstract HuffmanNode{T}

immutable HuffmanLeaf{T} <: HuffmanNode{T}
    value::T
    weight::Uint64
end

immutable HuffmanBranch{T} <: HuffmanNode{T}
    left::HuffmanNode{T}
    right::HuffmanNode{T}
    weight::Uint64
end
HuffmanBranch{T}(ha::HuffmanNode{T},hb::HuffmanNode{T}) = HuffmanBranch(ha, hb, ha.weight + hb.weight)

isless{T}(ha::HuffmanNode{T}, hb::HuffmanNode{T}) = isless(ha.weight,hb.weight)

show{T}(io::IO, t::HuffmanNode{T}) = show(io,typeof(t))

function getindex{T}(h::HuffmanBranch{T},u::Uint64) 
    while isa(h,HuffmanBranch{T})
        if u < h.left.weight
            h = h.left
        else 
            u -= h.left.weight
            h = h.right
        end
    end
    h.value
end

# build the huffman tree
# could be slightly more efficient using a Deque.
function huffman{T}(values::AbstractVector{T},weights::AbstractVector{Uint64})
    leafs = [HuffmanLeaf{T}(values[i],weights[i]) for i = 1:length(weights)]
    sort!(leafs,Base.Sort.Reverse)
    
    branches = Array(HuffmanBranch{T},0)
        
    while !isempty(leafs) || length(branches) > 1
        left = isempty(branches) || (!isempty(leafs) && first(leafs) < first(branches)) ? pop!(leafs) : pop!(branches)
        right = isempty(branches) || (!isempty(leafs) && first(leafs) < first(branches)) ? pop!(leafs) : pop!(branches)
        unshift!(branches,HuffmanBranch(left,right))
    end
    
    pop!(branches)    
end

function rand{T}(h::HuffmanNode{T})
    w = h.weight
    # generate uniform Uint64 objects on the range 0:(w-1)
    # unfortunately we can't use Range objects, as they don't have sufficient length
    u = rand(Uint64)
    if (w & (w-1)) == 0
        # power of 2
        u = u & (w-1)
    else
        m = typemax(Uint64)
        lim = m - (rem(m,w)+1)
        while u > lim
            u = rand(Uint64)
        end
        u = rem(u,w)
    end
    h[u]
end
