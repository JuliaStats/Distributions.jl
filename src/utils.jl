# Utility functions

function isprobvec(p::Vector{Float64})
    s = 0.
    for i = 1:length(p)
        pi = p[i]
        s += pi
        if pi < 0
            return false
        end
    end      
    return abs(s - 1.0) <= 1.0e-8
end


function _randu(Ku::Uint, U::Uint)   # ~ U[0:Ku-1]
	x = rand(Uint)
	while x > U
		x = rand(Uint)
	end
	rem(x, Ku)
end

function randi(K::Int)  # Fast method to draw a random integer from 1:K
	Ku = uint(K)
	U = div(typemax(Uint), Ku) * Ku
	int(_randu(Ku, U)) + 1
end

function randi(a::Int, b::Int)  # ~ U[a:b]
	Ku = uint(b - a + 1)
	U = div(typemax(Uint), Ku) * Ku
	int(_randu(Ku, U)) + a
end

immutable RandIntSampler
	a::Int
	Ku::Uint
	U::Uint

	function RandIntSampler(K::Int) # 1:K
		Ku = uint(K)
		U = div(typemax(Uint), Ku) * Ku
		new(1, Ku, U)
	end

	function RandIntSampler(a::Int, b::Int)  # a:b
		Ku = uint(b - a + 1)
		U = div(typemax(Uint), Ku) * Ku
		new(a, Ku, U)
	end
end

rand(s::RandIntSampler) = int(_randu(s.Ku, s.U)) + s.a
