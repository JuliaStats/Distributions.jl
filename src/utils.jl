# Utility functions

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



