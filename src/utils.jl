# Utility functions

function _randi(Ku::Uint, U::Uint)
	x = rand(Uint)
	while x > U
		x = rand(Uint)
	end
	int(rem(x, Ku)) + 1
end

function randi(K::Int)  # Fast method to draw a random integer from 1:K
	Ku = uint(K)
	U = div(typemax(Uint), Ku) * Ku
	_randi(Ku, U)
end
