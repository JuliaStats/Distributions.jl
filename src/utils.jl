# Store an alias table
immutable DiscreteDistributionTable
	table::Vector{Any}
	bounds::Vector{Int64}
end

# TODO: Test if bit operations can speed up Base64 mod's and fld's
function DiscreteDistributionTable{T <: Real}(probs::Vector{T})
	# Cache the cardinality of the outcome set
	n = length(probs)

	# Convert all Float64's into integers
	vals = Array(Int64, n)
	for i in 1:n
		vals[i] = int(probs[i] * 10^6)
	end

	# Allocate digit table and digit sums as table bounds
	table = Array(Any, 6)
	bounds = zeros(Int64, 6)

	# Fill tables
	multiplier = 1
	for index in 6:-1:1
		counts = Array(Int64, 0)
		for i in 1:n
			digit = mod(vals[i], 10)
			vals[i] = fld(vals[i], 10)
			bounds[index] += digit
			for itr in 1:digit
				push!(counts, i)
			end
		end
		bounds[index] *= multiplier
		table[index] = counts
		multiplier *= 10
	end

	# Make bounds cumulative
	bounds = cumsum(bounds)

	return DiscreteDistributionTable(table, bounds)
end

function draw(table::DiscreteDistributionTable)
	i = rand(1:10^6)
	bound = 1
	while i > table.bounds[bound] && bound < 6
		bound += 1
	end
	if bound > 1
		index = fld(i - table.bounds[bound - 1] - 1, 10^(6 - bound)) + 1
	else
		index = fld(i - 1, 10^(6 - bound)) + 1
	end
	return table.table[bound][index]
end
