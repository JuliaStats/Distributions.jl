# Store an alias table
immutable DiscreteDistributionTable
	table::Vector{Vector{Int64}}
	bounds::Vector{Int64}
end

# TODO: Test if bit operations can speed up Base64 mod's and fld's
function DiscreteDistributionTable{T <: Real}(probs::Vector{T})
	# Cache the cardinality of the outcome set
	n = length(probs)

	# Convert all Float64's into integers
	vals = Array(Int64, n)
	for i in 1:n
		vals[i] = int(probs[i] * 64^9)
	end

	# Allocate digit table and digit sums as table bounds
	table = Array(Vector{Int64}, 9)
	bounds = zeros(Int64, 9)

	# Special case for deterministic distributions
	for i in 1:n
		if vals[i] == 64^9
			table[1] = Array(Int64, 64)
			for j in 1:64
				table[1][j] = i
			end
			bounds[1] = 64^9
			for j in 2:9
				table[j] = Array(Int64, 0)
				bounds[j] = 64^9
			end
			return DiscreteDistributionTable(table, bounds)
		end
	end

	# Fill tables
	multiplier = 1
	for index in 9:-1:1
		counts = Array(Int64, 0)
		for i in 1:n
			digit = mod(vals[i], 64)
			vals[i] = fld(vals[i], 64)
			bounds[index] += digit
			for itr in 1:digit
				push!(counts, i)
			end
		end
		bounds[index] *= multiplier
		table[index] = counts
		multiplier *= 64
	end

	# Make bounds cumulative
	bounds = cumsum(bounds)

	return DiscreteDistributionTable(table, bounds)
end

function draw(table::DiscreteDistributionTable)
	i = rand(1:64^9)
	if i == 64^9
		return table.table[9][rand(1:length(table.table[9]))]
	end
	bound = 1
	while i > table.bounds[bound] && bound < 9
		bound += 1
	end
	if bound > 1
		index = fld(i - table.bounds[bound - 1] - 1, 64^(9 - bound)) + 1
	else
		index = fld(i - 1, 64^(9 - bound)) + 1
	end
	return table.table[bound][index]
end
