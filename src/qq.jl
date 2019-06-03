struct QQPair
    qx::Vector{Float64}
    qy::Vector{Float64}
end

function qqbuild(x::Vector, y::Vector)
    n = min(length(x), length(y))
    grid = [0.0:(1 / (n - 1)):1.0;]
    qx = quantile(x, grid)
    qy = quantile(y, grid)
    return QQPair(qx, qy)
end

function qqbuild(x::Vector, d::UnivariateDistribution)
    n = length(x)
    grid = [(1 / (n - 1)):(1 / (n - 1)):(1.0 - (1 / (n - 1)));]
    qx = quantile(x, grid)
    qd = quantile.(Ref(d), grid)
    return QQPair(qx, qd)
end

function qqbuild(d::UnivariateDistribution, x::Vector)
    n = length(x)
    grid = [(1 / (n - 1)):(1 / (n - 1)):(1.0 - (1 / (n - 1)));]
    qd = quantile.(Ref(d), grid)
    qx = quantile(x, grid)
    return QQPair(qd, qx)
end
