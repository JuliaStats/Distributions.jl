struct QQPair{U<:AbstractVector, V<:AbstractVector}
    qx::U
    qy::V
end

function qqbuild(x::AbstractVector, y::AbstractVector)
    n = min(length(x), length(y))
    grid = [0.0:(1 / (n - 1)):1.0;]
    qx = quantile(x, grid)
    qy = quantile(y, grid)
    return QQPair(qx, qy)
end


"""
    ppoints(n::Int)

Generate a sequence of probability points of length `n`:

```math
(k − 0.5)/n, \\qquad k \\in \\{1, \\ldots, n\\}
```

## References

https://ch.mathworks.com/help/stats/probplot.html
"""
function ppoints(n::Int)
    m = 2 * n
    return (1:2:(m - 1)) ./ m
end


function qqbuild(x::AbstractVector, d::UnivariateDistribution)
    n = length(x)
    grid = ppoints(n)
    qx = quantile(x, grid)
    qd = map(Base.Fix1(quantile, d), grid)
    return QQPair(qx, qd)
end

function qqbuild(d::UnivariateDistribution, x::AbstractVector)
    n = length(x)
    grid = ppoints(n)
    qd = map(Base.Fix1(quantile, d), grid)
    qx = quantile(x, grid)
    return QQPair(qd, qx)
end
