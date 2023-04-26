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
    ppoints(n::Int, a::Real=0.5)

Generate a sequence of probability points of length `n`:

```math
(k − a)/(n + 1 − 2a), k ∈ 1, ..., n
```

`a` should be a number in ``[0, 1]``.

## References

https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot#Heuristics
"""
function ppoints(n, a=0.5)
    start = (1-a)/(n + 1 - 2*a)
    stop = (n-a)/(n + 1 - 2*a)
    range(start, stop, length=n)
end

function qqbuild(x::AbstractVector, d::UnivariateDistribution)
    n = length(x)
    grid = ppoints(n)
    qx = quantile(x, grid)
    qd = quantile.(Ref(d), grid)
    return QQPair(qx, qd)
end

function qqbuild(d::UnivariateDistribution, x::AbstractVector)
    n = length(x)
    grid = ppoints(n)
    qd = quantile.(Ref(d), grid)
    qx = quantile(x, grid)
    return QQPair(qd, qx)
end
