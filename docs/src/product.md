# Product Distributions

Product distributions are joint distributions of multiple independent distributions.
It is recommended to use `product_distribution` to construct product distributions.
Depending on the type of the argument, it may construct a different distribution type.

## Multivariate products

```@docs
Distributions.product_distribution(::AbstractArray{<:Distribution{<:ArrayLikeVariate}})
Distributions.product_distribution(::AbstractVector{<:Normal})
Distributions.ProductDistribution
Distributions.Product
```

## NamedTuple-variate products

```@docs
Distributions.product_distribution(::NamedTuple{<:Any,<:Tuple{Distribution,Vararg{Distribution}}})
Distributions.ProductNamedTupleDistribution
```

## Index

```@index
Pages = ["product.md"]
```
