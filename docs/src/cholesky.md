# [Cholesky-variate Distributions](@id cholesky-variates)

*Cholesky-variate distributions* are distributions whose variate forms are `CholeskyVariate`. This means each draw is a factorization of a positive-definite matrix of type `LinearAlgebra.Cholesky` (the object produced by the function `LinearAlgebra.cholesky` applied to a dense positive-definite matrix.)

## Distributions

```@docs
LKJCholesky
```

## Index

```@index
Pages = ["cholesky.md"]
```
