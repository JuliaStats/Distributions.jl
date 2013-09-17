# Test type relations

using Distributions

@assert UnivariateDistribution <: Distribution
@assert MultivariateDistribution <: Distribution
@assert MatrixDistribution <: Distribution

@assert DiscreteDistribution <: Distribution
@assert ContinuousDistribution <: Distribution

@assert DiscreteUnivariateDistribution <: DiscreteDistribution 
@assert DiscreteUnivariateDistribution <: UnivariateDistribution
@assert ContinuousUnivariateDistribution <: ContinuousDistribution
@assert ContinuousUnivariateDistribution <: UnivariateDistribution
@assert DiscreteMultivariateDistribution <: DiscreteDistribution
@assert DiscreteMultivariateDistribution <: MultivariateDistribution
@assert ContinuousMultivariateDistribution <: ContinuousDistribution
@assert ContinuousMultivariateDistribution <: MultivariateDistribution
@assert DiscreteMatrixDistribution <: DiscreteDistribution
@assert DiscreteMatrixDistribution <: MatrixDistribution
@assert ContinuousMatrixDistribution <: ContinuousDistribution
@assert ContinuousMatrixDistribution <: MatrixDistribution
