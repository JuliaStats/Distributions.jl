# Parse distribution from a list entry

source("rdistributions.R")

read.entries <- function(filename) {
    # Read a list of entries from a given file
    lst <- read.table(filename, header=FALSE,
        sep="\t", blank.lines.skip = TRUE, comment.char="#")
    as.vector(lst$V1)
}

parse.entry <- function(entry) {
    # Parse an entry into a distribution name, and
    # a vector of argument values

    terms <- unlist(strsplit(entry, "\\s*,\\s*|\\(|\\)"))
    nt <- length(terms)
    list(name=terms[1],
         args=as.numeric(tail(terms, nt-1)))
}


get.distr <- function(entry) {
    # Get a distribution object based on a given entry
    parsed <- parse.entry(entry)
    dname <- parsed$name
    dargs <- parsed$args

    switch (dname,
        Bernoulli        = get.bernoulli(dargs),
        Binomial         = get.binomial(dargs),
        DiscreteUniform  = get.discrete.uniform(dargs),
        Geometric        = get.geometric(dargs),
        Hypergeometric   = get.hypergeometric(dargs),
        NegativeBinomial = get.negative.binomial(dargs),
        Poisson          = get.poisson(dargs)
    )
}

########################################
#
#  Get Distributions
#
########################################

get.arg <- function(a, i, v0) {
    if (!is.na(a) && i <= length(a)) { a[i] } else { v0 }
}

get.bernoulli <- function(args) {
    Bernoulli$new(p=get.arg(args, 1, 0.5))
}

get.binomial <- function(args) {
    Binomial$new(
        n=get.arg(args, 1, 1),
        p=get.arg(args, 2, 0.5))
}

get.discrete.uniform <- function(args) {
    nargs <- length(args)
    a <- 0
    b <- 1
    if (nargs == 1) {
        b <- args[1]
    } else if (nargs == 2) {
        a <- args[1]
        b <- args[2]
    }
    DiscreteUniform$new(a, b)
}

get.geometric <- function(args) {
    Geometric$new(p=get.arg(args, 1, 0.5))
}

get.hypergeometric <- function(args) {
    Hypergeometric$new(ns=args[1], nf=args[2], n=args[3])
}

get.negative.binomial <- function(args) {
    NegativeBinomial$new(
        r=get.arg(args, 1, 1.0),
        p=get.arg(args, 2, 0.5))
}

get.poisson <- function(args) {
    Poisson$new(lambda=get.arg(args, 1, 1.0))
}
