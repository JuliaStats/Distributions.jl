# A R script to generate reference values for testing distributions
#
#   This script will take the following two list of distribution
#   entries as input:
#
#       - continuous_test.lst
#       - discrete_test.lst
#
#   The generated results will be written to the following files,
#   in JSON format, respectively
#
#       - continuous_test.json
#       - discrete_test.json
#
#   To run this script:
#
#       Rscript gendref.R
#
#   Note: this scripts depends on a number of R libraries for
#         file I/O and the computation of statistical functions.
#

source("parseDistrs.R")

########################################
#
#  Point wise evaluation
#
########################################

eval.samples <- function(distr) {
    if (distr$is.discrete) {
        su <- distr$supp()
        dmin <- su[1]
        dmax <- su[2]
        vmin <- if (is.finite(dmin)) { dmin } else { distr$quan(0.01) }
        vmax <- if (is.finite(dmax)) { dmax } else { distr$quan(0.99) }

        if (vmax - vmin + 1 <= 10) {
            seq(vmin, vmax)
        } else {
            xs <- unique(round(distr$quan(seq(0.1, 0.9, 0.1))))
            if (vmin < xs[1]) {
                xs <- c(vmin, xs)
            }
            if (vmax > xs[length(xs)]) {
                xs <- c(xs, vmax)
            }
            xs
        }
    } else {
        distr$quan(seq(0.1, 0.9, 0.1))
    }
}

eval.points <- function(distr) {
    xs <- eval.samples(distr)
    list(x  = xs,
         pd = distr$pdf(xs),
         lp = distr$logpdf(xs),
         cp = distr$cdf(xs))
}

eval.quans <- function(distr) {
    qs <- c(0.10, 0.25, 0.50, 0.75, 0.90)
    list(q = qs,
         x = distr$quan(qs))
}


########################################
#
#  JSON output
#
########################################

json.str <- function(val) {
    # convert a value to json string
    v <- val
    if (is.numeric(val)) {
        if (is.nan(val)) {
            v <- "nan"
        } else if (!is.finite(val)) {
            v <- ifelse(val > 0, "inf", "-inf")
        }
    }

    if (is.numeric(v)) {
        if (v == floor(v)) {
            sprintf("%g", v)
        } else {
            sprintf("%.15g", v)
        }
    } else {
        sprintf("\"%s\"", v)
    }
}

cat.attribute <- function (prefix, name, val, last=FALSE) {
    s <- sprintf("\"%s\": %s", name, json.str(val))
    if (last) {
        cat(prefix, s, "\n", sep="")
    } else {
        cat(prefix, s, ",\n", sep="")
    }
}

cat.point <- function (prefix, points, i) {
    n <- length(points$x)
    x <- points$x[i]
    p <- points$pd[i]
    lp <- points$lp[i]
    cd <- points$cp[i]

    s <- sprintf("{ \"x\": %s, \"pdf\": %s, \"logpdf\": %s, \"cdf\": %s }",
        x, json.str(p), json.str(lp), json.str(cd))

    if (i == n) {
        cat(prefix, s, "\n", sep="")
    } else {
        cat(prefix, s, ",\n", sep="")
    }
}

cat.quan <- function (prefix, quans, i) {
    n <- length(quans$q)
    q <- quans$q[i]
    x <- quans$x[i]
    s <- sprintf("{ \"q\": %.2f, \"x\": %s }", q, json.str(x))
    if (i == n) {
        cat(prefix, s, "\n", sep="")
    } else {
        cat(prefix, s, ",\n", sep="")
    }
}

write.json <- function(info, last=FALSE) {
    # extract fields
    expr <- info$expr
    distr <- info$distr
    points <- info$points
    quans <- info$quans

    cat("{\n")
    dname <- class(distr)[1]
    suppr <- distr$supp()
    cat.attribute("  ", "expr", expr)
    cat.attribute("  ", "dtype", dname)
    cat.attribute("  ", "minimum", suppr[1])
    cat.attribute("  ", "maximum", suppr[2])

    # output properties
    cat("  \"properties\": {\n")
    props <- distr$properties()
    i <- 0
    n <- length(props)
    for (pn in names(props)) {
        i <- i + 1
        cat.attribute("    ", pn, props[[pn]], last=(i==n))
    }
    cat("  },\n")

    # output points
    n <- length(points$x)
    if (n > 0) {
        cat("  \"points\": [\n")
        for (i in 1:n) {
            cat.point("    ", points, i)
        }
        cat("  ],\n")
    } else {
        cat("  \"points\": [],\n")
    }

    # output quantile points
    n <- length(quans$q)
    if (n > 0) {
        cat("  \"quans\": [\n")
        for (i in 1:n) {
            cat.quan("    ", quans, i)
        }
        cat("  ]\n")
    } else {
        cat("  \"quans\": []\n")
    }

    # closing
    if (last) {
        cat("}\n")
    } else {
        cat("},\n")
    }
}


########################################
#
#  Main program
#
########################################

do.main <- function(lstname) {
    # read a list of entries from file <lstname>.lst
    cat("For list", lstname, "\n")
    cat("-------------------------------\n")
    lstfile <- sprintf("%s.lst", lstname)
    entries <- read.entries(lstfile)

    # derive information
    infolist <- list()
    n <- 0
    for (e in entries) {
        distr <- get.distr(e)
        dname <- class(distr)[1]
        cat("On ", e, " --> ", dname, ": ", sep="")
        for (sn in distr$names) {
            cat(sn, "=", distr[[sn]], " ", sep="")
        }
        cat("\n")
        n <- n + 1
        info <- list(
            expr=e,
            distr=distr,
            points=eval.points(distr),
            quans=eval.quans(distr))
        infolist[[n]] <- info
    }

    # output
    outfile = sprintf("%s.ref.json", lstname)
    sink(outfile)
    cat("[\n")
    for (i in 1:n) {
        write.json(infolist[[i]], last=(i==n))
    }
    cat("]\n")
    sink()

    cat("\n")
}

# do.main("discrete_test")
do.main("continuous_test")
