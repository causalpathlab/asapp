library(asapR)
set.seed(42)

.rnorm <- function(d1,d2) matrix(rnorm(d1*d2), d1, d2)
.zero <- function(d1,d2) matrix(rep(0, d1*d2), d1, d2)

mtx.data <- fileset.list("temp")

ntypes <- 5
ngenes <- 10000
ncells <- 3000
nanchor <- 100 ## anchor genes 

.anchors <- NULL
.beta <- .zero(ngenes, ntypes)
for(tt in 1:ntypes){
    .anchor.t <- sample(ngenes, nanchor)
    .beta[.anchor.t,tt] <- .rnorm(nanchor, 1) * 2
    .anchors <- c(.anchors, .anchor.t)
}
.beta <- .beta + .rnorm(ngenes, ntypes)
.membership <- sample(ntypes, ncells, replace=TRUE)
.theta <- Matrix::spMatrix(nrow=ntypes, ncol=ncells, i = .membership, j = 1:ncells, x = rep(1, ncells))

X <- apply(.beta %*% .theta, 2, scale) * sqrt(.5) + .rnorm(ngenes, ncells) * sqrt(.5)
X <- round(Matrix::Matrix(exp(X), sparse=TRUE))

# must remove the index file if it exists
unlink("temp.mtx.gz")
unlink("temp.mtx.gz.index")
mtx.data <- write.sparse(X, 1:ngenes, 1:ncells, "temp")