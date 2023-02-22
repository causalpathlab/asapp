library(asapR)

mtx.file = '/home/BCCRC.CA/ssubedi/project_data/data/pbmc/raw_filterd/matrix.mtx'

asap = fit.topic.asap(mtx.file,10)
