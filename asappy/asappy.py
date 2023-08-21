from .dutil import DataSet

class asap(object):
	def __init__(self,adata : DataSet):
		self.adata = adata
	
	def get_params(self):
		model_params = {}
		self.adata.uns['shape'] = list(self.adata.uns['shape'])
		model_params['tree_depth'] = self.tree_depth
		model_params['num_factors'] = self.num_factors
		model_params['downsample_pseudobulk'] = self.downsample_pseudobulk
		model_params['downsample_size'] = self.downsample_size
		model_params['number_batches'] = self.number_batches
		return model_params	
        