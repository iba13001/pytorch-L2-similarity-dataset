# this class is originally designed to prepare data for a scaled L2 (Euclidean) similarity training on pytorch.

from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
  # src_file: input dataframe. Organize the data into category, products (e.g. shoes is the category, and product_1, product_2, etc are the products, the first two columns), continuous attributes (e.g. price, views etc) and categorical attributes (e.g. style, color, etc.)
  # cont_shape: number of continuous features.
  # weight: array of weights assigned to each feature.
  
    def __init__(self, src_file, cont_shape, num_rows=None, weights=None):
        tmp_all = src_file.to_numpy()
        scaler = preprocessing.MinMaxScaler()

      # transform cont features, and append all features together in x_data
        if cont_shape != 0:
            self.x_cont_data = scaler.fit_transform(tmp_all[:,2:(cont_shape + 2)].astype(float)) * weights
            self.x_cat_data = tmp_all[:,(cont_shape + 2):] # Cat and products
            self.x_data = np.append(self.x_cont_data, self.x_cat_data, axis = 1)
        else:
            self.x_cat_data = tmp_all[:,(cont_shape + 2):]
            self.x_data = self.x_cat_data
        
        self.cat_id_data = tmp_all[:,0].astype(str)
        self.prod_data = tmp_all[:,1].astype(str)
        self.cat_id_prod_data = np.transpose(np.array([self.cat_id_data, self.prod_data]))

  # get length of dataset
    def __len__(self):
        return len(self.x_data)

  # get all category-prod combinations
    def __get_all_groups__(self):
        #preds = self.x_data[idx]
        #id = self.id_data[idx]  # np.str
        all_samples = {}
        for i, cat_id_prod in enumerate(self.cat_id_prod_data):
            all_samples[cat_id_prod[0],cat_id_prod[1]] = self.x_data[i]
            #all_samples[cat_id_prod[0]][cat_id_prod[1]] = self.x_data[i]
            
        return all_samples

  # get only one specific cat-prod
    def __get_group__(self, group_id):
        all_samples = self.__get_all_groups__()
        prods = [key[1] for  key, value in all_samples.items() if  group_id in key]
        attrs = np.array([value for  key, value in all_samples.items() if  group_id in key]).astype(float)
        
        return prods,attrs

  # transform a group into a tensor  
    def group_to_tensor(self, group_id):
        all_samples = self.__get_all_groups__()
        prods,attrs = self.__get_group__(group_id)
        
        return prods, torch.tensor(attrs)


# EXAMPLE
# CONVERT ALL_GRPS DF TO A DATASET
dataset = CustomDataset(all_grps, cont_shape_StylePrice, weights = weights)
# CALCULATE L2 SIMILARITy (THIS COULD BE PARALLIZED ON CPUs OR GPUs instead)
for cat_id in cat_ids:
    brands, attrs = dataset.group_to_tensor(cat_id)
    sim = torch.cdist(attrs, attrs, p=2)
    df = pd.DataFrame(sim, index = brands, columns = brands)
    print(cat_id)









