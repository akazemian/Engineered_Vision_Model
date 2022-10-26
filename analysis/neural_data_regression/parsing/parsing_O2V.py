import xarray as xr
import numpy as np
import os

raw_data_path = '/data/shared/brainio/bonner-datasets'
save_path = '/data/atlas/neural_data/object2vec'
regions = ['roi_EVC','roi_LOC']


# saving each region from each subject separately 
for subject in range(4):
    print('subject',subject)
    
    da = xr.open_dataset(os.path.join(raw_data_path,f'bonner2021.object2vec-subject{subject}.nc'))
    # get data for shared images
        
    for region in regions:
        print(region)
        
        da = xr.open_dataset(os.path.join(raw_data_path,f'bonner2021.object2vec-subject{subject}.nc'))
        # get data for shared images


        # get region's voxels
        da_region = da.where((da[region] == True), drop=True)
        l = list(da_region.coords)

        # remove all other regions
        l.remove(region) # keep desired region
        l.remove('stimulus_id') # keep stimulus id
        da_region = da_region.drop(l) # drop other coords
        da_region = da_region.groupby('stimulus_id').mean()
        da_region = da_region.rename({f'bonner2021.object2vec-subject{subject}':'x'})
        da_region.to_netcdf(os.path.join(save_path,f'SUBJECT_{subject}_REGION_{region}'))