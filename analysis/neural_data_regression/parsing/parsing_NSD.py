import xarray as xr
import numpy as np

# code for saving NSD neural responses from shared images per subject and per region. The output is an xarray with dims: responses (x) and presentation (stimulus_id)




save_path = '/data/atlas/neural_data/naturalscenes'
raw_data_path = '/data/shared/brainio/bonner-datasets/old'
regions = ['roi_prf-visualrois_V1v','roi_prf-visualrois_V2v','roi_prf-visualrois_V3v','roi_prf-visualrois_hV4']


# get shared stimulus ids
l = []
for subject in range(8):
    l.append(set(xr.open_dataset(os.path.join(raw_data_path,f'allen2021.natural-scenes-subject{subject}.nc')).stimulus_id.values))
shared_ids = set.intersection(*l) 



# saving each region from each subject separately 
for subject in range(8):
    print('subject',subject)
    
    da = xr.open_dataset(os.path.join(raw_data_path,f'allen2021.natural-scenes-subject{subject}.nc')
    # get data for shared images
    da = da.where(da.stimulus_id.isin(list(shared_ids)),drop=True)
        
    for region in regions:
        print(region)
        
        # get region's voxels
        da_region = da.where((da[region] == True), drop=True)
        l = list(da_region.coords)

        # remove all other regions
        l.remove(region) # keep desired region
        l.remove('stimulus_id') # keep stimulus id
        da_region = da_region.drop(l) # drop other coords

        # get the average voxel response per image for region's voxels
        da_region = da_region.groupby('stimulus_id').mean()
        da_region = da_region.rename({f'allen2021.natural-scenes-subject{subject}':'x'})
        da_region.to_netcdf(os.path.join(save_path,f'SUBJECT_{subject}_REGION_{region}'))