import rasterio as rio
from rasterio.merge import merge
from rasterio.plot import show
import glob
import os

# Import all fire risk Geotiff files

base_dir = r'D:/Globus_collection/Map_23_24'
fnames = [os.path.join(base_dir, fname) for fname in sorted(os.listdir(base_dir))]

dfs_names = sorted(os.listdir(base_dir))

#fnames[0][124:131]

dfs_names[0][-14:-4]


#k=0
for i in range(len (fnames)):
#for i in range(5970,len (fnames)):
    #i=k
    #Let’s first create an empty list for the datafiles that will be part of the mosaic.
    src_files_to_mosaic = []
    day=dfs_names[i][-14:-4]
    #year_day=dfs_names[i][-11:-4]
    for j in range (len(fnames)):
    #for j in range(5970,len (fnames)):
        # print(j)
        # print(fnames [j])
        if int(dfs_names[j][-14:-4])==int(day):
        #if int(dfs_names[j][-11:-4])==int(year_day):
            src = rio.open(fnames [j])
            src_files_to_mosaic.append(src)
    
    # It is important to have 4 images for each day otherwise mosaic will no be done correctly
    if len (src_files_to_mosaic)< 4:
        print ('error in number of images')
        #break
        continue
    
    print(len(src_files_to_mosaic))
    #k=j+1
    # Merge function returns a single mosaic array and the transformation info
    mosaic, out_trans = merge(src_files_to_mosaic)
    #show(mosaic, cmap='terrain')
    
    # Now we are ready to save our mosaic to disk
    # Copy the metadata
    out_meta = src.meta.copy()
    
    # Update the metadata
    
    out_meta.update({"driver": "GTiff","height": mosaic.shape[1],"width": mosaic.shape[2],"transform": out_trans})
    
    # Write the mosaic raster to disk
    # with rio.open(out_fp, "w", **out_meta) as dest: dest.write(mosaic)
    
    with rio.open(r'D:/Globus_collection/Mosaic_23_24/FR_hawaii_state_%s.tif'%day, 'w', **out_meta) as dst:
        dst.write(mosaic) 
    # with rio.open(r'D:/project/ndvi_hawaii/original ndvi/hawaii_state_mosaic/NDVI_16day_250m_%s.tif'%int(year_day), 'w', **out_meta) as dst:
    #     dst.write(mosaic) 

