
import rasterio 
from datetime import datetime
import pandas as pd
import numpy as np
import os
import shutil  
from datetime import datetime, timedelta

########Generate API maps USING 6 MONTHS OF RAINFALL

base_dir = r'D:\wildfire prediction\Near real time Wildfire\Data\Oahu\Rainfall_updated'
all_files = [os.path.join(base_dir, fname) for fname in os.listdir(base_dir)]

dfs_names = (os.listdir(base_dir))
dfs_names[0][8:-4]

start_date = datetime.strptime("2024-07-02", "%Y-%m-%d")
end_date = datetime.strptime("2024-09-07", "%Y-%m-%d")

date_list = pd.date_range(start_date, end_date, freq='D')
print(f"Creating list of dates starting from {start_date} to {end_date}")
print(date_list)

# if you want dates in string format then convert it into string
date_list=date_list.strftime("%Y_%m_%d")
print(date_list)

for j in range(len(all_files)):
    for i in range(len(date_list)):
        if dfs_names[j][-14:-4]==date_list[i]:
            print(j)
            with rasterio.open(all_files[j-1]) as src:
                result_array = src.read()
                result_profile = src.profile 

            # Add on the rest one at a time
            for f in all_files[j-180:j-1]:
                with rasterio.open(f) as src:
                    # Only sum the arrays if the profiles match. 
                    #assert result_profile == src.profile, 'stopping, file {} and  {} do not have matching profiles'.format(all_files[j], f)
                    result_array = result_array + src.read()
                    
            with rasterio.open('D:/wildfire prediction/Near real time Wildfire/Data/Oahu/API_updated/API%s.tif'%dfs_names[j][8:-4], 'w', **result_profile) as dst:
                    dst.write(result_array, indexes=[1])



#####GENERATE DAILY NDVI MAPS FROM 16-DAY NDVI MAPS

# start date
start_date = datetime.strptime("2002-01-01", "%Y-%m-%d")
end_date = datetime.strptime("2023-07-12", "%Y-%m-%d")


date_list = pd.date_range(start_date, end_date, freq='D')
print(f"Creating list of dates starting from {start_date} to {end_date}")
print(date_list)

# if you want dates in string format then convert it into string
date_list=date_list.strftime("%Y_%m_%d")
print(date_list)



#base_dir = r'D:/wildfire prediction/Near real time Wildfire/Data/Maui county/Maui county ndvi'
base_dir = r'D:/wildfire prediction/Near real time Wildfire/Data/Kauai/16-day ndvi'
#base_dir = r'D:\wildfire prediction\Near real time Wildfire\Data\Oahu\ndvi-oahu-hcdp'

fnames = [os.path.join(base_dir, fname) for fname in os.listdir(base_dir)]

dfs_names = (os.listdir(base_dir))
for j in range(len(fnames)):
    for i in range(len(date_list)):
        date_ndvi=dfs_names[j][-14:-4]
        image_date= datetime(int(date_ndvi[0:4]),int(date_ndvi[5:7]),int(date_ndvi[8:10]))
        day_ndvi=image_date.strftime("%Y_%m_%d")
        date_ndvi_16=dfs_names[j+1][-14:-4]
        image_date_16= datetime(int(date_ndvi_16[0:4]),int(date_ndvi_16[5:7]),int(date_ndvi_16[8:10]))
        day_ndvi_16=image_date_16.strftime("%Y_%m_%d")
        d1=datetime.strptime(day_ndvi, '%Y_%m_%d')
        d16=datetime.strptime(day_ndvi_16, '%Y_%m_%d')
        if (d16-d1).days==16:
            if date_list[i]==day_ndvi:
                with rasterio.open(fnames[j]) as src:
                    result_array = src.read()
                    result_profile = src.profile 
                        
                with rasterio.open('D:/wildfire prediction/Near real time Wildfire/Data/Kauai/daily ndvi/ndvi_Kauai_%s.tif'%date_list[i], 'w', **result_profile) as dst:
                        dst.write(result_array, indexes=[1])
                
                #shutil.copy(fnames[j],target )
            if (date_list[i]> day_ndvi) and (date_list[i]< day_ndvi_16) and (date_list[i][:4]==day_ndvi[:4]):
                dt1=datetime.strptime(date_list[i], '%Y_%m_%d')
                dt2=datetime.strptime(day_ndvi, '%Y_%m_%d')
                diff=(dt1-dt2).days
                with rasterio.open(fnames[j]) as src:
                    result_array = src.read()
                    result_profile = src.profile 

                # Add on the rest one at a time
                
                with rasterio.open(fnames[j+1]) as src:
                        # Only sum the arrays if the profiles match. 
                    assert result_profile == src.profile, 'stopping, file {} and  {} do not have matching profiles'.format(fnames[j], fnames[j+1])
                    result_array = result_array + (src.read()-result_array)*diff/17
                        
                with rasterio.open('D:/wildfire prediction/Near real time Wildfire/Data/Kauai/daily ndvi/ndvi_Kauai_%s.tif'%date_list[i], 'w', **result_profile) as dst:
                        dst.write(result_array, indexes=[1])
        else:
            if date_list[i]==day_ndvi:
                with rasterio.open(fnames[j]) as src:
                    result_array = src.read()
                    result_profile = src.profile 
                        
                with rasterio.open('D:/wildfire prediction/Near real time Wildfire/Data/Kauai/daily ndvi/ndvi_Kauai_%s.tif'%date_list[i], 'w', **result_profile) as dst:
                        dst.write(result_array, indexes=[1])
                
                #shutil.copy(fnames[j],target )
            if (date_list[i]> day_ndvi) and (date_list[i]< day_ndvi_16) and (date_list[i][:4]==day_ndvi[:4]):
                dt1=datetime.strptime(date_list[i], '%Y_%m_%d')
                dt2=datetime.strptime(day_ndvi, '%Y_%m_%d')
                diff=(dt1-dt2).days
                with rasterio.open(fnames[j]) as src:
                    result_array = src.read()
                    result_profile = src.profile 

                # Add on the rest one at a time
                
                with rasterio.open(fnames[j+1]) as src:
                        # Only sum the arrays if the profiles match. 
                    assert result_profile == src.profile, 'stopping, file {} and  {} do not have matching profiles'.format(fnames[j], fnames[j+1])
                    result_array = result_array + (src.read()-result_array)*diff/18
                        
                with rasterio.open('D:/wildfire prediction/Near real time Wildfire/Data/Kauai/daily ndvi/ndvi_Kauai_%s.tif'%date_list[i], 'w', **result_profile) as dst:
                        dst.write(result_array, indexes=[1])
            