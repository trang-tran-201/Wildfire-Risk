import time
start= time.time()
import pandas as pd
import numpy as np
data= pd.read_csv ("D:/wildfire prediction/Near real time Wildfire/Data/Oahu/Oahu_FireData-2002-2019.csv")
df=data.replace([-32768,-9999], np.nan)   
df=df.dropna(ignore_index=True)

df.columns

X = df.iloc[:,:-1]
Y = df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=0)
import numpy as np

from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(random_state=1)
RF.fit(X_train, y_train)

probRF=RF.predict_proba(X_test)[::,1] 

###Evaulation of model performance####

from sklearn.metrics import roc_auc_score, confusion_matrix

def calculate_metrics(y_true, y_pred_probs, threshold=0.5):
    # Convert probabilities to binary predictions based on the threshold
    y_pred = (y_pred_probs >= threshold).astype(int)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate sensitivity (True Positive Rate)
    sensitivity = tp / (tp + fn)

    
    # Calculate specificity (True Negative Rate)
    specificity = tn / (tn + fp)
    
    # Calculate NPV (Negative Predictive Value)
    npv = tn / (tn + fn)
    
    # Calculate PPV (Positive Predictive Value)
    ppv = tp / (tp + fp)
    
    # Calculate AUC (Area Under the Curve)
    auc = roc_auc_score(y_true, y_pred_probs)
    
    return sensitivity, specificity, npv, ppv, auc

sensitivity, specificity, npv, ppv, auc = calculate_metrics( y_test, probRF)
print(sensitivity)
print(specificity)
print(npv)
print(ppv)
print(auc) 

import matplotlib.pyplot as plt
import rasterio as rio
#print(rio.__version__)
import os, shutil
###Open any rainfall map of Oahu to get the coordinates of the island
ds= rio.open('D:/wildfire prediction/Near real time Wildfire/Data/Oahu/Rainfall_updated/rainfall_new_day_oa_data_map_2010_01_01.tif')

a = ds.read(1) #Read as a numpy array
ref_geotiff_meta = ds.profile
ref_geotiff_pixelSizeX, ref_geotiff_pixelSizeY = ds.res

height, width = a.shape #Find the height and width of the array

#Two arrays with the same shape as the input array/raster, where each value is the x or y index of that cell
cols, rows = np.meshgrid(np.arange(width), np.arange(height)) 

#Two arrays with the same shape as the input array/raster, where each value is the x or y coordinate of that cell 
xs, ys = rio.transform.xy(ds.transform, rows, cols) 

#They are actually lists, convert them to arrays
x_coordinates = np.array(xs)
y_coordinates = np.array(ys)


###Create a date list on which you want to genetate fire map
from datetime import datetime, timedelta
# start date
start_date = datetime.strptime("2024-08-14", "%Y-%m-%d")
end_date = datetime.strptime("2024-08-31", "%Y-%m-%d")
#end_date = datetime.strptime("2023-07-12", "%Y-%m-%d")
date_list = pd.date_range(start_date, end_date, freq='D')
print(f"Creating list of dates starting from {start_date} to {end_date}")
print(date_list)

date_list=date_list.strftime("%Y_%m_%d")
print(date_list)
for v in date_list:
    print(v)
    
    base_dir = r'D:/wildfire prediction/Near real time Wildfire/Data/Oahu/relative_humidity'

    fnames = [os.path.join(base_dir, fname) for fname in os.listdir(base_dir)]

    dfs_names = (os.listdir(base_dir))

    lon=[]
    lat=[]
    RH=[]
    for j in range(len(dfs_names)):
        if dfs_names[j][-14:-4]==str(v):
            print(dfs_names[j])
            Rh_raster=rio.open(fnames[j-1], "r")
            for i in range (x_coordinates.shape[0]):
                #print(i)
                for j in range (x_coordinates.shape[1]):
                    for val in Rh_raster.sample([(x_coordinates[i,j], y_coordinates[i,j])]):
                        #print(val)
                        RH.append(val)
                        lon.append(x_coordinates[i,j])
                        lat.append(y_coordinates[i,j])

        
    df=pd.DataFrame(RH,columns=['RH'])

    df['lon'] = np.array(lon)
    df['lat'] = np.array(lat)
    df.dropna()
    ###########EXTRACT fractional landcover FOR FIRE POINTS

    path='D:/wildfire prediction/Near real time Wildfire/Data/Oahu/perCov2016model_oa.tif'
    #path='D:/wildfire prediction/Near real time Wildfire/Data/Maui county/perCov2016model_ma.tif'

    LC_band1=[]
    LC_band2=[]
    LC_band3=[]
    with rio.open(path) as lc_raster:
        band1 = lc_raster.read(1) 
        band2 = lc_raster.read(2)
        band3 = lc_raster.read(3) 
        for i in range (x_coordinates.shape[0]):
            for j in range (x_coordinates.shape[1]):
                m, n = lc_raster.index(x_coordinates[i,j], y_coordinates[i,j])
                LC_band1.append(band1[m, n])
                LC_band2.append(band2[m, n])
                LC_band3.append(band3[m, n])
                
    df['LC_band1'] = np.array(LC_band1)
    df['LC_band2'] = np.array(LC_band2)
    df['LC_band3'] = np.array(LC_band3) 

    df=df.replace([-32768], np.nan)  
 
    ##EXTRACT CLIMATE DATA FOR A PARTICULAR DAY AND ESTIMATE PROBABILITY OF WILDFIRE ON THAT DAY

    base_dir = r'D:/wildfire prediction/Near real time Wildfire/Data/Oahu/temperature'

    fnames = [os.path.join(base_dir, fname) for fname in os.listdir(base_dir)]

    dfs_names = (os.listdir(base_dir))

    Tmax=[]
    for j in range(len(dfs_names)):
        if dfs_names[j][-14:-4]==str(v):
            print(dfs_names[j])
            Tmax_raster=rio.open(fnames[j-1], "r")
            for i in range (x_coordinates.shape[0]):
                #print(i)
                for j in range (x_coordinates.shape[1]):
                    for val in Tmax_raster.sample([(x_coordinates[i,j], y_coordinates[i,j])]):
                        #print(val)
                        Tmax.append(val)
                
    df['Tmax'] = np.array(Tmax)
    df=df.replace([-9999], np.nan)   
    df.dropna()

    ##EXTRACT CLIMATE DATA FOR A PARTICULAR DAY AND ESTIMATE PROBABILITY OF WILDFIRE ON THAT DAY
    from datetime import datetime, timedelta

    base_dir = r'D:/wildfire prediction/Near real time Wildfire/Data/Oahu/daily ndvi'

    fnames = [os.path.join(base_dir, fname) for fname in os.listdir(base_dir)]

    dfs_names = (os.listdir(base_dir))

    NDVI=[]
    for j in range(len(dfs_names)):
        if dfs_names[j][-14:-4]==str(v):
            print(dfs_names[j])
            NDVI_raster=rio.open(fnames[j-1], "r")
            for i in range (x_coordinates.shape[0]):
                #print(i)
                for j in range (x_coordinates.shape[1]):
                    for val in NDVI_raster.sample([(x_coordinates[i,j], y_coordinates[i,j])]):
                        #print(val)
                        NDVI.append(val)
                
    df['NDVI'] = np.array(NDVI)
    df=df.replace([-3.3999999521443642e+38], np.nan)   
    df.dropna()


    ##EXTRACT CLIMATE DATA FOR A PARTICULAR DAY AND ESTIMATE PROBABILITY OF WILDFIRE ON THAT DAY

    base_dir = r'D:/wildfire prediction/Near real time Wildfire/Data/Oahu/Rainfall_updated'

    fnames = [os.path.join(base_dir, fname) for fname in os.listdir(base_dir)]

    dfs_names = (os.listdir(base_dir))

    Pre=[]
    for j in range(len(dfs_names)):
        if dfs_names[j][-14:-4]==str(v):
            print(dfs_names[j])
            Pre_raster=rio.open(fnames[j-1], "r")
            pre_meta =Pre_raster.profile
            for i in range (x_coordinates.shape[0]):
                #print(i)
                for j in range (x_coordinates.shape[1]):
                    for val in Pre_raster.sample([(x_coordinates[i,j], y_coordinates[i,j])]):
                        #print(val)
                        Pre.append(val)

    df['Precipitation'] = np.array(Pre)


    df=df.replace([-3.3999999521443642e+38], np.nan) 

      
    df.dropna()

    df[df.isnull().any(axis=1)]

    #df3=pd.concat([df2, df1])

    base_dir = r'D:/wildfire prediction/Near real time Wildfire/Data/Oahu/API_updated'

    fnames = [os.path.join(base_dir, fname) for fname in os.listdir(base_dir)]

    dfs_names = (os.listdir(base_dir))


    API=[]
    for j in range(len(dfs_names)):
        if dfs_names[j][-14:-4]==str(v):
            print(dfs_names[j])
            API_raster=rio.open(fnames[j-1], "r")
            pre_meta =API_raster.profile
            for i in range (x_coordinates.shape[0]):
                #print(i)
                for j in range (x_coordinates.shape[1]):
                    for val in API_raster.sample([(x_coordinates[i,j], y_coordinates[i,j])]):
                        #print(val)
                        API.append(val)

    df['API'] = np.array(API)
    df=df.replace([-3.3999999521443642e+38], np.nan) 
    
    df2=df.dropna()
    df2
    fire=df2.drop(columns=['lon', 'lat'])
    fire['class'] = 0
    fireX = fire.iloc[:,:-1]
    fireY = fire.iloc[:,-1]

    probRF=RF.predict_proba(fireX)[::,1] 


    prob=pd.DataFrame(probRF,columns=['prob'], index=df2.index) 


    pro_concat=prob.loc[:,'prob']-(df['lat']/100000000000)


    prob_final=pd.DataFrame(pro_concat,columns=['prob'])


    bbox=ds.bounds
    bbox
    latitude=(bbox[1],bbox[3])
    longitude=(bbox[0],bbox[2])
    bbox = ((bbox[0],   bbox[2],      
         bbox[1], bbox[3]))
    bbox 


            
    Prob_2d_clf=np.reshape(prob_final['prob'].values,(a.shape[0],a.shape[1]))
        
    import matplotlib.pyplot as plt            

    from matplotlib import cm
    from matplotlib.colors import ListedColormap,LinearSegmentedColormap
    # modified hsv in 256 color class
    hsv_modified = cm.get_cmap('nipy_spectral', 256)# create new hsv colormaps in range of 0.3 (green) to 0.7 (blue)
    newcmp = ListedColormap(hsv_modified(np.linspace(0.5, 0.9, 256)))# show figure
    fig, ax = plt.subplots(figsize=(8,6), dpi=400)
    fig.patch.set_alpha(0)
    plt.imshow(Prob_2d_clf, extent=bbox, cmap=newcmp, zorder=1)
    im_ratio = Prob_2d_clf.shape[0]/Prob_2d_clf.shape[1]
    cbar=plt.colorbar(fraction=0.047*im_ratio)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Fire risk', fontsize=18)
    #plt.colorbar(label='fire risk')
    plt.clim(0,1)
    plt.title(str(v), size=24)
    plt.xlabel('Longitude', size=18)
    plt.ylabel('Latitude', size=18)
    plt.tight_layout()        
    fig.savefig('D:/wildfire prediction/Near real time Wildfire/Data/Oahu/FR map png/FR_Oahu_%s.png'%str(v))          

    ref_geotiff_meta['dtype'] = "float64"
    with rio.open('D:/wildfire prediction/Near real time Wildfire/Data/Oahu/FR map final/FR_Oahu_%s.tif'%str(v), 'w', **ref_geotiff_meta) as dst:
        dst.write(Prob_2d_clf, 1)         

    
