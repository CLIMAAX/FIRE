"""Supporting functions for wildfire hazard assessment (ML approach)

A workflow from the CLIMAAX Handbook and FIRE GitHub repository.

https://handbook.climaax.eu/
https://github.com/CLIMAAX/FIRE

Contributors:
- Andrea Trucchia (Andrea.trucchia@cimafoundation.org)
- Farzad Ghasemiazma (Farzad.ghasemiazma@cimafoundation.org)
- Giorgio Meschi (Giorgio.meschi@cimafoundation.org)
"""

import os

from tqdm import tqdm
import numpy as np
from osgeo import gdal
import rasterio
from rasterio import features
from rasterio.plot import show
from scipy import signal
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def save_raster_as(array, output_file, reference_file, **kwargs):
    """Save a raster from a 2D numpy array using another raster as reference to get the spatial extent and projection.
    
    :param array: 2D numpy array with the data
    :param output_file: Path to the output raster
    :param reference_file: Path to a raster who's geotransform and projection will be used
    :param kwargs: Keyword arguments to be passed to rasterio.open when creating the output raster
    """
    with rasterio.open(reference_file) as f:
        profile = f.profile
        profile.update(**kwargs)
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(array.astype(profile['dtype']), 1)


def plot_raster_V2(raster, ref_arr, cmap='seismic', title='', figsize=(10, 8), dpi=300, outpath=None,
        array_classes=[], classes_colors=[], classes_names=[], shrink_legend=1, xy=(0.5, 1.1), labelsize=10,
        basemap=False, basemap_params = {'crs' : 'EPSG:4326', 'source' : None, 'alpha' : 0.5, 'zoom' : '11'},
        add_to_ax: tuple = None, plot_kwargs=None):
    '''Plot a raster object with possibility to add basemap and continuing to build upon the same ax.

    Example with discrete palette:
    array_classes = [-1, 0, 0.2, 0.4, 0.6, 0.8, 1],  # all the values including nodata
    classes_colors = ['#0bd1f700','#0bd1f8', '#1ff238', '#ea8d1b', '#dc1721', '#ff00ff'], # a color for each range
    classes_names = [ 'no data', 'Very Low', 'Low', 'Medium', 'High', 'Extreme'], # names

    add_to_ax: pass an axs to overlay other object to the same ax. it is a tuple (fig, ax)
    '''
    if plot_kwargs is None:
        plot_kwargs = {}
    
    if add_to_ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = add_to_ax[0]
        ax = add_to_ax[1]

    if len(array_classes) > 0 and len(classes_colors) > 0 and len(classes_names) > 0:
        cmap = mcolors.ListedColormap(classes_colors)
        norm = mcolors.BoundaryNorm(array_classes, cmap.N)

        # plot the raster
        f = show(np.where(ref_arr == -9999, np.nan,raster), ax=ax,
                cmap=cmap, norm=norm, interpolation='none', **plot_kwargs)

        img = f.get_images()[0]

        # trick to shift ticks labels in the center of each color
        cumulative = np.cumsum(array_classes, dtype = float)
        cumulative[2:] = cumulative[2:] - cumulative[:-2]
        ticks_postions_ = cumulative[2 - 1:]/2
        ticks_postions = []
        ticks_postions.extend(list(ticks_postions_))

        # plot colorbar
        cbar = fig.colorbar(img, boundaries=array_classes, ticks=ticks_postions, shrink = shrink_legend)
        cbar.ax.set_yticklabels(classes_names)
        cbar.ax.tick_params(labelsize = labelsize)
    else:
        # use imshow so that we have something to map the colorbar to
        image = show(np.where(ref_arr == -9999, np.nan,raster), ax=ax, cmap=cmap, **plot_kwargs)
        img = image.get_images()[0]
        cbar = fig.colorbar(img, ax=ax, shrink=shrink_legend)
        cbar.ax.tick_params(labelsize=labelsize)

    ax.set_xticks([])
    ax.set_yticks([])
    for s in ["top", 'bottom', "left", 'right']:
        ax.spines[s].set_visible(False)

    ax.annotate(title, xy=xy, xytext=xy, va='center', ha='center', xycoords='axes fraction',
            fontfamily='sans-serif', fontsize=12, fontweight='bold')

    if basemap:
        if basemap_params['source'] is None:
            ax.add_basemap(ax, crs=basemap_params['crs'], source=ax.providers.OpenStreetMap.Mapnik,
                    alpha=basemap_params['alpha'], zorder=-1)
        else:
            ax.add_basemap(ax, crs=basemap_params['crs'], source=basemap_params['source'],
                    alpha=basemap_params['alpha'], zorder=-1, zoom=basemap_params['zoom'])

    if outpath is not None:
        fig.savefig(outpath, dpi = dpi, bbox_inches='tight')

    return fig, ax


def save_raster_as_h(array, output_file, reference_file, **kwargs):
    """Save a raster from a 2D numpy array using another raster as reference to get the spatial extent and projection.
    
    :param array: 2D numpy array with the data
    :param output_file: Path to the output raster
    :param reference_file: Path to a raster who's geotransform and projection will be used
    :param kwargs: Keyword arguments to be passed to rasterio.open when creating the output raster
    """
    with rasterio.open(reference_file) as f:
        profile = f.profile
        profile.update(**kwargs)
        mask = array == profile['nodata']
        array  = np.ma.array(array, mask=mask)
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(array.astype(profile['dtype']), 1)


def process_dem(dem_path, output_folder, verbose=False):
    """Calculate slope and aspect from a DEM and save them to the output folder.
    
    :param dem_path: Path to the DEM
    :param output_folder: Path to the output folder
    :param verbose: If True, print some messages
    :return: Nothing
    """
    slope_path = os.path.join(output_folder, "slope.tif")
    aspect_path = os.path.join(output_folder, "aspect.tif")
    northing_path = os.path.join(output_folder, "northing.tif")
    easting_path = os.path.join(output_folder, "easting.tif")
    roughness_path = os.path.join(output_folder, "roughness.tif")
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    with rasterio.open(dem_path) as src:
        if verbose:
            print(f'Reading dem file {dem_path}')
        dem = src.read(1, masked  = True)
        if verbose:
            print(f'This is what {dem_path} looks like')
            plt.imshow(dem)
            plt.title('DEM')
            plt.colorbar(shrink = 0.5)
            plt.xticks([])
            plt.yticks([])
            plt.show()
            print('Calculating slope, aspect and roughness')
            gdal.DEMProcessing(slope_path, dem_path, 'slope')
            gdal.DEMProcessing(aspect_path, dem_path, 'aspect')
            gdal.DEMProcessing(roughness_path, dem_path, 'roughness')

    with rasterio.open(aspect_path) as f:
        if verbose:
            print('Calculating northing and easting files')
            print(f'Reading aspect file {aspect_path}')
        aspect = f.read(1,   masked = True)
        if verbose:
            print('Aspect looks like this...')
            plt.imshow(aspect)
            plt.title('Aspect')
            plt.colorbar(shrink = 0.5)
            plt.xticks([])
            plt.yticks([])
            plt.show()
        #aspect[aspect <= -9999] = np.NaN
    northing = np.cos(aspect * np.pi/180.0)
    print(f'Saving northing file {northing_path}')
    save_raster_as(northing, northing_path, aspect_path)
    del northing
    print(f'Saving easting file {easting_path}')
    easting = np.sin(aspect * np.pi/180.0)
    save_raster_as(easting, easting_path, aspect_path)
    del easting


def rasterize_numerical_feature(gdf, reference_file, column=None, verbose=True):
    """Rasterize a vector file using a reference raster to get the shape and the transform.
    
    :param gdf: GeoDataFrame with the vector data
    :param reference_file: Path to the reference raster
    :param column: Name of the column to rasterize. If None, it will rasterize the geometries
    :return: Rasterized version of the vector file
    """
    with rasterio.open(reference_file) as f:
        out = f.read(1,   masked = True)
        myshape = out.shape
        mytransform = f.transform #f. ...
    del out
    if verbose:
        print("Shape of the reference raster:", myshape)
        print("Transform of the reference raster:", mytransform)
    out_array = np.zeros(myshape)#   out.shape)
    # this is where we create a generator of geom, value pairs to use in rasterizing
    if column is not None:
        shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[column]))
    else:
        shapes = ((geom, 1) for geom in gdf.geometry)
    print("Features.rasterize is launched...")
    burned = features.rasterize(shapes=shapes, fill=np.NaN, out=out_array, transform=mytransform)#, all_touched=True)
    #    out.write_band(1, burned)
    print("Features.rasterize is done...")
    return burned


class MyRaster:
    """Handle the path and metadata of a raster
    
    dem_raster = MyRaster(dem_path, "dem")
    dem_raster.read_raster()
    slope_raster = MyRaster(slope_path, "slope")
    slope_raster.read_raster()
    northing_raster = MyRaster(aspect_path, "aspect")
    northing_raster.read_raster()
    easting_raster = MyRaster(easting_path, "easting")
    easting_raster.read_raster()
    roughness_raster = MyRaster(roughness_path, "roughness")
    dem_rasters = [dem_raster, slope_raster, northing_raster, easting_raster, roughness_raster]
    """
    def __init__(self, path, label):
        self.path = path
        self.label = label
        self.data = None
        self.mask = None
        self.nodata = None
        self.read_raster()

    def read_raster(self):
        with rasterio.open(self.path) as src:
            self.data = src.read(1, masked=True)
            self.mask = src.read_masks(1)
            self.nodata = src.nodata

    def set_data(self, data):
        self.data = data

    def get_data(self):
        return self.data

    def get_mask(self):
        return self.mask

    def get_nodata(self):
        return self.nodata

    def get_label(self):
        return self.label

    def get_path(self):
        return self.path

    def get_shape(self):
        return self.data.shape


def assemble_dem_dict(dem_paths, dem_labels):
    """Assemble the dictionary with all the rasters to be used in the model which are related to topography.
    
    :param dem_paths: paths to the dem and related rasters
    :param dem_labels: labels of the dem and related rasters
    :return: a dictionary with all the rasters to be used in the model
    usage example:
    dem_paths = [dem_path, slope_path, aspect_path, easting_path, northing_path, roughness_path]
    dem_labels = ["dem", "slope", "aspect", "easting", "northing", "roughness"]
    dem_dict = assemble_dem_dict(dem_paths, dem_labels)
    """
    # Create the dictionary
    dem_dict = {}
    for path, label in zip(dem_paths, dem_labels):
        dem_dict[label] = MyRaster(path, label)
        dem_dict[label].read_raster()

    return dem_dict


def assemble_veg_dictionary(veg_path, dem_path, verbose=False):
    """Assemble the dictionary with all the rasters to be used in the model which are related to vegetation.
    
    This comprises of:
    - the vegetation raster
    - the rasters of vegetation densities: a fuzzy counting of the neighboring vegetation for each type of vegetation
        See Tonini et al. 2020 for more details.

    :param veg_path: path to the vegetation raster.
    :return: a dictionary with all the veg rasters to be used in the model
    
    usage example:
    veg_dict = assemble_veg_dictionary(veg_path, dem_path, verbose = True)
    """
    # Create the dictionary
    veg_dict = {}
    veg_dict["veg"] = MyRaster(veg_path, "veg")
    veg_dict["veg"].read_raster()
    veg_arr = veg_dict["veg"].data.astype(int)
    dem_raster = MyRaster(dem_path, "dem")
    dem_raster.read_raster()
    dem_arr = dem_raster.data
    dem_nodata = dem_raster.nodata

    veg_mask = np.where(veg_arr == 0, 0, 1)
    # complete the mask selecting the points where also dem exists.
    mask = (veg_mask == 1) & (dem_arr != dem_nodata)

    # evaluation of perc just in vegetated area, non vegetated are grouped in code 0
    veg_int = veg_arr.astype(int)
    veg_int = np.where(mask == 1, veg_int, 0)
    window_size = 2
    types = np.unique(veg_int)
    # remove zero
    types = types[types != 0]
    if verbose:
        print("types of vegetation in the veg raster:", types)

    counter = np.ones((window_size*2+1, window_size*2+1))
    take_center = 1
    counter[window_size, window_size] = take_center
    counter = counter / np.sum(counter)

    # perc --> neighbouring vegetation generation
    for t in tqdm(types, desc="processing vegetation density"):
        density_entry = 'perc_' + str(int(t))
        if verbose:
            print(f'Processing vegetation density {density_entry}')
        temp_data = 100 * signal.convolve2d(veg_int==t, counter, boundary='fill', mode='same')
        temp_raster = MyRaster(dem_path, density_entry) # the path is dummy... I need just the other metadata.
        temp_raster.read_raster()
        temp_raster.set_data(temp_data)
        veg_dict[density_entry] = temp_raster

    return veg_dict, mask



def preprocessing(dem_dict, veg_dict, climate_dict, fires_raster, mask, verbose=True):
    """
    Usage:
    X_all, Y_all, columns = preprocessing(dem_dict, veg_dict, climate_dict, fires_raster, mask)
    """
    # creaate X and Y datasets
    n_pixels = len(dem_dict["dem"].data[mask])

    # the number of features is given by all the dem layers, all the veg layers, all the climate layers.
    # TODO: add the possibility to add other layers which belong to a "misc" category.
    n_features = len(dem_dict.keys())+ len(veg_dict.keys()) + len(climate_dict.keys())
    #create the dictionary with all the data
    data_dict = {**dem_dict, **veg_dict, **climate_dict}

    X_all = np.zeros((n_pixels, n_features), dtype=np.float32) # This is going to be big... Maybe use dask?
    Y_all = fires_raster.data[mask]

    if verbose:
        print('Creating dataset for RandomForestClassifier')
    columns = data_dict.keys()
    for col, k in tqdm(enumerate(data_dict), "processing columns"):
        if verbose:
            print(f'Processing column: {k}')
        data = data_dict[k]
        # data is a MyRaster object and data.data is the numpy array with the data
        X_all[:, col] = data.data[mask]

    return X_all, Y_all, columns


def prepare_sample(X_all, Y_all, percentage=0.1, max_depth=8, number_of_trees=50):
    """
    Usage:
    model, X_train, X_test, y_train, y_test = train(X_all, Y_all, percentage)
    
    parameters:
    X_all: the X dataset with the descriptive features
    Y_all: the Y dataset with the target variable (burned or not burned)
    percentage: the percentage of the dataset to be used for training
    max_depth: random forest parameter
    number_of_trees: random forest parameter
    """
    # filter df taking info in the burned points
    fires_rows = Y_all.data != 0
    print(f'Number of burned points: {np.sum(fires_rows)}')
    X_presence = X_all[fires_rows]

    # sampling training set
    print(' I am random sampling the dataset ')
    # reduction of burned points --> reduction of training points
    reduction = int((X_presence.shape[0]*percentage))
    print(f"reducted df points: {reduction} of {X_presence.shape[0]}")

    # sampling and update presences
    X_presence_indexes = np.random.choice(X_presence.shape[0], size=reduction, replace=False)

    X_presence = X_presence[X_presence_indexes, :]
    # select not burned points

    X_absence = X_all[~fires_rows] #why it is zero?
    print("X_absence.shape[0]", X_absence.shape[0])
    print("X_presence.shape[0]", X_presence.shape[0])
    X_absence_choices_indexes = np.random.choice(X_absence.shape[0], size=X_presence.shape[0], replace=False)

    X_pseudo_absence = X_absence[X_absence_choices_indexes, :]
    # create X and Y with same number of burned and not burned points
    X = np.concatenate([X_presence, X_pseudo_absence], axis=0)
    Y = np.concatenate([np.ones((X_presence.shape[0],)), np.zeros((X_presence.shape[0],))])
    # create training and testing df with random sampling
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    print(f'Running RF on data sample: {X_train.shape}')
    model  = RandomForestClassifier(n_estimators=number_of_trees, max_depth = max_depth, verbose = 2)

    return model, X_train, X_test, y_train, y_test


def fit_and_print_stats(model, X_train, y_train, X_test, y_test, columns):
    """Fit the model and prints the stats on the training and test datasets.
    
    Input:
    model: the model to fit
    X_train: the training dataset
    y_train: the training labels
    X_test: the test dataset
    y_test: the test labels
    columns: the columns of the dataset (list of strings that were the keys of the dictionary)
    
    example usage:
    fit_and_print_stats(model, X_train, y_train, X_test, y_test, columns)
    """
    # fit model
    model.fit(X_train, y_train)
    # stats on training df
    p_train = model.predict_proba(X_train)[:,1]

    auc_train = sklearn.metrics.roc_auc_score(y_train, p_train)
    print(f'AUC score on train: {auc_train:.2f}')

    # stats on test df
    p_test = model.predict_proba(X_test)[:,1]
    auc_test = sklearn.metrics.roc_auc_score(y_test, p_test)
    print(f'AUC score on test: {auc_test:.2f}')
    mse = sklearn.metrics.mean_squared_error(y_test, p_test)
    print(f'MSE: {mse:.2f}')
    p_test_binary = model.predict(X_test)
    accuracy = sklearn.metrics.accuracy_score(y_test, p_test_binary)
    print(f'accuracy: {accuracy:.2f}')

    # features impotance
    print('I am evaluating features importance')
    imp = model.feature_importances_

    perc_imp_list = []
    list_imp_noPerc = []

    # separate the perc featuers with the others
    for i,j in zip(columns, imp):
        if i.startswith('perc_'):
            perc_imp_list.append(j)
        else:
            list_imp_noPerc.append(j)

    # aggregate perc importances
    perc_imp = sum(perc_imp_list)
    # add the aggregated result
    list_imp_noPerc.append(perc_imp)

    # list of columns of interest
    cols = [col for col in columns if not col.startswith('perc_')]
    cols.append('perc')

    # print results
    print('importances')
    dict_imp = dict(zip(cols, list_imp_noPerc))
    dict_imp_sorted = {k: v for k, v in sorted(dict_imp.items(),
                                                key=lambda item: item[1],
                                                reverse=True)}
    for i in dict_imp_sorted:
        print(f'{i} : {round(dict_imp_sorted[i], 2)}')


### Functions to define hazard ###

def corine_to_fuel_type(corine_codes_array, converter_dict, visualize_result = False):
    """Convert the corine land cover raster to a raster with the fuel types.
    
    The fuel types are defined in the converter_dict dictionary.
    """
    # Use float type to allow for NaN fill for unsuccessful lookups
    converted_band = np.vectorize(converter_dict.get, otypes=[np.float64])(corine_codes_array)
    # Fill with int-compatible placeholder value and convert to int for output
    converted_band = np.nan_to_num(converted_band, nan=-1)
    converted_band = converted_band.astype(int)
    if visualize_result:
        plt.matshow(converted_band)
        # discrete colorbar
        plt.colorbar(ticks=[0, 1, 2, 3, 4, 5, 6])
    return converted_band


def susc_classes( susc_arr, quantiles):
    '''Take a raster map and a list of quantiles and returns a categorical raster map related to the quantile classes.
    
    Parameters:
    susc_arr: the susceptibility array
    quantiles: the quantiles to use to create the classes (see np.digitize documentation)
    '''
    bounds = list(quantiles)
    # Convert the raster map into a categorical map based on quantile values
    out_arr = np.digitize(susc_arr, bounds, right=True)
    out_arr = out_arr.astype(np.int8)
    return out_arr


def contigency_matrix_on_array(xarr, yarr, xymatrix, nodatax, nodatay):
    '''
    xarr: 2D array, rows entry of contingency matrix
    yarr: 2D array, cols entry of contingency matrix
    xymatrix: 2D array, contingency matrix
    nodatax1: value for no data in xarr : if your array has nodata = np.nan >> nodatax or nodatay has to be 1
    nodatax2: value for no data in yarr : if your array has nodata = np.nan >> nodatax or nodatay has to be 1
    '''
    # if arr have nan, mask it with lowest class
    xarr = np.where(np.isnan(xarr)==True , 1, xarr)
    yarr = np.where(np.isnan(yarr)==True , 1, yarr)
    # convert to int
    xarr = xarr.astype(int)
    yarr = yarr.astype(int)

    mask = np.where(((xarr == nodatax) | (yarr ==nodatay)), 0, 1)

    # put lowest class in place of no data
    yarr[~mask] = 1
    xarr[~mask] = 1

    # apply contingency matrix
    output = xymatrix[ xarr - 1, yarr - 1]
    # mask out no data
    output[~mask] = 0
    return output
