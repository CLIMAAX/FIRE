{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Fire Hazard Notebook\n",
    "\n",
    "This Jupyter Notebook is used to analyze fire hazard data and perform various preprocessing and analysis tasks.\n",
    "\n",
    "## Table of Contents\n",
    "1. Introduction\n",
    "2. Data Preparation\n",
    "3. Data Analysis\n",
    "4. Results\n",
    "5. Conclusion\n",
    "6. Bibliography\n",
    "\n",
    "## 1. Introduction\n",
    "In this section, we will provide an overview of the project and the goals of the analysis.\n",
    "The analysis is based on the hazard mapping works of Tonini et al 2020, Trucchia et al. 2023.\n",
    "\n",
    "The workflow is based on the following steps:\n",
    "\n",
    "- Data gathering and preprocessing.\n",
    "- Building a model for wildfire susceptibility using present climate conditions and synoptic wildfire events. \n",
    "- Projecting the model to future climate conditions. \n",
    "- For both cases, susceptibility can be evolved to hazard by considering the different plant functional types, which are a proxy for the intensity of potential wildfires. See Trucchia et al. 2023 for more details. \n",
    "- In the next weeks, we will also include the damage assessment for infrastructure and exposed elements in order to get risk maps.\n",
    "- Regarding climate, the analysis revolves around a  High-resolution gridded climate data for Europe based on bias-corrected EURO-CORDEX: the ECLIPS-2.0 dataset. ECLIPS (European CLimate Index ProjectionS) dataset contains gridded data for 80 annual, seasonal, and monthly climate variables for two past (1961-1990, 1991-2010) and five future periods (2011-2020, 2021-2140, 2041-2060, 2061-2080, 2081-2100). The future data are based on five Regional Climate Models (RCMs)driven by two greenhouse gas concentration scenarios, RCP 4.5 and 8.5. See Debojyoti et al. 2020 for more details.\n",
    "\n",
    "## 2. Data Preparation\n",
    "This section will cover the steps taken to prepare the data for analysis, including importing libraries, loading datasets, and preprocessing steps.\n",
    "\n",
    "At the present stage, the analysis case study is the Catalonia region in Spain. The data is stored in the `data` folder. \n",
    "\n",
    "Most of the analysis is based on raster calculations. The \"base\" raster is the clipped dem file, which has been clipped using the extent of the Catalonia adm shapefile.  The raster is metric, using the EPSG:3035 projection, with 100m resolution, and with extent given by:  3488731.355 1986586.650 3769731.355 2241986.650.  \n",
    "\n",
    "\n",
    "\n",
    "### References\n",
    "\n",
    "- Tonini, M.; D’Andrea, M.; Biondi, G.; Degli Esposti, S.; Trucchia, A.; Fiorucci, P. A Machine Learning-Based Approach for Wildfire Susceptibility Mapping. The Case Study of the Liguria Region in Italy. Geosciences 2020, 10, 105. [https://doi.org/10.3390/geosciences10030105](https://doi.org/10.3390/geosciences10030105)\n",
    "\n",
    "- Trucchia, A.; Meschi, G.; Fiorucci, P.; Gollini, A.; Negro, D. Defining Wildfire Susceptibility Maps in Italy for Understanding Seasonal Wildfire Regimes at the National Level. Fire 2022, 5, 30. [https://doi.org/10.1071/WF22138](https://doi.org/10.3390/fire5010030)\n",
    "\n",
    "- Trucchia, A.; Meschi, G.; Fiorucci, P.; Provenzale, A.; Tonini, M.; Pernice, U.  Wildfire hazard mapping in the eastern Mediterranean landscape. International Journal of Wildland Fire 2023, 32, 417-434. [https://doi.org/10.1071/WF22138](https://doi.org/10.1071/WF22138)\n",
    "\n",
    "- Chakraborty Debojyoti, Dobor Laura, Zolles Anita, Hlásny Tomáš, & Schueler Silvio. (2020). High-resolution gridded climate data for Europe based on bias-corrected EURO-CORDEX: the ECLIPS-2.0 dataset [Data set]. Zenodo. [https://doi.org/10.5281/zenodo.3952159] (https://doi.org/10.5281/zenodo.3952159)\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# How to download the raw climatic data? \n",
    "# try to use the following command:\n",
    "# !zenodo_get https://doi.org/10.5281/zenodo.3952159"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "import rasterio\n",
    "from rasterio.warp import calculate_default_transform, reproject, Resampling\n",
    "import rasterio.plot\n",
    "import os\n",
    "from tqdm import tqdm\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "from osgeo import gdal, ogr\n",
    "import geopandas as gpd \n",
    "import pandas as pd\n",
    "from scipy import signal\n",
    "import sklearn \n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "\n",
    "\n",
    "#Paths to data (mostly raster files and shapefiles)\n",
    "\n",
    "ECLIPS2p0_path = \"/share/ander/Dev/climaax/ECLIPS2.0/\"\n",
    "dem_path = \"/share/ander/Dev/climaax/data_cat/hazard/input_hazard/dem2_3035.tif\"\n",
    "dem_path_clip = \"/share/ander/Dev/climaax/data_cat/hazard/input_hazard/dem_3035_clip.tif\"\n",
    "slope_path = \"/share/ander/Dev/climaax/dem_processing/slope.tif\"\n",
    "aspect_path = \"/share/ander/Dev/climaax/dem_processing/aspect.tif\"\n",
    "easting_path = \"/share/ander/Dev/climaax/dem_processing/easting.tif\"\n",
    "northing_path = \"/share/ander/Dev/climaax/dem_processing/northing.tif\"\n",
    "roughness_path = \"/share/ander/Dev/climaax/dem_processing/roughness.tif\"\n",
    "\n",
    "clc_path = \"/share/ander/Dev/climaax/data_cat/hazard/input_hazard/veg_corine_reclass.tif\"\n",
    "clc_path_clip = \"/share/ander/Dev/climaax/data_cat/hazard/input_hazard/veg_corine_reclass_clip.tif\"\n",
    "clc_path_clip_nb = \"/share/ander/Dev/climaax/data_cat/hazard/input_hazard/veg_corine_reclass_clip_nb.tif\"\n",
    "fires_raster_path = \"/share/ander/Dev/climaax/data_cat/hazard/input_hazard/fires_raster.tif\"\n",
    "\n",
    "res_clim_dir_path = \"/share/ander/Dev/climaax/resized_climate/\"\n",
    "folder_1961_1990 = ECLIPS2p0_path+\"ECLIPS2.0_196190/\"\n",
    "folder_1991_2010 = ECLIPS2p0_path+\"ECLPS2.0_199110/\"\n",
    "folder_rcp45 = ECLIPS2p0_path+\"ECLIPS2.0_RCP45/\"\n",
    "folder_rcp85 = ECLIPS2p0_path+\"ECLIPS2.0_RCP85/\"\n",
    "CLMcom_CCLM_4p5_path = folder_rcp45 +\"CLMcom_CCLM_4.5/\"\n",
    "\n",
    "\"\"\"\n",
    "The data I want is the following set of variables:\n",
    "\n",
    "- MWMT, Mean warmest month temperature\n",
    "- TD, Continentality\n",
    "- AHM, Annual Heat-Moisture Index\n",
    "- SHM, Summer Heat-Moisture Index\n",
    "- DDbelow0, Degree-days below 0°C\n",
    "- DDabove18, Degree-days above 18°C\n",
    "- MAT, Annual mean temperaure\n",
    "- MAP, Annual total precipitation\n",
    "- Tave_sm, Mean summer temperature\n",
    "- Tmax_sm, Maximum summer temperature\n",
    "- PPT_at, Mean autumn precipitation\n",
    "- PPT_sm, Mean summer precipitation\n",
    "- PPt_sp, Mean spring precipitation\n",
    "- PPT_wt, Mean winter precipitation\n",
    "\"\"\"\n",
    "\n",
    "# names of climate variables and name of all the raster of climate variables\n",
    "var_names = [\"MWMT\", \"TD\", \"AHM\", \"SHM\", \"DDbelow0\", \"DDabove18\", \"MAT\", \"MAP\", \"Tave_sm\", \"Tmax_sm\", \"PPT_at\", \"PPT_sm\", \"PPT_sp\", \"PPT_wt\"]\n",
    "# In the following we have the file names of the climate variables for all the time periods \n",
    "f_rcp45_2011_2020 = [ CLMcom_CCLM_4p5_path + vv + \"_201120.tif\" for vv in var_names ]\n",
    "f_rcp45_2021_2040 = [CLMcom_CCLM_4p5_path +  vv + \"_202140.tif\" for vv in var_names ]\n",
    "f_hist_1961_1990 = [ folder_1961_1990 + vv + \"_196190.tif\" for vv in var_names ]\n",
    "f_hist_1991_2010 = [folder_1991_2010 + vv + \"_199110.tif\" for vv in var_names ]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Processing rcp45_2011_2020:   0%|          | 0/14 [00:00<?, ?it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Creating output file that is 2810P x 2554L.\n",
      "Processing /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/MWMT_201120.tif [1/1] : 0Using internal nodata values (e.g. -3.4e+38) for image /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/MWMT_201120.tif.\n",
      "Copying nodata values from source /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/MWMT_201120.tif to destination ./resized_climate/rcp45_2011_2020/MWMT_201120.tif.\n",
      "...10...20...30...40...50...60...70...80...90..."
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Processing rcp45_2011_2020:   7%|▋         | 1/14 [00:01<00:18,  1.45s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "100 - done.\n",
      "Creating output file that is 2810P x 2554L.\n",
      "Processing /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/TD_201120.tif [1/1] : 0Using internal nodata values (e.g. -3.4e+38) for image /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/TD_201120.tif.\n",
      "Copying nodata values from source /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/TD_201120.tif to destination ./resized_climate/rcp45_2011_2020/TD_201120.tif.\n",
      "...10...20...30...40...50...60...70...80...90..."
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Processing rcp45_2011_2020:  14%|█▍        | 2/14 [00:02<00:14,  1.21s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "100 - done.\n",
      "Creating output file that is 2810P x 2554L.\n",
      "Processing /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/AHM_201120.tif [1/1] : 0Using internal nodata values (e.g. -3.4e+38) for image /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/AHM_201120.tif.\n",
      "Copying nodata values from source /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/AHM_201120.tif to destination ./resized_climate/rcp45_2011_2020/AHM_201120.tif.\n",
      "...10...20...30...40...50...60...70...80...90..."
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Processing rcp45_2011_2020:  21%|██▏       | 3/14 [00:03<00:12,  1.15s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "100 - done.\n",
      "Creating output file that is 2810P x 2554L.\n",
      "Processing /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/SHM_201120.tif [1/1] : 0Using internal nodata values (e.g. -3.4e+38) for image /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/SHM_201120.tif.\n",
      "Copying nodata values from source /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/SHM_201120.tif to destination ./resized_climate/rcp45_2011_2020/SHM_201120.tif.\n",
      "...10...20...30...40...50...60...70...80...90..."
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Processing rcp45_2011_2020:  29%|██▊       | 4/14 [00:04<00:11,  1.12s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "100 - done.\n",
      "Creating output file that is 2810P x 2554L.\n",
      "Processing /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/DDbelow0_201120.tif [1/1] : 0Using internal nodata values (e.g. -3.4e+38) for image /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/DDbelow0_201120.tif.\n",
      "Copying nodata values from source /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/DDbelow0_201120.tif to destination ./resized_climate/rcp45_2011_2020/DDbelow0_201120.tif.\n",
      "...10...20...30...40...50...60...70...80...90..."
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Processing rcp45_2011_2020:  36%|███▌      | 5/14 [00:05<00:09,  1.11s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "100 - done.\n",
      "Creating output file that is 2810P x 2554L.\n",
      "Processing /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/DDabove18_201120.tif [1/1] : 0Using internal nodata values (e.g. -3.4e+38) for image /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/DDabove18_201120.tif.\n",
      "Copying nodata values from source /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/DDabove18_201120.tif to destination ./resized_climate/rcp45_2011_2020/DDabove18_201120.tif.\n",
      "...10...20...30...40...50...60...70...80...90..."
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Processing rcp45_2011_2020:  43%|████▎     | 6/14 [00:06<00:08,  1.10s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "100 - done.\n",
      "Creating output file that is 2810P x 2554L.\n",
      "Processing /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/MAT_201120.tif [1/1] : 0Using internal nodata values (e.g. -3.4e+38) for image /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/MAT_201120.tif.\n",
      "Copying nodata values from source /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/MAT_201120.tif to destination ./resized_climate/rcp45_2011_2020/MAT_201120.tif.\n",
      "...10...20...30...40...50...60...70...80...90..."
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Processing rcp45_2011_2020:  50%|█████     | 7/14 [00:07<00:07,  1.07s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "100 - done.\n",
      "Creating output file that is 2810P x 2554L.\n",
      "Processing /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/MAP_201120.tif [1/1] : 0Using internal nodata values (e.g. -3.4e+38) for image /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/MAP_201120.tif.\n",
      "Copying nodata values from source /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/MAP_201120.tif to destination ./resized_climate/rcp45_2011_2020/MAP_201120.tif.\n",
      "...10...20...30...40...50...60...70...80...90..."
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Processing rcp45_2011_2020:  57%|█████▋    | 8/14 [00:08<00:06,  1.08s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "100 - done.\n",
      "Creating output file that is 2810P x 2554L.\n",
      "Processing /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/Tave_sm_201120.tif [1/1] : 0Using internal nodata values (e.g. -3.4e+38) for image /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/Tave_sm_201120.tif.\n",
      "Copying nodata values from source /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/Tave_sm_201120.tif to destination ./resized_climate/rcp45_2011_2020/Tave_sm_201120.tif.\n",
      "...10...20...30...40...50...60...70...80...90..."
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Processing rcp45_2011_2020:  64%|██████▍   | 9/14 [00:10<00:05,  1.08s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "100 - done.\n",
      "Creating output file that is 2810P x 2554L.\n",
      "Processing /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/Tmax_sm_201120.tif [1/1] : 0Using internal nodata values (e.g. -3.4e+38) for image /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/Tmax_sm_201120.tif.\n",
      "Copying nodata values from source /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/Tmax_sm_201120.tif to destination ./resized_climate/rcp45_2011_2020/Tmax_sm_201120.tif.\n",
      "...10...20...30...40...50...60...70...80...90..."
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Processing rcp45_2011_2020:  71%|███████▏  | 10/14 [00:10<00:04,  1.05s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "100 - done.\n",
      "Creating output file that is 2810P x 2554L.\n",
      "Processing /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/PPT_at_201120.tif [1/1] : 0Using internal nodata values (e.g. -3.4e+38) for image /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/PPT_at_201120.tif.\n",
      "Copying nodata values from source /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/PPT_at_201120.tif to destination ./resized_climate/rcp45_2011_2020/PPT_at_201120.tif.\n",
      "...10...20...30...40...50...60...70...80...90..."
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Processing rcp45_2011_2020:  79%|███████▊  | 11/14 [00:12<00:03,  1.13s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "100 - done.\n",
      "Creating output file that is 2810P x 2554L.\n",
      "Processing /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/PPT_sm_201120.tif [1/1] : 0Using internal nodata values (e.g. -3.4e+38) for image /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/PPT_sm_201120.tif.\n",
      "Copying nodata values from source /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/PPT_sm_201120.tif to destination ./resized_climate/rcp45_2011_2020/PPT_sm_201120.tif.\n",
      "...10...20...30...40...50...60...70...80...90..."
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Processing rcp45_2011_2020:  86%|████████▌ | 12/14 [00:13<00:02,  1.10s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "100 - done.\n",
      "Creating output file that is 2810P x 2554L.\n",
      "Processing /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/PPT_sp_201120.tif [1/1] : 0Using internal nodata values (e.g. -3.4e+38) for image /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/PPT_sp_201120.tif.\n",
      "Copying nodata values from source /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/PPT_sp_201120.tif to destination ./resized_climate/rcp45_2011_2020/PPT_sp_201120.tif.\n",
      "...10...20...30...40...50...60...70...80...90..."
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Processing rcp45_2011_2020:  93%|█████████▎| 13/14 [00:14<00:01,  1.16s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "100 - done.\n",
      "Creating output file that is 2810P x 2554L.\n",
      "Processing /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/PPT_wt_201120.tif [1/1] : 0Using internal nodata values (e.g. -3.4e+38) for image /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/PPT_wt_201120.tif.\n",
      "Copying nodata values from source /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/PPT_wt_201120.tif to destination ./resized_climate/rcp45_2011_2020/PPT_wt_201120.tif.\n",
      "...10...20...30...40...50...60...70...80...90..."
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Processing rcp45_2011_2020: 100%|██████████| 14/14 [00:15<00:00,  1.13s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "100 - done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Processing rcp45_2021_2040:   0%|          | 0/14 [00:00<?, ?it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Creating output file that is 2810P x 2554L.\n",
      "Processing /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/MWMT_202140.tif [1/1] : 0Using internal nodata values (e.g. -3.4e+38) for image /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/MWMT_202140.tif.\n",
      "Copying nodata values from source /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/MWMT_202140.tif to destination ./resized_climate/rcp45_2021_2040/MWMT_202140.tif.\n",
      "...10...20...30...40...50...60...70...80...90..."
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Processing rcp45_2021_2040:   7%|▋         | 1/14 [00:01<00:13,  1.03s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "100 - done.\n",
      "Creating output file that is 2810P x 2554L.\n",
      "Processing /share/ander/Dev/climaax/ECLIPS2.0/ECLIPS2.0_RCP45/CLMcom_CCLM_4.5/TD_202140.tif [1/1] : 0"
     ]
    }
   ],
   "source": [
    "# Resizing the climate rasters DONE\n",
    "\n",
    "# This is the path of the folder where the resized climate rasters will be stored\n",
    "output_folder = \"./resized_climate/\"\n",
    "\n",
    "# Create the output folders if they don't exist\n",
    "os.makedirs(output_folder, exist_ok=True)\n",
    "# In the folder there will be 4 subfolders, one for each time period\n",
    "raster_subfolders = [\"rcp45_2011_2020\", \"rcp45_2021_2040\", \"hist_1961_1990\", \"hist_1991_2010\"]\n",
    "\n",
    "# Loop through the lists and perform gdalwarp for each raster... I need to do a zip with the subfolder list and the raster list\n",
    "for raster_list, raster_subfolder_name in zip([f_rcp45_2011_2020, f_rcp45_2021_2040, f_hist_1961_1990, f_hist_1991_2010], raster_subfolders):\n",
    "    for raster_path in tqdm(raster_list, desc=f\"Processing {raster_subfolder_name}\"):\n",
    "        # Get the filename from the path\n",
    "        filename = os.path.basename(raster_path)\n",
    "        output_subfolder = os.path.join(output_folder, raster_subfolder_name)\n",
    "        # Construct the output path by joining the output folder, the subfolder and the filename\n",
    "        output_path = os.path.join(output_subfolder, filename)\n",
    "        # create the output subfolder if it doesn't exist\n",
    "        os.makedirs(output_subfolder, exist_ok=True)\n",
    "        # Perform gdalwarp using the string command. I use the data of the blueprint DEM raster as a reference\n",
    "        os.system(f\"gdalwarp -t_srs EPSG:3035 -tr 100 -100 -te 3488731.355 1986586.650 3769731.355 2241986.650 -r bilinear  -overwrite -co COMPRESS=LZW -co TILED=YES -co BIGTIFF=YES -co BLOCKXSIZE=512 -co BLOCKYSIZE=512 {raster_path} {output_path}\")\n",
    "\n",
    "# And now, we perform a check to see if the dimensions of the output rasters are the same as the reference raster\n",
    "# Path to the reference raster\n",
    "reference_raster_path = \"./dem_3035_clip.tif\"\n",
    "check_resizing = True\n",
    "if check_resizing:\n",
    "    # Loop through the output files and compare their dimensions with the reference raster\n",
    "    for raster_list, raster_subfolder_name in zip([f_rcp45_2011_2020, f_rcp45_2021_2040, f_hist_1961_1990, f_hist_1991_2010], raster_subfolders):\n",
    "        for raster_path in tqdm(raster_list, desc=f\"Processing {raster_subfolder_name}\"):\n",
    "            # Open the output raster\n",
    "            output_raster_path = os.path.join(output_folder, raster_subfolder_name, os.path.basename(raster_path))\n",
    "            with rasterio.open(output_raster_path) as output_raster:\n",
    "                # Open the reference raster\n",
    "                with rasterio.open(reference_raster_path) as reference_raster:\n",
    "                    # Get the dimensions of the output raster\n",
    "                    output_rows, output_cols = output_raster.shape\n",
    "                    # Get the dimensions of the reference raster\n",
    "                    reference_rows, reference_cols = reference_raster.shape\n",
    "                    \n",
    "                    # Compare the dimensions\n",
    "                    if output_rows == reference_rows and output_cols == reference_cols:\n",
    "                        print(f\"{raster_path} has the same dimensions as the reference raster.\")\n",
    "                    else:\n",
    "                        print(f\"{raster_path} does NOT have the same dimensions as the reference raster.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Clip raster with shapefile of Catalonia - ALREADY DONE\n",
    "# In this cell we clip a raster using the extent of a shapefile. It is like this that \n",
    "# the DEM file has been clipped to the extent of Catalonia.\n",
    "from osgeo import gdal, ogr\n",
    "import geopandas as gpd \n",
    "import rasterio\n",
    "from rasterio.mask import mask \n",
    "catalonia_adm_path = \"/share/ander/Dev/climaax/data_cat/adm_level_stanford/catalonia_adm_3035.shp\"\n",
    "gdf = gpd.read_file(catalonia_adm_path)\n",
    "shapes = gdf.geometry.values\n",
    "\n",
    "with rasterio.open(dem_path) as src:\n",
    "    out_image, out_transform = mask(src, shapes, crop=True)\n",
    "    out_meta = src.meta.copy()\n",
    "\n",
    "out_meta.update({\"driver\": \"GTiff\",\n",
    "                 \"height\": out_image.shape[1],\n",
    "                 \"width\": out_image.shape[2],\n",
    "                 \"transform\": out_transform})\n",
    "\n",
    "with rasterio.open(\"./provaclip.tiff\", \"w\", **out_meta) as dest:\n",
    "    dest.write(out_image)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# DEM PREPROCESSING to be run to get also the functions to save raster\n",
    "# In this cell we calculate slope, aspect and roughness from the DEM and save them to the output folder.\n",
    "# There is also the definition of save_raster_as, used to save rasters from numpy arrays.\n",
    "\n",
    "def save_raster_as(array, output_file, reference_file, **kwargs):\n",
    "    \"\"\"\n",
    "    Save a raster from a 2D numpy array using another raster as reference to get the spatial extent and projection.\n",
    "    TODO: move this to a utils file\n",
    "    :param array: 2D numpy array with the data \n",
    "    :param output_file: Path to the output raster \n",
    "    :param reference_file: Path to a raster who's geotransform and projection will be used\n",
    "    :param kwargs: Keyword arguments to be passed to rasterio.open when creating the output raster\n",
    "    \"\"\"\n",
    "    with rasterio.open(reference_file) as f:\n",
    "        profile = f.profile\n",
    "        profile.update(**kwargs)\n",
    "\n",
    "        with rasterio.open(output_file, 'w', **profile) as dst:\n",
    "            dst.write(array.astype(profile['dtype']), 1)\n",
    "            #dst.write(array.astype(profile['dtype']), 1)\n",
    "\n",
    "\n",
    "def process_dem(dem_path, output_folder, verbose = False):\n",
    "    \"\"\"\n",
    "    Calculate slope and aspect from a DEM and save them to the output folder\n",
    "    :param dem_path: Path to the DEM\n",
    "    :param output_folder: Path to the output folder\n",
    "    :param verbose: If True, print some messages\n",
    "    :return: Nothing\n",
    "    \"\"\"\n",
    "\n",
    "    slope_path = os.path.join(output_folder, \"slope.tif\")\n",
    "    aspect_path = os.path.join(output_folder, \"aspect.tif\")\n",
    "    northing_path = os.path.join(output_folder, \"northing.tif\")\n",
    "    easting_path = os.path.join(output_folder, \"easting.tif\")\n",
    "    roughness_path = os.path.join(output_folder, \"roughness.tif\")\n",
    "    # I need to create the output folder if it doesn't exist\n",
    "    os.makedirs(output_folder, exist_ok=True)\n",
    "\n",
    "\n",
    "    with rasterio.open(dem_path) as src:\n",
    "        if verbose:\n",
    "            print(f'Reading dem file {dem_path}')\n",
    "        dem = src.read(1, masked  = True)\n",
    "        if verbose:\n",
    "            print(f'This is what {dem_path} looks like')\n",
    "            plt.imshow(dem)\n",
    "            print(f'Calculating slope, aspect and roughness')\n",
    "            gdal.DEMProcessing(slope_path, dem_path, 'slope')\n",
    "            gdal.DEMProcessing(aspect_path, dem_path, 'aspect')\n",
    "            gdal.DEMProcessing(roughness_path, dem_path, 'roughness')\n",
    "\n",
    "    with rasterio.open(aspect_path) as f:\n",
    "        if verbose:\n",
    "            print(f'Calculating northing and easting files')\n",
    "            print(f'Reading aspect file {aspect_path}')\n",
    "        aspect = f.read(1,   masked = True)\n",
    "        if verbose:\n",
    "            print(f'Aspect looks like this...')\n",
    "            plt.imshow(aspect)\n",
    "        #aspect[aspect <= -9999] = np.NaN\n",
    "    northing = np.cos(aspect * np.pi/180.0)\n",
    "    print(f'Saving northing file {northing_path}')\n",
    "    save_raster_as(northing, northing_path, aspect_path)\n",
    "    del northing \n",
    "    print(f'Saving easting file {easting_path}')    \n",
    "    easting = np.sin(aspect * np.pi/180.0)\n",
    "    save_raster_as(easting, easting_path, aspect_path)\n",
    "    del easting\n",
    "\n",
    "        \n",
    "#process_dem(dem_path, \"./dem_processing\", verbose = True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Clip, warp and explore the corine raster DONE\n",
    "# In this cell we clip the corine raster to the extent of Catalonia, reproject it to EPSG:3035. \n",
    "raster_clc_path = \"/share/ander/Dev/climaax/data_cat/hazard/input_hazard/veg_corine_reclass.tif\"\n",
    "output_clc_path = \"/share/ander/Dev/climaax/data_cat/hazard/input_hazard/veg_corine_reclass_clip.tif\"\n",
    "os.system(f\"gdalwarp -t_srs EPSG:3035 -tr 100 -100 -te 3488731.355 1986586.650 3769731.355 2241986.650 -r near  -overwrite -co COMPRESS=LZW -co TILED=YES -co BIGTIFF=YES -co BLOCKXSIZE=512 -co BLOCKYSIZE=512 {raster_clc_path} {output_clc_path}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'gpd' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[1], line 10\u001b[0m\n\u001b[1;32m      7\u001b[0m \u001b[38;5;66;03m# Open the shapefile\u001b[39;00m\n\u001b[1;32m      8\u001b[0m catalonia_adm_path \u001b[38;5;241m=\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m/share/ander/Dev/climaax/data_cat/adm_level_stanford/catalonia_adm_3035.shp\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m---> 10\u001b[0m catalonia \u001b[38;5;241m=\u001b[39m \u001b[43mgpd\u001b[49m\u001b[38;5;241m.\u001b[39mread_file(catalonia_adm_path)\n\u001b[1;32m     12\u001b[0m \u001b[38;5;66;03m# Read the raster data\u001b[39;00m\n\u001b[1;32m     13\u001b[0m \u001b[38;5;28;01mwith\u001b[39;00m rasterio\u001b[38;5;241m.\u001b[39mopen(raster_clc_clipped) \u001b[38;5;28;01mas\u001b[39;00m src:\n\u001b[1;32m     14\u001b[0m     \u001b[38;5;66;03m# Read the raster band\u001b[39;00m\n",
      "\u001b[0;31mNameError\u001b[0m: name 'gpd' is not defined"
     ]
    }
   ],
   "source": [
    "# PLOT THE LAND COVER DONE\n",
    "# In this cell we plot the land cover raster, clipped to the extent of Catalonia. \n",
    "\n",
    "\n",
    "# Open the raster\n",
    "raster_clc_clipped = \"/share/ander/Dev/climaax/data_cat/hazard/input_hazard/veg_corine_reclass_clip.tif\"\n",
    "# Open the shapefile\n",
    "catalonia_adm_path = \"/share/ander/Dev/climaax/data_cat/adm_level_stanford/catalonia_adm_3035.shp\"\n",
    "\n",
    "catalonia = gpd.read_file(catalonia_adm_path)\n",
    "\n",
    "# Read the raster data\n",
    "with rasterio.open(raster_clc_clipped) as src:\n",
    "    # Read the raster band\n",
    "    band = src.read(1)\n",
    "\n",
    "    fig, ax = plt.subplots(figsize=(15, 15))\n",
    "    # Plot the raster\n",
    "    rasterio.plot.show(src, ax=ax)\n",
    "    # Plot the shapefile\n",
    "    catalonia.plot(ax=ax, facecolor=\"none\", edgecolor=\"red\", linewidth=2)\n",
    "    # band is the raster data.  I want to know nrows, ncols, NODATA_value, dtype. \n",
    "\n",
    "    print(band.shape, band.dtype,band.min(),band.max())\n",
    "    print(np.unique(band))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>description</th>\n",
       "      <th>RGB_color</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>111</th>\n",
       "      <td>Continuous urban fabric</td>\n",
       "      <td>(230, 0, 77)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>112</th>\n",
       "      <td>Discontinuous urban fabric</td>\n",
       "      <td>(255, 0, 0)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>121</th>\n",
       "      <td>Industrial or commercial units</td>\n",
       "      <td>(204, 77, 242)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>122</th>\n",
       "      <td>Road and rail networks and associated land</td>\n",
       "      <td>(204, 0, 0)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>123</th>\n",
       "      <td>Port areas</td>\n",
       "      <td>(230, 204, 204)</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                    description        RGB_color\n",
       "111                     Continuous urban fabric     (230, 0, 77)\n",
       "112                  Discontinuous urban fabric      (255, 0, 0)\n",
       "121              Industrial or commercial units   (204, 77, 242)\n",
       "122  Road and rail networks and associated land      (204, 0, 0)\n",
       "123                                  Port areas  (230, 204, 204)"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# create an info dataframe on CLC land cover DONE \n",
    "# In this cell we build a dictionary with the values the description of the values, and also a RGB color for each value\n",
    "# TO BE DONE: add the RGB color to the dataframe\n",
    "corine_land_cover_level_3 = {\n",
    "    111: \"Continuous urban fabric\",\n",
    "    112: \"Discontinuous urban fabric\",\n",
    "    121: \"Industrial or commercial units\",\n",
    "    122: \"Road and rail networks and associated land\",\n",
    "    123: \"Port areas\",\n",
    "    124: \"Airports\",\n",
    "    131: \"Mineral extraction sites\",\n",
    "    132: \"Dump sites\",\n",
    "    133: \"Construction sites\",\n",
    "    141: \"Green urban areas\",\n",
    "    142: \"Sport and leisure facilities\",\n",
    "    211: \"Non-irrigated arable land\",\n",
    "    212: \"Permanently irrigated land\",\n",
    "    213: \"Rice fields\",\n",
    "    221: \"Vineyards\",\n",
    "    222: \"Fruit trees and berry plantations\",\n",
    "    223: \"Olive groves\",\n",
    "    231: \"Pastures\",\n",
    "    241: \"Annual crops associated with permanent crops\",\n",
    "    242: \"Complex cultivation patterns\",\n",
    "    243: \"Land principally occupied by agriculture, with significant areas of natural vegetation\",\n",
    "    244: \"Agro-forestry areas\",\n",
    "    311: \"Broad-leaved forest\",\n",
    "    312: \"Coniferous forest\",\n",
    "    313: \"Mixed forest\",\n",
    "    321: \"Natural grasslands\",\n",
    "    322: \"Moors and heathland\",\n",
    "    323: \"Sclerophyllous vegetation\",\n",
    "    324: \"Transitional woodland-shrub\",\n",
    "    331: \"Beaches, dunes, sands\",\n",
    "    332: \"Bare rocks\",\n",
    "    333: \"Sparsely vegetated areas\",\n",
    "    334: \"Burnt areas\",\n",
    "    335: \"Glaciers and perpetual snow\",\n",
    "    411: \"Inland marshes\",\n",
    "    412: \"Peat bogs\",\n",
    "    421: \"Salt marshes\",\n",
    "    422: \"Salines\",\n",
    "    423: \"Intertidal flats\",\n",
    "    511: \"Water courses\",\n",
    "    512: \"Water bodies\",\n",
    "    521: \"Coastal lagoons\",\n",
    "    522: \"Estuaries\",\n",
    "    523: \"Sea and ocean\"\n",
    "}\n",
    "\n",
    "df = pd.DataFrame.from_dict(corine_land_cover_level_3, orient=\"index\", columns=[\"description\"])\n",
    "# Add a column with the RGB color\n",
    "# source: https://clc.gios.gov.pl/doc/clc/CLC_Legend_EN.pdf\n",
    "\n",
    "df[\"RGB_color\"] = [(0,0,0)]*len(df)\n",
    "df.at[111, \"RGB_color\"] = (230, 0, 77)\n",
    "df.at[112, \"RGB_color\"] = (255, 0, 0)\n",
    "df.at[121, \"RGB_color\"] = (204, 77, 242)\n",
    "df.at[122, \"RGB_color\"] = (204, 0, 0)\n",
    "df.at[123, \"RGB_color\"] = (230, 204, 204)\n",
    "df.at[124, \"RGB_color\"] = (230, 204, 230)\n",
    "df.at[131, \"RGB_color\"] = (166, 0, 204)\n",
    "df.at[132, \"RGB_color\"] = (166, 77, 0)\n",
    "df.at[133, \"RGB_color\"] =  (255, 77, 255)\n",
    "df.at[141, \"RGB_color\"] = (255, 166, 204)\n",
    "df.at[142, \"RGB_color\"] = (255, 230, 255)\n",
    "df.at[211, \"RGB_color\"] = (255, 255, 168)\n",
    "# to be continued...\n",
    "df.head()\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# save the raster of clc with all non burnable classes set to 0 DONE\n",
    "\n",
    "\n",
    "list_of_non_burnable_clc = [111,112,121,122,123,124,131,132,133,141,142,331,332,333,335,411,412,421,422,423,511,512,521,522,523]\n",
    "# Below, some considerations in the 3XX classes of the CLC and their degree of burnability.\n",
    "# 331 332 333  are respectively beaches, dunes, sands; bare rocks; sparsely vegetated areas; in this case I will consider them as non-burnable\n",
    "# 334 burnt areas in this case I will consider them as burnable\n",
    "# 335 glaciers and perpetual snow in this case I will consider them as non-burnable\n",
    "\n",
    "# Open the raster\n",
    "output_clc_path = \"/share/ander/Dev/climaax/data_cat/hazard/input_hazard/veg_corine_reclass_clip_nb.tif\"\n",
    "with rasterio.open(raster_clc_clipped) as src:\n",
    "    # Read the raster band\n",
    "    band = src.read(1)\n",
    "    # First plot, left side of the multiplot\n",
    "    fig, ax = plt.subplots(figsize=(10, 10))\n",
    "    # Plot the raster\n",
    "    plt.imshow(band)\n",
    "    # Set the values in list_of_non_burnable_clc to 0\n",
    "    band[np.isin(band, list_of_non_burnable_clc)] = 0\n",
    "    # Plot the raster, right side of the multiplot\n",
    "    fig, ax = plt.subplots(figsize=(10, 10))\n",
    "    plt.imshow(band)\n",
    "    # Save the modified raster\n",
    "    with rasterio.open(output_clc_path, 'w', **src.profile) as dst:\n",
    "        dst.write(band, 1)\n",
    "        \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "# This was not working due to problems in the rasterio library (warping the raster was not working)\n",
    "# I had to use the gdalwarp command line tool instead\n",
    "\n",
    "def resize_rasters(path_reference_raster, path_rasters_to_resize = None, output_folder = None, verbose = True):\n",
    "    \"\"\"\n",
    "    This function resizes the rasters to the same size as the last raster.\n",
    "    path_reference_raster: path to the referenec raster of the simulation. \n",
    "    path_rasters_to_resize: list of paths to the rasters to resize. If None, it will resize all the rasters in the folder \n",
    "    which are not the reference raster.\n",
    "    verbose: if True, it will print the path of the rasters that are being resized as well as additional information. \n",
    "    \"\"\"\n",
    "\n",
    "    with rasterio.open(path_reference_raster) as src:\n",
    "        dst_crs = src.crs\n",
    "        dst_transform = src.transform\n",
    "        dst_height = src.height\n",
    "        dst_width = src.width\n",
    "        dst_bounds = src.bounds\n",
    "        dst_meta = src.meta\n",
    "        if verbose: \n",
    "            print(\"destination raster crs:\", dst_crs)\n",
    "            print(\"destination raster transform:\", dst_transform)\n",
    "            print(\"destination raster height:\", dst_height)\n",
    "            print(\"destination raster width:\", dst_width)\n",
    "            print(\"destination raster bounds:\", dst_bounds)\n",
    "\n",
    "    if path_rasters_to_resize is None:\n",
    "        # in this case, I will search for all the rasters in the folder of the last raster, which are not the last raster\n",
    "        path_rasters_to_resize = []\n",
    "        # I get the folder given the path of the last raster\n",
    "        input_folder = os.path.dirname(path_reference_raster)\n",
    "        # I get the name of the last raster\n",
    "        target_file = os.path.basename(path_reference_raster)\n",
    "        for filename in os.listdir(input_folder):\n",
    "            if filename.endswith('.tiff') and (os.path.basename(filename) is not target_file):\n",
    "                path_rasters_to_resize.append(os.path.join(input_folder, filename))\n",
    "    if verbose:\n",
    "        print(\"rasters to resize:\", path_rasters_to_resize)\n",
    " \n",
    "    for path_raster in path_rasters_to_resize:\n",
    "        if verbose:\n",
    "            print(\"resizing raster:\", path_raster)\n",
    "        with rasterio.open(path_raster) as src:\n",
    "            src_crs = src.crs\n",
    "            src_transform = src.transform\n",
    "            src_height = src.height\n",
    "            src_width = src.width\n",
    "            src_bounds = src.bounds\n",
    "            if verbose: \n",
    "                print(\"source raster crs:\", src_crs)\n",
    "                print(\"source raster transform:\", src_transform)\n",
    "                print(\"source raster height:\", src_height)\n",
    "                print(\"source raster width:\", src_width)\n",
    "                print(\"source raster bounds:\", src_bounds)     \n",
    "            #calculate transform array and shape of reprojected raster\n",
    "            transform, width, height = calculate_default_transform(\n",
    "            src_crs, dst_crs, dst_width, dst_height, *dst_bounds)\n",
    "            if verbose: \n",
    "                print(\"transform array of source raster\")\n",
    "                print(src_transform)\n",
    "                print(\"transform array of destination raster\")\n",
    "                print(transform)\n",
    "            #working off the meta for the destination raster\n",
    "            kwargs = src.meta.copy()\n",
    "            kwargs.update({\n",
    "            'crs': dst_crs,\n",
    "            'transform': transform,\n",
    "            'width': width,\n",
    "            'height': height\n",
    "            })\n",
    "            # open  newly resized raster, write it and save it\n",
    "            # I will use the same name as the original raster, but with the suffix \"_resized\"\n",
    "            # I will save it in the same folder as the original raster\n",
    "            if output_folder is None:\n",
    "                output_folder = os.path.dirname(path_raster)\n",
    "            with rasterio.open(os.path.join(output_folder, os.path.basename(path_raster).split('.tiff')[0] + \"_resized0.tiff\"), 'w', **kwargs)  as resized_raster:\n",
    "                    #reproject and save raster band data\n",
    "                    for i in range(1, src.count + 1):\n",
    "                        reproject(\n",
    "                        source=rasterio.band(src, i),\n",
    "                        destination=rasterio.band(resized_raster, i),\n",
    "                        src_transform=src_transform,\n",
    "                        src_crs=src_crs,\n",
    "                        dst_transform=transform,\n",
    "                        dst_crs=dst_crs,\n",
    "                        resampling=Resampling.nearest)\n",
    "            \n",
    "            \n",
    "            \n",
    "def clean_resized_tiff(folder_name):\n",
    "    \"\"\"this function removes the resized tiff files. It deletes any file with the suffix \"_resized in the name and that \n",
    "     are tiff files\n",
    "    \"\"\"\n",
    "    for filename in os.listdir(folder_name):\n",
    "        if filename.endswith('.tiff') and (\"_resized\" in filename):\n",
    "            os.remove(os.path.join(folder_name, filename))\n",
    "            print(\"removed file:\", filename)\n",
    "            \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of rows... 1217\n",
      "Number of rows discarded: 395\n",
      "These are the buggy entries in the FIRE_YEAR column... ['Unknown' '1995 - 2015 - 2012' '1995 - 2015 - 2021' '1995 - 2012 - 2021'\n",
      " '1995 - 2006 - 2010' '1995 - 1999 - 2002' '1995 - 2000 - 2021'\n",
      " '1986 - 1995 - 2019' '1986 - 1998 - 2004' '1988 - 1993 - 2001'\n",
      " '1986 - 2011' '1986 - 2012' '1986 - 1997' '1986 - 1995' '1998 - 2020'\n",
      " '1986 - 1999 - 2012' '1986 - 2003 - 2012' '1986 - 2004 - 2012'\n",
      " '1986 - 2006 - 2012' '1986 - 2007 - 2012' '1989 - 1994' '1989 - 2003'\n",
      " '1989 - 2005' '1989 - 2020' '1989 - 2014' '1989 -2012' '1990 - 1994'\n",
      " '1995 - 2010' '1995 - 2017' '1986 - 2000 - 2021' '1986 - 2001 - 2003'\n",
      " '1986 - 2001 - 2022' '1986 - 2002 - 2022' '1988 - 2001' '1988 - 2003'\n",
      " '1988 - 2009' '1988 - 2012' '1988 - 1991' '1988 - 1994' '1988 - 2007'\n",
      " '1986 - 2016' '1986 - 2022' '1986 - 1994' '1986 - 2005' '1994 - 2003'\n",
      " '1994 - 2011' '1994 - 1987' '1990 - 2016' '1990 - 2020' '2005 - 2020'\n",
      " '2005 - 2013' '2006 - 2012' '1986 - 2004 - 2022' '1986 - 1994 - 1997'\n",
      " '1986 - 1994 - 1996' '1994 - 2005' '1994 - 2015' '1994 - 2002'\n",
      " '1994 - 2000' '2002 - 2020' '2003 - 2004' '1994 - 2006' '2006 - 2010'\n",
      " '2007 - 2012' '2008 - 2010' '2008 - 2015' '2008 - 2020' '1986 - 1993'\n",
      " '1986 - 2000' '1986 - 2001' '1993 - 2005' '2003 - 2011' '2003 - 2012'\n",
      " '2003 - 2015' '2003 - 2020' '2003 - 2019' '1999 - 2002' '1999 - 2004'\n",
      " '1999 - 2006' '1999 - 2012' '1990 - 1993' '1990 - 2001' '1993 - 1995'\n",
      " '1993 - 1994' '1995 - 2012' '2004 - 2017' '2004 - 2012' '2004 - 2015'\n",
      " '2004 - 2006' '2004 - 2022' '1994 - 2003 - 2011' '1994 - 2005 - 2013'\n",
      " '1994 - 2005 - 2015' '1995 - 2000 - 2015' '1995 - 2000 - 2012'\n",
      " '1990 - 1991' '1990 - 2019' '1994 - 1998' '1995 - 2009' '1995 - 2006'\n",
      " '1986 - 1994 - 2003' '1986 - 1994 - 2011' '2005 - 2007' '2005 - 2017'\n",
      " '2005 - 2008' '2005 - 2021' '1986 - 2007' '1986 - 2013' '2009 - 2020'\n",
      " '2009 - 2016' '1986 - 1995 - 1999' '1986 - 1995 - 2002' '1986 - 2002'\n",
      " '1986 - 2003' '1986 - 2021' '1986 - 1988' '1986 - 1996' '1986 - 2004'\n",
      " '1986 - 1999 - 2002' '1986 - 2003 - 2011' '1986 - 1995 - 1997'\n",
      " '1986 - 2000 - 2010' '1986 - 2000 - 2022' '1994 - 2016' '1994 - 1995'\n",
      " '1994 - 2017' '2009 - 2019' '2018 - 2019' '2016 - 2019' '1993 - 2001'\n",
      " '1993 - 2003' '1993 - 2000' '1993 - 2022' '2012 - 2016' '2016 - 2021'\n",
      " '2015 - 2014' '2014 - 2022' '2012 - 2021' '2000 - 2018' '2000 - 2016'\n",
      " '2000 - 2021' '2000 - 2006' '2000 - 2003' '2000 - 2012' '2000 - 2007'\n",
      " '1988 - 2001 - 2003' '1989 - 2012 - 2017' '1991 - 1993' '1991 - 1994'\n",
      " '1991 - 2016' '1994 - 2012' '1995 - 1999' '1995 - 2002' '1996 - 2006'\n",
      " '1996 - 2000' '1989 - 1994 - 2001' '1989 - 1992 - 1994' '1986 - 2009'\n",
      " '1986 - 1999' '1986 - 1998' '1994 - 2007' '1994 - 2009' '1987 - 2012'\n",
      " '1986 - 1989 - 2000' '1986 - 1999 - 2004' '1986 - 1999 - 2006'\n",
      " '1989 - 2017' '1989 - 2012' '1989 - 2013' '1989 - 2001' '1988 - 1993'\n",
      " '1994 - 2013' '2013 - 2019' '2015 - 2019' '2012 - 2019' '2018 - 2012'\n",
      " '2018 - 2016' '1986 - 2015' '1994 - 1999' '1989 - 1993' '1989 - 1991'\n",
      " '1994 - 2020' '2000 - 2010' '2001 - 2003' '2001 - 2020' '1986 - 1989'\n",
      " '1991 - 2009' '1991 - 2019' '1997 - 2015' '1989 - 1992' '1989 - 2000'\n",
      " '1989 - 2007' '2001 - 2010' '2001 - 2018' '2002 - 2017' '1991 - 2020'\n",
      " '1991 - 1998' '1991 - 2000' '1995 - 2000' '1995 - 2015' '1997 - 2014'\n",
      " '1997 - 2006' '1997 - 2007' '1997 - 2005' '1989 - 2000 - 2020'\n",
      " '1989 - 2000 - 2018' '1994 - 2001' '1986 - 2012 - 2018'\n",
      " '1986 - 1993 - 2001' '1986 - 1993 - 2000' '1986 - 1993 - 2002'\n",
      " '1986 - 1993 - 2022' '1995 - 2021' '1995 - 1997' '1995 - 2013'\n",
      " '2012 - 2018' '2017 - 2020' '2017 - 2016' '1990 - 2016 - 2021'\n",
      " '1990 - 1993 - 1994' '1990 - 1993 - 2005' '1993 - 1994 - 2002'\n",
      " '1994 - 1995 - 1999' '1994 - 2014' '1997 - 2014 - 2015'\n",
      " '1998 - 2009 - 2020' '1999 - 2004 - 2012' '2012 - 2017' '2016 - 2020'\n",
      " '1994 - 1997' '1999 - 2006 - 2012' '2014 - 2015 - 2019'\n",
      " '2012 - 2016 - 2018' '1986 - 1999 - 2004 - 2012'\n",
      " '1986 - 1999 - 2006 - 2012' '1986 - 1993 - 2001 - 2022'\n",
      " '1986 - 1993 - 2002 - 2022' '1986 - 1994 - 2003 - 2011'\n",
      " '1986 - 1995 - 1999 - 2002' '1986 - 2006' '1992 - 1994' '1992 - 1995'\n",
      " '1992 - 2011' '1992 - 2020' '1998 - 2003' '1998 - 2009' '1994 - 1996'\n",
      " '1998 - 2006' '1995 - 2000 - 2012 - 2015' '1995 - 2000 - 2015 - 2021'\n",
      " '1995 - 2000 - 2012 - 2021' '1995 - 2012 - 2015 - 2021'\n",
      " '1999 - 2004 - 2006 - 2012' '1986 - 1999 - 2004 - 2006 - 2012'\n",
      " '1995 - 2000 - 2012 - 2015 - 2021' '1993 - 2002' '1998 - 2005'\n",
      " '1998 - 2004' '1986 - 2016 - 2022' '1986 - 2005 - 2007']\n",
      "......\n",
      "I will convert to int the YEAR_FIRE column of the non buggy data...\n",
      "The filtered dataset comprises of  822 rows, spanning the years 1986 to 2022\n",
      "After the filtering the final filtered dataset comprises of  717 rows, spanning the years 1991 to 2022\n"
     ]
    }
   ],
   "source": [
    "# data cleaning of Catalan wildfires. Discard all rows whose \"YEAR_FIRE\" is not of 4 characters. How many are the \n",
    "# rows discarded? The row discarded seem to be the overlapping fires...for this they have multiple date entries.\n",
    "import geopandas as gpd\n",
    "fires_path = \"/share/ander/Dev/climaax/data_cat/hazard/input_hazard/fires/Forest_Fire_1986_2022_filtrat_3035.shp\"\n",
    "fires = gpd.read_file(fires_path)\n",
    "print(\"Number of rows...\", len(fires))\n",
    "rows_discarded = len(fires) - len(fires[fires['YEAR_FIRE'].str.len() == 4])\n",
    "fires_2 = fires[fires['YEAR_FIRE'].str.len() == 4]\n",
    "fires_0 = fires[fires['YEAR_FIRE'].str.len() != 4]\n",
    "print(\"Number of rows discarded:\", rows_discarded)\n",
    "print(\"These are the buggy entries in the FIRE_YEAR column...\",fires_0.YEAR_FIRE.unique())\n",
    "print(\"......\")\n",
    "print(\"I will convert to int the YEAR_FIRE column of the non buggy data...\")\n",
    "\n",
    "#fires_2.loc['YEAR_FIRE'] = fires_2['YEAR_FIRE'].astype(int) \n",
    "fires_2.loc[:, 'YEAR_FIRE'] = fires_2['YEAR_FIRE'].astype(int) # to avoid setting with copy warning\n",
    "\n",
    "print(\"The filtered dataset comprises of \", len(fires_2), \"rows, spanning the years\" , fires_2.YEAR_FIRE.min(), \"to\", fires_2.YEAR_FIRE.max())\n",
    "# in the following, fires_2 will be our geo dataframe with the fires. \n",
    "# Now selecting just the fires_2 with year > 1990\n",
    "fires_2 = fires_2[fires_2['YEAR_FIRE'] > 1990]\n",
    "print(\"After the filtering the final filtered dataset comprises of \", len(fires_2), \"rows, spanning the years\" , fires_2.YEAR_FIRE.min(), \"to\", fires_2.YEAR_FIRE.max())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "shape of the reference raster: (2554, 2810)\n",
      "transform of the reference raster: | 100.00, 0.00, 3488731.35|\n",
      "| 0.00,-100.00, 2241986.65|\n",
      "| 0.00, 0.00, 1.00|\n",
      "sto prima di features.rasterize\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "sto dopo  di  features.rasterize\n"
     ]
    }
   ],
   "source": [
    "# In this cell, a util to rasterize a vector file is defined. \n",
    "# The geodataframe of the fires is rasterized and saved to file, using the corine land cover raster\n",
    "# as a reference.\n",
    "\n",
    "\n",
    "# Util to rasterize a vector file \n",
    "\n",
    "import rasterio\n",
    "from rasterio import features\n",
    "\n",
    "def rasterize_numerical_feature(gdf, reference_file, column=None, verbose = True):\n",
    "    \"\"\"\n",
    "    Rasterize a vector file using a reference raster to get the shape and the transform.\n",
    "    :param gdf: GeoDataFrame with the vector data\n",
    "    :param reference_file: Path to the reference raster\n",
    "    :param column: Name of the column to rasterize. If None, it will rasterize the geometries\n",
    "    :return: Rasterized version of the vector file\n",
    "    \"\"\"\n",
    "\n",
    "    with rasterio.open(reference_file) as f:\n",
    "        out = f.read(1,   masked = True)\n",
    "        myshape = out.shape\n",
    "        mytransform = f.transform #f. ...\n",
    "    del out\n",
    "    if verbose:\n",
    "        print(\"shape of the reference raster:\", myshape)\n",
    "        print(\"transform of the reference raster:\", mytransform)       \n",
    "    out_array = np.zeros(myshape)#   out.shape)\n",
    "    # this is where we create a generator of geom, value pairs to use in rasterizing\n",
    "    if column is not None:\n",
    "        shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[column]))\n",
    "    else:\n",
    "        shapes = ((geom, 1) for geom in gdf.geometry)\n",
    "    print(\"I am before features.rasterize\")\n",
    "    burned = features.rasterize(shapes=shapes, fill=np.NaN, out=out_array, transform=mytransform)#, all_touched=True)\n",
    "    #    out.write_band(1, burned)\n",
    "    print(\"I am after features.rasterize\")\n",
    "\n",
    "    return burned\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "raster_clc_clipped = \"/share/ander/Dev/climaax/data_cat/hazard/input_hazard/veg_corine_reclass_clip.tif\"\n",
    "# Rasterize the fires...\n",
    "fires_rasterized = rasterize_numerical_feature(fires_2, raster_clc_clipped, column=None)\n",
    "# save to file\n",
    "\n",
    "save_raster_as(fires_rasterized, fires_raster_path, raster_clc_clipped)\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'np' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[2], line 5\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[38;5;66;03m# In this cell, I check that the rasterized fires can assume just the values 0 and 1 and \u001b[39;00m\n\u001b[1;32m      2\u001b[0m \u001b[38;5;66;03m# that there are no nan values. \u001b[39;00m\n\u001b[1;32m      3\u001b[0m \n\u001b[1;32m      4\u001b[0m \u001b[38;5;66;03m# values of the rasterized fires\u001b[39;00m\n\u001b[0;32m----> 5\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[43mnp\u001b[49m\u001b[38;5;241m.\u001b[39munique(fires_rasterized))\n\u001b[1;32m      7\u001b[0m \u001b[38;5;66;03m# does the rasterized files have nan? \u001b[39;00m\n\u001b[1;32m      8\u001b[0m \u001b[38;5;28mprint\u001b[39m(np\u001b[38;5;241m.\u001b[39misnan(fires_rasterized)\u001b[38;5;241m.\u001b[39many())\n",
      "\u001b[0;31mNameError\u001b[0m: name 'np' is not defined"
     ]
    }
   ],
   "source": [
    "# In this cell, I check that the rasterized fires can assume just the values 0 and 1 and \n",
    "# that there are no nan values. \n",
    "\n",
    "# values of the rasterized fires\n",
    "print(np.unique(fires_rasterized))\n",
    "\n",
    "# does the rasterized files have nan? \n",
    "print(np.isnan(fires_rasterized).any())\n",
    "\n",
    "plt.matshow(fires_rasterized)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "# I define a class that will handle the path and metadata of a raster.\n",
    "class MyRaster:\n",
    "    def __init__(self, path, label):\n",
    "        self.path = path\n",
    "        self.label = label\n",
    "        self.data = None\n",
    "        self.mask = None\n",
    "        self.nodata = None\n",
    "        self.read_raster()\n",
    "    def read_raster(self):\n",
    "        with rasterio.open(self.path) as src:\n",
    "            self.data = src.read(1, masked=True)\n",
    "            self.mask = src.read_masks(1)\n",
    "            self.nodata = src.nodata\n",
    "    def set_data(self, data):\n",
    "        self.data = data\n",
    "    def get_data(self):\n",
    "        return self.data\n",
    "    def get_mask(self):\n",
    "        return self.mask\n",
    "    def get_nodata(self):\n",
    "        return self.nodata\n",
    "    def get_label(self):\n",
    "        return self.label\n",
    "    def get_path(self):\n",
    "        return self.path\n",
    "    def get_shape(self):\n",
    "        return self.data.shape\n",
    "    def get_transform(self):\n",
    "        return self.transform\n",
    "    def get_crs(self):\n",
    "        return self.crs\n",
    "    def get_bounds(self):\n",
    "        return self.bounds\n",
    "    def get_meta(self):\n",
    "        return self.meta\n",
    "    def get_height(self):\n",
    "        return self.height\n",
    "    def get_width(self):\n",
    "        return self.width\n",
    "    def get_dtype(self):\n",
    "        return self.dtype\n",
    "    def get_count(self):\n",
    "        return self.count\n",
    "    def get_index(self):\n",
    "        return self.index\n",
    "    def get_nodatavals(self):\n",
    "        return self.nodatavals\n",
    "\n",
    "# In the following, I define a function that assembles the dictionary with all the rasters\n",
    "# to be used in the model which are related\n",
    "# to climate. The same is done with the DEM-related layers and the vegetation related layers.\n",
    "\n",
    "climate_paths = []\n",
    "clim_var_names = [\"MWMT\", \"TD\", \"AHM\", \"SHM\", \"DDbelow0\", \"DDabove18\", \"MAT\", \"MAP\", \"Tave_sm\", \"Tmax_sm\", \"PPT_at\", \"PPT_sm\", \"PPT_sp\", \"PPT_wt\"]\n",
    "climate_root = \"/share/ander/Dev/climaax/resized_climate\"\n",
    "\n",
    "def assemble_climate_dict(clim_category,clim_cat_namefile, clim_var_names, climate_root = \"/share/ander/Dev/climaax/resized_climate\"):\n",
    "    \"\"\"\n",
    "    Parameters\n",
    "    ----------\n",
    "    clim_category : string that can have the following values: \"rcp45_2011_2020\", \"rcp45_2021_2040\", \"hist_1961_1990\", \"hist_1991_2010\"\n",
    "    clim_cat_namefile: string which classifies the filenames of the category (tipically the years)\n",
    "        examples are: \"201120\", \"202140\", \"196190\", \"199110\"\n",
    "    clim_var_names: a list of strings with the names of the climate variables to be used in the model. An example of such list is:\n",
    "    [\"MWMT\", \"TD\", \"AHM\", \"SHM\", \"DDbelow0\", \"DDabove18\", \"MAT\", \"MAP\", \"Tave_sm\", \"Tmax_sm\", \"PPT_at\", \"PPT_sm\", \"PPT_sp\", \"PPT_wt\"]\n",
    "    Returns: a dictionary with the climate variables as keys and the MyRaster objects as values\n",
    "    sample usage: \n",
    "    climate_dict = assemble_climate_dict(\"hist_1991_2010\", \"1991_2010\", clim_var_names)\n",
    "    future_climate_dict = assemble_climate_dict(\"rcp45_2021_2040\", \"202140\", clim_var_names)\n",
    "    \"\"\"\n",
    "    \n",
    "    climate_paths = []\n",
    "    for clim_var_name in clim_var_names:\n",
    "        climate_paths.append(os.path.join(climate_root, clim_category, clim_var_name + \"_\" + clim_cat_namefile + \".tif\"))\n",
    "\n",
    "    climate_dict = {}\n",
    "    for path, label in zip(climate_paths, clim_var_names):\n",
    "        climate_dict[label] = MyRaster(path, label)\n",
    "        climate_dict[label].read_raster()\n",
    "    return climate_dict\n",
    "\n",
    "def assemble_dem_dict(dem_paths, dem_labels):\n",
    "    \"\"\"\n",
    "    This function assembles the dictionary with all the rasters to be used in the model which are related\n",
    "    to topography. \n",
    "    :param dem_paths: paths to the dem and related rasters\n",
    "    :param dem_labels: labels of the dem and related rasters\n",
    "    :return: a dictionary with all the rasters to be used in the model\n",
    "    usage example: \n",
    "    dem_paths = [dem_path, slope_path, aspect_path, easting_path, northing_path, roughness_path]\n",
    "    dem_labels = [\"dem\", \"slope\", \"aspect\", \"easting\", \"northing\", \"roughness\"]\n",
    "    dem_dict = assemble_dem_dict(dem_paths, dem_labels)\n",
    "    \"\"\"\n",
    "    # Create the dictionary\n",
    "    dem_dict = {}\n",
    "    for path, label in zip(dem_paths, dem_labels):\n",
    "        dem_dict[label] = MyRaster(path, label)\n",
    "        dem_dict[label].read_raster()\n",
    "    \n",
    "    return dem_dict\n",
    "\n",
    "\"\"\"         \n",
    "dem_raster = MyRaster(dem_path, \"dem\")\n",
    "dem_raster.read_raster()\n",
    "slope_raster = MyRaster(slope_path, \"slope\")\n",
    "slope_raster.read_raster()\n",
    "northing_raster = MyRaster(aspect_path, \"aspect\")\n",
    "northing_raster.read_raster()\n",
    "easting_raster = MyRaster(easting_path, \"easting\")\n",
    "easting_raster.read_raster()\n",
    "roughness_raster = MyRaster(roughness_path, \"roughness\")\n",
    "dem_rasters = [dem_raster, slope_raster, northing_raster, easting_raster, roughness_raster]\n",
    "\"\"\"\n",
    "\n",
    "\n",
    "def assemble_veg_dictionary(veg_path, dem_path, verbose = False):\n",
    "    \"\"\"\n",
    "    This function assembles the dictionary with all the rasters to be used in the model which are related\n",
    "    to vegetation. This comprises of:\n",
    "    - the vegetation raster\n",
    "    - the rasters of vegetation densities: a fuzzy counting of the neighboring vegetation for each type of vegetation\n",
    "        See Tonini et al. 2020 for more details.\n",
    "    :param veg_path: path to the vegetation raster. \n",
    "    :return: a dictionary with all the veg rasters to be used in the model\n",
    "    usage example:\n",
    "    veg_dict = assemble_veg_dictionary(veg_path, dem_path, verbose = True)\n",
    "    \"\"\"\n",
    "    # Create the dictionary\n",
    "    veg_dict = {}\n",
    "    veg_dict[\"veg\"] = MyRaster(veg_path, \"veg\")\n",
    "    veg_dict[\"veg\"].read_raster()\n",
    "    veg_arr = veg_dict[\"veg\"].data.astype(int)\n",
    "    dem_raster = MyRaster(dem_path, \"dem\")\n",
    "    dem_raster.read_raster()\n",
    "    dem_arr = dem_raster.data\n",
    "    dem_nodata = dem_raster.nodata\n",
    "    \n",
    "    veg_mask = np.where(veg_arr == 0, 0, 1)\n",
    "\n",
    "\n",
    "    # complete the mask selecting the points where also dem exists.\n",
    "    mask = (veg_mask == 1) & (dem_arr != dem_nodata)\n",
    "    \n",
    "    # evaluation of perc just in vegetated area, non vegetated are grouped in code 0\n",
    "    veg_int = veg_arr.astype(int)\n",
    "    veg_int = np.where(mask == 1, veg_int, 0)\n",
    "    window_size = 2\n",
    "    types = np.unique(veg_int)\n",
    "    if verbose:\n",
    "        print(\"types of vegetation in the veg raster:\", types)\n",
    "    types_presence = {}\n",
    "    \n",
    "    counter = np.ones((window_size*2+1, window_size*2+1))\n",
    "    take_center = 1\n",
    "    counter[window_size, window_size] = take_center \n",
    "    counter = counter / np.sum(counter)\n",
    "\n",
    "    # perc --> neightboring vegetation generation                \n",
    "    for t in tqdm(types):\n",
    "        density_entry = 'perc_' + str(int(t))\n",
    "        if verbose:\n",
    "            print(f'Processing vegetation density {density_entry}')\n",
    "        temp_data = 100 * signal.convolve2d(veg_int==t, counter, boundary='fill', mode='same')\n",
    "        temp_raster = MyRaster(dem_path, density_entry) # the path is dummy... I need just the other metadata.        \n",
    "        temp_raster.read_raster()\n",
    "        temp_raster.set_data(temp_data)\n",
    "        veg_dict[density_entry] = temp_raster\n",
    "\n",
    "\n",
    "    return veg_dict, mask\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "def preprocessing(dem_dict, veg_dict, climate_dict, fires_raster, mask):\n",
    "    \"\"\"\n",
    "    Usage: \n",
    "    X_all, Y_all, columns = preprocessing(dem_dict, veg_dict, climate_dict, fires_raster, mask)\n",
    "\n",
    "\n",
    "    \"\"\"\n",
    "   \n",
    "    # creaate X and Y datasets\n",
    "    n_pixels = len(dem_dict[\"dem\"].data[mask])\n",
    "\n",
    "    # the number of features is given by all the dem layers, all the veg layers, all the climate layers.\n",
    "    # TODO: add the possibility to add other layers which belong to a \"misc\" category.\n",
    "    n_features = len(dem_dict.keys())+ len(veg_dict.keys()) + len(climate_dict.keys())\n",
    "    #create the dictionary with all the data \n",
    "    data_dict = {**dem_dict, **veg_dict, **climate_dict}\n",
    "    \n",
    "    X_all = np.zeros((n_pixels, n_features)) # This is going to be big... Maybe use dask? \n",
    "    Y_all = fires_raster.data[mask]\n",
    "\n",
    "    print('Creating dataset for RandomForestClassifier')\n",
    "    columns = data_dict.keys()\n",
    "    for col, k in enumerate(data_dict):\n",
    "        print(f'Processing column: {k}')\n",
    "        data = data_dict[k]\n",
    "        # data is a MyRaster object and data.data is the numpy array with the data\n",
    "        X_all[:, col] = data.data[mask] \n",
    "\n",
    "    return X_all, Y_all, columns \n",
    "\n",
    "def prepare_sample(X_all, Y_all, percentage=0.1): \n",
    "    \"\"\" usage:\n",
    "    model, X_train, X_test, y_train, y_test = train(X_all, Y_all, percentage)\n",
    "    parameters:\n",
    "    X_all: the X dataset with the descriptive features\n",
    "    Y_all: the Y dataset with the target variable (burned or not burned)\n",
    "    percentage: the percentage of the dataset to be used for training\n",
    "    \"\"\"\n",
    "\n",
    "    # randomforest parameters\n",
    "    max_depth = 10\n",
    "    number_of_trees = 100\n",
    "    # filter df taking info in the burned points\n",
    "    fires_rows = Y_all.data != 0\n",
    "    print(f'Number of burned points: {np.sum(fires_rows)}')\n",
    "    X_presence = X_all[fires_rows]\n",
    "    \n",
    "    # sampling training set       \n",
    "    print(' I am random sampling the dataset ')\n",
    "    # reduction of burned points --> reduction of training points       \n",
    "    reduction = int((X_presence.shape[0]*percentage))\n",
    "    print(f\"reducted df points: {reduction} of {X_presence.shape[0]}\")\n",
    "    \n",
    "    # sampling and update presences \n",
    "    X_presence_indexes = np.random.choice(X_presence.shape[0], size=reduction, replace=False)\n",
    "\n",
    "    X_presence = X_presence[X_presence_indexes, :]         \n",
    "    # select not burned points\n",
    "\n",
    "    X_absence = X_all[~fires_rows] #why it is zero?  \n",
    "    print(\"X_absence.shape[0]\", X_absence.shape[0])\n",
    "    print(\"X_presence.shape[0]\", X_presence.shape[0])\n",
    "    X_absence_choices_indexes = np.random.choice(X_absence.shape[0], size=X_presence.shape[0], replace=False)\n",
    "\n",
    "    X_pseudo_absence = X_absence[X_absence_choices_indexes, :]\n",
    "    # create X and Y with same number of burned and not burned points\n",
    "    X = np.concatenate([X_presence, X_pseudo_absence], axis=0)\n",
    "    Y = np.concatenate([np.ones((X_presence.shape[0],)), np.zeros((X_presence.shape[0],))])\n",
    "    # create training and testing df with random sampling\n",
    "    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)\n",
    "    \n",
    "    print(f'Running RF on data sample: {X_train.shape}')\n",
    "    model  = RandomForestClassifier(n_estimators=number_of_trees, max_depth = max_depth, verbose = 2)\n",
    "    \n",
    "    return model, X_train, X_test, y_train, y_test\n",
    "\n",
    "\n",
    "def fit_and_print_stats(model, X_train, y_train, X_test, y_test, columns):\n",
    "    \"\"\"\n",
    "    This function fits the model and prints the stats on the training and test datasets\n",
    "    Input:\n",
    "    model: the model to fit\n",
    "    X_train: the training dataset\n",
    "    y_train: the training labels\n",
    "    X_test: the test dataset\n",
    "    y_test: the test labels\n",
    "    columns: the columns of the dataset (list of strings that were the keys of the dictionary)\n",
    "    example usage:\n",
    "    fit_and_print_stats(model, X_train, y_train, X_test, y_test, columns)\n",
    "    \"\"\" \n",
    "    # fit model \n",
    "    model.fit(X_train, y_train)\n",
    "    # stats on training df\n",
    "    p_train = model.predict_proba(X_train)[:,1]\n",
    "\n",
    "    auc_train = sklearn.metrics.roc_auc_score(y_train, p_train)\n",
    "    print(f'AUC score on train: {auc_train:.2f}')\n",
    "    \n",
    "    # stats on test df\n",
    "    p_test = model.predict_proba(X_test)[:,1]\n",
    "    auc_test = sklearn.metrics.roc_auc_score(y_test, p_test)\n",
    "    print(f'AUC score on test: {auc_test:.2f}')\n",
    "    mse = sklearn.metrics.mean_squared_error(y_test, p_test)\n",
    "    print(f'MSE: {mse:.2f}')\n",
    "    p_test_binary = model.predict(X_test)\n",
    "    accuracy = sklearn.metrics.accuracy_score(y_test, p_test_binary)\n",
    "    print(f'accuracy: {accuracy:.2f}')\n",
    "    \n",
    "    # features impotance\n",
    "    print('I am evaluating features importance')       \n",
    "    imp = model.feature_importances_\n",
    "    \n",
    "    perc_imp_list = list()\n",
    "    list_imp_noPerc = list()\n",
    "    \n",
    "    # separate the perc featuers with the others \n",
    "    for i,j in zip(columns, imp):\n",
    "        if i.startswith('perc_'):\n",
    "            perc_imp_list.append(j)\n",
    "        else:\n",
    "            list_imp_noPerc.append(j)\n",
    "            \n",
    "    # aggregate perc importances\n",
    "    perc_imp = sum(perc_imp_list)\n",
    "    # add the aggregated result\n",
    "    list_imp_noPerc.append(perc_imp)\n",
    "    \n",
    "    # list of columns of interest\n",
    "    cols = [col for col in columns if not col.startswith('perc_')]\n",
    "    cols.append('perc')\n",
    "    \n",
    "    # print results\n",
    "    print('importances')\n",
    "    dict_imp = dict(zip(cols, list_imp_noPerc))\n",
    "    dict_imp_sorted = {k: v for k, v in sorted(dict_imp.items(), \n",
    "                                                key=lambda item: item[1], \n",
    "                                                reverse=True)}\n",
    "    for i in dict_imp_sorted:\n",
    "        print('{} : {}'.format(i, round(dict_imp_sorted[i], 2)))\n",
    "\n",
    "            \n",
    "\n",
    "def get_results( model, X_all,dem_arr, mask):\n",
    "    \"\"\"\n",
    "    This function gets the results of the model and returns a raster with the results\n",
    "    Input:\n",
    "    model: the model to fit\n",
    "    X_all: the dataset with the descriptive features\n",
    "    dem_arr: the dem array data\n",
    "    mask: the mask of the dem array with all the valid and burnable pixels\n",
    "    example usage:\n",
    "    Y_raster = get_results( model, X_all,dem_arr, mask)\n",
    "    \"\"\"\n",
    "    \n",
    "    \n",
    "    # prediction over all the points\n",
    "    Y_out = model.predict_proba(X_all)\n",
    "    # array of predictions over the valid pixels \n",
    "    Y_raster = np.zeros_like(dem_arr)\n",
    "    Y_raster[mask] = Y_out[:,1]\n",
    "    \n",
    "    # clip susc where dem exsits\n",
    "    Y_raster[~mask] = -1\n",
    "            \n",
    "    \n",
    "    return Y_raster"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "types of vegetation in the veg raster: [  0 211 212 213 221 222 223 231 241 242 243 311 312 313 321 322 323 324\n",
      " 334]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "  0%|          | 0/19 [00:00<?, ?it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processing vegetation density perc_0\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "  5%|▌         | 1/19 [00:02<00:47,  2.65s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processing vegetation density perc_211\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 11%|█         | 2/19 [00:05<00:44,  2.61s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processing vegetation density perc_212\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 16%|█▌        | 3/19 [00:07<00:41,  2.60s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processing vegetation density perc_213\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 21%|██        | 4/19 [00:09<00:31,  2.10s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processing vegetation density perc_221\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 26%|██▋       | 5/19 [00:10<00:25,  1.83s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processing vegetation density perc_222\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 32%|███▏      | 6/19 [00:11<00:21,  1.66s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processing vegetation density perc_223\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 37%|███▋      | 7/19 [00:13<00:18,  1.56s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processing vegetation density perc_231\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 42%|████▏     | 8/19 [00:14<00:16,  1.49s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processing vegetation density perc_241\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 47%|████▋     | 9/19 [00:15<00:14,  1.46s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processing vegetation density perc_242\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 53%|█████▎    | 10/19 [00:17<00:12,  1.42s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processing vegetation density perc_243\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 58%|█████▊    | 11/19 [00:18<00:11,  1.39s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processing vegetation density perc_311\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 63%|██████▎   | 12/19 [00:19<00:09,  1.37s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processing vegetation density perc_312\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 68%|██████▊   | 13/19 [00:22<00:10,  1.74s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processing vegetation density perc_313\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 74%|███████▎  | 14/19 [00:23<00:08,  1.62s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processing vegetation density perc_321\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 79%|███████▉  | 15/19 [00:25<00:06,  1.54s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processing vegetation density perc_322\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 84%|████████▍ | 16/19 [00:26<00:04,  1.49s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processing vegetation density perc_323\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 89%|████████▉ | 17/19 [00:27<00:02,  1.45s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processing vegetation density perc_324\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 95%|█████████▍| 18/19 [00:29<00:01,  1.43s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processing vegetation density perc_334\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 19/19 [00:30<00:00,  1.61s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Creating dataset for RandomForestClassifier\n",
      "Processing column: dem\n",
      "Processing column: slope\n",
      "Processing column: aspect\n",
      "Processing column: easting\n",
      "Processing column: northing\n",
      "Processing column: roughness\n",
      "Processing column: veg\n",
      "Processing column: perc_0\n",
      "Processing column: perc_211\n",
      "Processing column: perc_212\n",
      "Processing column: perc_213\n",
      "Processing column: perc_221\n",
      "Processing column: perc_222\n",
      "Processing column: perc_223\n",
      "Processing column: perc_231\n",
      "Processing column: perc_241\n",
      "Processing column: perc_242\n",
      "Processing column: perc_243\n",
      "Processing column: perc_311\n",
      "Processing column: perc_312\n",
      "Processing column: perc_313\n",
      "Processing column: perc_321\n",
      "Processing column: perc_322\n",
      "Processing column: perc_323\n",
      "Processing column: perc_324\n",
      "Processing column: perc_334\n",
      "Processing column: MWMT\n",
      "Processing column: TD\n",
      "Processing column: AHM\n",
      "Processing column: SHM\n",
      "Processing column: DDbelow0\n",
      "Processing column: DDabove18\n",
      "Processing column: MAT\n",
      "Processing column: MAP\n",
      "Processing column: Tave_sm\n",
      "Processing column: Tmax_sm\n",
      "Processing column: PPT_at\n",
      "Processing column: PPT_sm\n",
      "Processing column: PPT_sp\n",
      "Processing column: PPT_wt\n",
      "Number of burned points: 190212\n",
      " I am random sampling the dataset \n",
      "reducted df points: 19021 of 190212\n",
      "X_absence.shape[0] 6986528\n",
      "X_presence.shape[0] 19021\n",
      "Running RF on data sample: (25488, 40)\n",
      "building tree 1 of 100\n",
      "building tree 2 of 100\n",
      "building tree 3 of 100\n",
      "building tree 4 of 100\n",
      "building tree 5 of 100\n",
      "building tree 6 of 100\n",
      "building tree 7 of 100\n",
      "building tree 8 of 100\n",
      "building tree 9 of 100\n",
      "building tree 10 of 100\n",
      "building tree 11 of 100\n",
      "building tree 12 of 100\n",
      "building tree 13 of 100\n",
      "building tree 14 of 100\n",
      "building tree 15 of 100\n",
      "building tree 16 of 100\n",
      "building tree 17 of 100\n",
      "building tree 18 of 100\n",
      "building tree 19 of 100\n",
      "building tree 20 of 100\n",
      "building tree 21 of 100\n",
      "building tree 22 of 100\n",
      "building tree 23 of 100\n",
      "building tree 24 of 100\n",
      "building tree 25 of 100\n",
      "building tree 26 of 100\n",
      "building tree 27 of 100\n",
      "building tree 28 of 100\n",
      "building tree 29 of 100\n",
      "building tree 30 of 100\n",
      "building tree 31 of 100\n",
      "building tree 32 of 100\n",
      "building tree 33 of 100\n",
      "building tree 34 of 100\n",
      "building tree 35 of 100\n",
      "building tree 36 of 100\n",
      "building tree 37 of 100\n",
      "building tree 38 of 100\n",
      "building tree 39 of 100\n",
      "building tree 40 of 100\n",
      "building tree 41 of 100\n",
      "building tree 42 of 100\n",
      "building tree 43 of 100\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[Parallel(n_jobs=1)]: Done  40 tasks      | elapsed:    2.1s\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "building tree 44 of 100\n",
      "building tree 45 of 100\n",
      "building tree 46 of 100\n",
      "building tree 47 of 100\n",
      "building tree 48 of 100\n",
      "building tree 49 of 100\n",
      "building tree 50 of 100\n",
      "building tree 51 of 100\n",
      "building tree 52 of 100\n",
      "building tree 53 of 100\n",
      "building tree 54 of 100\n",
      "building tree 55 of 100\n",
      "building tree 56 of 100\n",
      "building tree 57 of 100\n",
      "building tree 58 of 100\n",
      "building tree 59 of 100\n",
      "building tree 60 of 100\n",
      "building tree 61 of 100\n",
      "building tree 62 of 100\n",
      "building tree 63 of 100\n",
      "building tree 64 of 100\n",
      "building tree 65 of 100\n",
      "building tree 66 of 100\n",
      "building tree 67 of 100\n",
      "building tree 68 of 100\n",
      "building tree 69 of 100\n",
      "building tree 70 of 100\n",
      "building tree 71 of 100\n",
      "building tree 72 of 100\n",
      "building tree 73 of 100\n",
      "building tree 74 of 100\n",
      "building tree 75 of 100\n",
      "building tree 76 of 100\n",
      "building tree 77 of 100\n",
      "building tree 78 of 100\n",
      "building tree 79 of 100\n",
      "building tree 80 of 100\n",
      "building tree 81 of 100\n",
      "building tree 82 of 100\n",
      "building tree 83 of 100\n",
      "building tree 84 of 100\n",
      "building tree 85 of 100\n",
      "building tree 86 of 100\n",
      "building tree 87 of 100\n",
      "building tree 88 of 100\n",
      "building tree 89 of 100\n",
      "building tree 90 of 100\n",
      "building tree 91 of 100\n",
      "building tree 92 of 100\n",
      "building tree 93 of 100\n",
      "building tree 94 of 100\n",
      "building tree 95 of 100\n",
      "building tree 96 of 100\n",
      "building tree 97 of 100\n",
      "building tree 98 of 100\n",
      "building tree 99 of 100\n",
      "building tree 100 of 100\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[Parallel(n_jobs=1)]: Done  40 tasks      | elapsed:    0.1s\n",
      "[Parallel(n_jobs=1)]: Done  40 tasks      | elapsed:    0.0s\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "AUC score on train: 0.98\n",
      "AUC score on test: 0.97\n",
      "MSE: 0.06\n",
      "accuracy: 0.92\n",
      "I am evaluating features importance\n",
      "importances\n",
      "perc : 0.22\n",
      "roughness : 0.12\n",
      "dem : 0.08\n",
      "slope : 0.06\n",
      "veg : 0.05\n",
      "northing : 0.05\n",
      "AHM : 0.04\n",
      "aspect : 0.04\n",
      "easting : 0.04\n",
      "MWMT : 0.03\n",
      "Tave_sm : 0.03\n",
      "PPT_sm : 0.03\n",
      "DDabove18 : 0.03\n",
      "SHM : 0.03\n",
      "PPT_wt : 0.03\n",
      "PPT_sp : 0.02\n",
      "MAP : 0.02\n",
      "MAT : 0.02\n",
      "DDbelow0 : 0.02\n",
      "TD : 0.02\n",
      "Tmax_sm : 0.02\n",
      "PPT_at : 0.01\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[Parallel(n_jobs=1)]: Done  40 tasks      | elapsed:    0.0s\n"
     ]
    }
   ],
   "source": [
    "climate_dict = assemble_climate_dict(\"hist_1991_2010\", \"199110\", clim_var_names)\n",
    "dem_paths = [dem_path_clip, slope_path, aspect_path, easting_path, northing_path, roughness_path]\n",
    "dem_labels = [\"dem\", \"slope\", \"aspect\", \"easting\", \"northing\", \"roughness\"]\n",
    "dem_dict = assemble_dem_dict(dem_paths, dem_labels)\n",
    "veg_dict, mask = assemble_veg_dictionary(clc_path_clip_nb, dem_path_clip, verbose = True)\n",
    "fires_raster = MyRaster(fires_raster_path, \"fires\")\n",
    "\n",
    "X_all, Y_all, columns = preprocessing(dem_dict, veg_dict, climate_dict, fires_raster, mask)\n",
    "\n",
    "model, X_train, X_test, y_train, y_test = prepare_sample(X_all, Y_all, percentage=0.1)\n",
    "fit_and_print_stats(model, X_train, y_train, X_test, y_test, columns)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[Parallel(n_jobs=1)]: Done  40 tasks      | elapsed:   18.4s\n"
     ]
    }
   ],
   "source": [
    "# In this cell, I get the results of the model. It is still a simple array.\n",
    "Y_raster = get_results( model, X_all, dem_dict[\"dem\"].get_data().data, mask)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "# In this cell, I convert the array to a shapefile.\n",
    "save_raster_as(Y_raster, \"/share/ander/Dev/climaax/data_cat/hazard/Hazard_output/my_suscep_present.tif\", dem_path_clip)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'Y_raster' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[3], line 4\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[38;5;66;03m# I want to extract 50, 75, 90, 95 quantiles of Y_raster[Y_raster>=0.0]. \u001b[39;00m\n\u001b[1;32m      3\u001b[0m \u001b[38;5;28mprint\u001b[39m(\n\u001b[0;32m----> 4\u001b[0m \u001b[43mY_raster\u001b[49m[Y_raster\u001b[38;5;241m>\u001b[39m\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m0.0\u001b[39m]\u001b[38;5;241m.\u001b[39mmin(),\n\u001b[1;32m      5\u001b[0m Y_raster[Y_raster\u001b[38;5;241m>\u001b[39m\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m0.0\u001b[39m]\u001b[38;5;241m.\u001b[39mmax(),\n\u001b[1;32m      6\u001b[0m Y_raster[Y_raster\u001b[38;5;241m>\u001b[39m\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m0.0\u001b[39m]\u001b[38;5;241m.\u001b[39mmean(),\n\u001b[1;32m      7\u001b[0m Y_raster[Y_raster\u001b[38;5;241m>\u001b[39m\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m0.0\u001b[39m]\u001b[38;5;241m.\u001b[39mstd(),\n\u001b[1;32m      8\u001b[0m np\u001b[38;5;241m.\u001b[39mquantile(Y_raster[Y_raster\u001b[38;5;241m>\u001b[39m\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m0.0\u001b[39m], \u001b[38;5;241m0.5\u001b[39m),\n\u001b[1;32m      9\u001b[0m np\u001b[38;5;241m.\u001b[39mquantile(Y_raster[Y_raster\u001b[38;5;241m>\u001b[39m\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m0.0\u001b[39m], \u001b[38;5;241m0.75\u001b[39m),\n\u001b[1;32m     10\u001b[0m np\u001b[38;5;241m.\u001b[39mquantile(Y_raster[Y_raster\u001b[38;5;241m>\u001b[39m\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m0.0\u001b[39m], \u001b[38;5;241m0.9\u001b[39m),\n\u001b[1;32m     11\u001b[0m np\u001b[38;5;241m.\u001b[39mquantile(Y_raster[Y_raster\u001b[38;5;241m>\u001b[39m\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m0.0\u001b[39m], \u001b[38;5;241m0.95\u001b[39m) )\n",
      "\u001b[0;31mNameError\u001b[0m: name 'Y_raster' is not defined"
     ]
    }
   ],
   "source": [
    "# I want to extract 50, 75, 90, 95 quantiles of Y_raster[Y_raster>=0.0]. \n",
    "# Since the susceptibility is the output of a random forest classifier with proba = True, \n",
    "# its outputs are distributed between 0 and 1. I need to extract the quantiles of the positive values only,\n",
    "# in order to get meaningful data from the arbitrary decisions of the classifier.\n",
    "\n",
    "print(\n",
    "Y_raster[Y_raster>=0.0].min(),\n",
    "Y_raster[Y_raster>=0.0].max(),\n",
    "Y_raster[Y_raster>=0.0].mean(),\n",
    "Y_raster[Y_raster>=0.0].std(),\n",
    "np.quantile(Y_raster[Y_raster>=0.0], 0.5),\n",
    "np.quantile(Y_raster[Y_raster>=0.0], 0.75),\n",
    "np.quantile(Y_raster[Y_raster>=0.0], 0.9),\n",
    "np.quantile(Y_raster[Y_raster>=0.0], 0.95) )\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# In this cell, I plot the susceptibility for present climate. \n",
    "import rasterio\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Open the raster\n",
    "raster_susceptibility_path = \"/share/ander/Dev/climaax/data_cat/hazard/Hazard_output/my_suscep_present.tif\"\n",
    "#raster_future_susceptibility_path = \"/share/ander/Dev/climaax/data_cat/hazard/Hazard_output/my_suscep_future.tif\"\n",
    "with rasterio.open(raster_susceptibility_path) as src:\n",
    "    # Read the raster band\n",
    "    band = src.read(1)\n",
    "\n",
    "    fig, ax = plt.subplots(figsize=(15, 15))\n",
    "    # Plot the raster\n",
    "    rasterio.plot.show(src, ax=ax)\n",
    "    # Plot the shapefile\n",
    "    catalonia.plot(ax=ax, facecolor=\"none\", edgecolor=\"red\", linewidth=2)\n",
    "    fires_2.plot(ax=ax, facecolor=\"none\", edgecolor=\"blue\", linewidth=2) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "# To project the model in the future, I need to repeat the same steps as above, but with the future climate data. \n",
    "# of course, DEM will be the same, and vegetation will be the same (in that case, it is a simplification, but it is ok for now)\n",
    "future_climate_dict = assemble_climate_dict(\"rcp45_2021_2040\", \"202140\", clim_var_names)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Creating dataset for RandomForestClassifier\n",
      "Processing column: dem\n",
      "Processing column: slope\n",
      "Processing column: aspect\n",
      "Processing column: easting\n",
      "Processing column: northing\n",
      "Processing column: roughness\n",
      "Processing column: veg\n",
      "Processing column: perc_0\n",
      "Processing column: perc_211\n",
      "Processing column: perc_212\n",
      "Processing column: perc_213\n",
      "Processing column: perc_221\n",
      "Processing column: perc_222\n",
      "Processing column: perc_223\n",
      "Processing column: perc_231\n",
      "Processing column: perc_241\n",
      "Processing column: perc_242\n",
      "Processing column: perc_243\n",
      "Processing column: perc_311\n",
      "Processing column: perc_312\n",
      "Processing column: perc_313\n",
      "Processing column: perc_321\n",
      "Processing column: perc_322\n",
      "Processing column: perc_323\n",
      "Processing column: perc_324\n",
      "Processing column: perc_334\n",
      "Processing column: MWMT\n",
      "Processing column: TD\n",
      "Processing column: AHM\n",
      "Processing column: SHM\n",
      "Processing column: DDbelow0\n",
      "Processing column: DDabove18\n",
      "Processing column: MAT\n",
      "Processing column: MAP\n",
      "Processing column: Tave_sm\n",
      "Processing column: Tmax_sm\n",
      "Processing column: PPT_at\n",
      "Processing column: PPT_sm\n",
      "Processing column: PPT_sp\n",
      "Processing column: PPT_wt\n"
     ]
    }
   ],
   "source": [
    "X_all_future, _, columns = preprocessing(dem_dict, veg_dict, future_climate_dict, fires_raster, mask)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# In this cell, I obtain the susceptibilty for the future climate. \n",
    "Y_raster_future = get_results( model, X_all_future, dem_dict[\"dem\"].get_data().data, mask)\n",
    "save_raster_as(Y_raster_future, \"/share/ander/Dev/climaax/data_cat/hazard/Hazard_output/my_suscep_future.tif\", dem_path_clip)\n",
    "\n",
    "# In the following, I plot the susceptibility for present and future climate.\n",
    "\n",
    "import rasterio\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Open the raster\n",
    "raster_susceptibility_path = \"/share/ander/Dev/climaax/data_cat/hazard/Hazard_output/my_suscep_present.tif\"\n",
    "raster_future_susceptibility_path = \"/share/ander/Dev/climaax/data_cat/hazard/Hazard_output/my_suscep_future.tif\"\n",
    "with rasterio.open(raster_susceptibility_path) as src:\n",
    "    # Read the raster band\n",
    "    band = src.read(1)\n",
    "\n",
    "    fig, ax = plt.subplots(figsize=(15, 15))\n",
    "    # Plot the raster\n",
    "    rasterio.plot.show(src, ax=ax)\n",
    "    # Plot the shapefile\n",
    "    catalonia.plot(ax=ax, facecolor=\"none\", edgecolor=\"red\", linewidth=2)\n",
    "    fires_2.plot(ax=ax, facecolor=\"none\", edgecolor=\"blue\", linewidth=2) \n",
    "\n",
    "with rasterio.open(raster_future_susceptibility_path) as src:\n",
    "    # Read the raster band\n",
    "    band = src.read(1)\n",
    "\n",
    "    fig, ax = plt.subplots(figsize=(15, 15))\n",
    "    # Plot the raster\n",
    "    rasterio.plot.show(src, ax=ax)\n",
    "    # Plot the shapefile\n",
    "    catalonia.plot(ax=ax, facecolor=\"none\", edgecolor=\"red\", linewidth=2)\n",
    "    fires_2.plot(ax=ax, facecolor=\"none\", edgecolor=\"blue\", linewidth=2)\n",
    "plt.matshow(Y_raster > 0.91)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 143,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "<>:29: SyntaxWarning: invalid escape sequence '\\i'\n",
      "<>:29: SyntaxWarning: invalid escape sequence '\\i'\n",
      "/tmp/ipykernel_3416009/2273145878.py:29: SyntaxWarning: invalid escape sequence '\\i'\n",
      "  '''\n"
     ]
    }
   ],
   "source": [
    "def corine_to_fuel_type(corine_codes_array, converter_dict, visualize_result = False):\n",
    "    \"\"\"\n",
    "    This function converts the corine land cover raster to a raster with the fuel types.\n",
    "    The fuel types are defined in the converter_dict dictionary.\n",
    "    \"\"\"\n",
    "    #\n",
    "    # mydict = dict(zip(myveg, myaggr))\n",
    "\n",
    "    converted_band = np.vectorize(converter_dict.get)(corine_codes_array)\n",
    "\n",
    "    converted_band[converted_band == None] = -1\n",
    "    #convert to int \n",
    "    converted_band = converted_band.astype(int)\n",
    "\n",
    "    if visualize_result:\n",
    "        plt.matshow(converted_band)\n",
    "        # discrete colorbar \n",
    "        cbar = plt.colorbar(ticks=[0, 1, 2, 3, 4, 5, 6])    \n",
    "    \n",
    "    return converted_band\n",
    "\n",
    "\n",
    "def susc_classes( susc_arr, quantiles):\n",
    "    '''\n",
    "    This function takes a raster map and a list of quantiles and returns a categorical raster map \n",
    "    related to the quantile classes.\n",
    "    parameters:\n",
    "    susc_arr: the susceptibility array\n",
    "    quantiles: the quantiles to use to create the classes (see np.digitize documentation)\n",
    "    '''\n",
    "    bounds = list(quantiles) \n",
    "\n",
    "    # Convert the raster map into a categorical map based on quantile values\n",
    "    out_arr = np.digitize(susc_arr, bounds, right=True)\n",
    "    out_arr = out_arr.astype(np.int8())\n",
    "\n",
    "    return out_arr\n",
    "\n",
    "def hazard_matrix( arr1, arr2): \n",
    "    '''\n",
    "    This function takes two arrays and returns a matrix with the hazard values.\n",
    "    arr1 take values on the rows -> susc., arr2 on the columns -> intensity\n",
    "    parameters:\n",
    "    arr1: the susceptibility array\n",
    "    arr2: the intensity array\n",
    "    #   s\\i  1 2 3 4\t\n",
    "    #     1! 1 2 3 4\n",
    "    #     2! 2 3 4 5\n",
    "    #     3! 3 3 5 6\n",
    "    '''\n",
    "    matrix_values = np.array([\n",
    "    [1, 2, 3, 4],\n",
    "    [2, 3, 4, 5],\n",
    "    [3, 3, 5, 6]])\n",
    "    combined_array = matrix_values[ arr1 - 1, arr2 - 1]\n",
    "\n",
    "    return combined_array\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 146,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "int64\n"
     ]
    },
    {
     "data": {
      "image/png": "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ibCxsRGf+tSn0NjYP8PtcMBoGWsc58jCaBknMIrGOkxthCPWazQiIiIiYnCQeI1O+rP6eI0e/NtB9RqNNsKIiIiIiFGNEasajYiIiIgYZAxT1WgkwoiIiIiI+qCMOhBhXXpSE6JqNCIiIiJiVCNKhBERERER9UFUjUZEREREjGqcOgWgn3GApwY/jjCqRiMiIiIiRjWiRBgRERERUR9E1WhERERExKjGMCXCqBqNiIiIiBjViBJhRERERER9MEzLMEUijIiIiIioC8rlUyj3s3pEf4/vC6JqNCIiIiJiVCNKhBERERER9UG53H/VZvQajYiIiIgYtijXwUYYiTAiIiIiYtji1ClgzPCrUB9thBERERERoxpRIoyIiIiIqA+iajQiIiIiYjSjfOoUyv1UjcbwiYiIiIiIiEFGlAgjIiIiIuqDqBqNiIiIiBjVOFUGxgw/Ioyq0YiIiIiIUY0oEUZERERE1AflMvpdoT6qRiMiIiIihivKp8oo91M1Wo6q0YiIiIiIiMFFlAgjIiIiIuqD8in0XzU6+HGEkQgjIiIiIuqCqBqNiIiIiIgYhogSYUREREREXfBiubvfqs0X8UKdelMckQgjIiIiIvqFsWPHYuLEibjvyPfr0t7EiRMxduzYurRVBGPKp0MhGxERERExovD888/j5MmTdWlr7NixOOuss+rSVhFEIoyIiIiIGNWIzjIREREREaMakQgjIiIiIkY1IhFGRERERIxqRCKMiIiIiBjViEQYERERETGqEYkwIiIiImJUIxJhRERERMSoxv8HOvjahhfjxI8AAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 528.113x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "\n",
    "# In this cell I will create the intensity matrix by converting the CLC raster.\n",
    "\n",
    "my_clc_raster = MyRaster(clc_path_clip_nb, \"clc\")\n",
    "import pandas as pd\n",
    "\n",
    "converter = pd.read_excel(\"/share/ander/Dev/climaax/CORINE_to_FuelType.xlsx\")\n",
    "converter_dict = dict(zip(converter.veg.values, converter.aggr.values))\n",
    "\n",
    "# I obtain the array of the fuel types converting the corine land cover raster\n",
    "converted_band = corine_to_fuel_type(my_clc_raster.data.data, converter_dict)\n",
    "\n",
    "# dtype of the converted band\n",
    "\n",
    "print(converted_band.dtype)\n",
    "print(converter_dict.values())\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 153,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.00090626 0.25102284]\n",
      "now I have just the susc classes [1 2 3]\n"
     ]
    }
   ],
   "source": [
    "# in this cell I obtain the hazard map for present climate crossing the susceptibility map with the intensity map.\n",
    "# compute quantiles for Y_raster \n",
    "\n",
    "quantiles = np.quantile(Y_raster[Y_raster>=0.0], [0.5, 0.75 ])\n",
    "print(quantiles)\n",
    "\n",
    "# compute discrete susc array\n",
    "susc_arr = susc_classes( Y_raster, quantiles) + 1 # I add 1 to avoid 0 values\n",
    "print(\"now I have just the susc classes\", np.unique(susc_arr))\n",
    "\n",
    "\n",
    "# compute discrete hazard \n",
    "\n",
    "hazard_arr = hazard_matrix( susc_arr, converted_band)\n",
    "\n",
    "\n",
    "Y_raster_future = MyRaster(raster_future_susceptibility_path, \"future_susceptibility\").data\n",
    "\n",
    "\n",
    "# compute susc discrete array for future \n",
    "susc_arr_future = susc_classes( Y_raster_future, quantiles) + 1 # I add 1 to avoid 0 values\n",
    "\n",
    "# compute hazard discrete array for future\n",
    "\n",
    "hazard_arr_future = hazard_matrix( susc_arr_future, converted_band)\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 158,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 528.113x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# I visualize the difference between future hazard classes and present hazard classes\n",
    "plt.matshow(hazard_arr_future - hazard_arr)\n",
    "# discrete colormap \n",
    "\n",
    "cbar = plt.colorbar(ticks=[0, 1, 2, 3, 4, 5, 6], cmap = \"Blues\")\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "climate_root = \"/share/ander/Dev/climaax/resized_climate\"\n",
    "import os\n",
    "os.path.join(climate_root, \"hist_1991_2010\", \"AHM\" + \"_199110.tif\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Structure of the ECLIPS2.0 folder \n",
    "Four different folders \n",
    "\n",
    "ECLIPS2.0_196190 -> historical data from 1961 to 1990\n",
    "ECLPS2.0_199110  -> historical data from 1991 to 2010\n",
    "ECLIPS2.0_RCP45  -> emission scenario RCP4.5\n",
    "ECLIPS2.0_RCP85  -> emission scenario RCP8.5\n",
    "\n",
    "\n",
    "\n",
    "#### Structure of the ECLIPS2.0_RCP45 folder\n",
    "\n",
    "In this folder, analogous to the other folder RCP85, there are 4 subfolders:\n",
    "\n",
    "I assumed that CCLM stands for Regional nonhydrostatic Consortium for Small-Scale Modeling (COSMO) model in Climate Mode (COSMO-CLM; abbreviated as CCLM) [1] but this is not in line with the description of the dataset [2].\n",
    "\n",
    "\n",
    "CLMC stands for Climate Limited-Area Modelling Community [2]\n",
    "In [2], the Authors specified that they used five daily bias-corrected regional climate model\n",
    "results out of nine projections available in the EURO-CORDEX database at the time of this study. The\n",
    "criteria used to select the five climate projections were as follows: (a) representation of all available RCMs and GCMs; (b) two RCMs being nested in the same driving GCM; and (c) one RCM being driven by two different GCMs. Such criteria were adopted to ensure the representativeness of all combinations of RCMs and GCMs available in the EURO-CORDEX database. The models were driven by two Representative Concentration Pathways scenarios RCP4.5 and RCP8.5 (Moss et al., 2010). The simulations were run for the EUR-11 domain with 0.11° × 0.11° horizontal resolution (Giorgi et al., 2009; Jacob et al., 2014). \n",
    "\n",
    "CLMcom_CCLM_4.5  CLMcom_CLM_4.5  DMI_HIRAM_4.5  KMNI_RAMCO_4.5  MPI_CSC_REMO2009_4.5\n",
    "\n",
    "- [1] Rolf Zentek and Günther Heinemann, Verification of the regional atmospheric model CCLM v5.0 with conventional data and lidar measurements in Antarctica, Volume 13, issue 4, Geoscientifc Model Development, 13, 1809–1825, 2020  https://doi.org/10.5194/gmd-13-1809-2020\n",
    "- [2] High-resolution gridded climate data for Europe based on bias-corrected EURO-CORDEX: The ECLIPS dataset\n",
    "Debojyoti Chakraborty, Laura Dobor, Anita Zolles, Tomáš Hlásny, and Silvio Schueler.Geoscience Data Journal,2021;8:121–131. DOI: 10.1002/gdj3.110\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Data that can be found in CLMcom_CCLM_4.5 folder \n",
    "\n",
    "|  \t\t\t\tAcronym \t\t\t              |  \t\t\t\tVariable \t\t\t\tname \t\t\t                         |  \t\t\t\tUnit \t\t\t  |\n",
    "|------------------------|-----------------------------------------|---------|\n",
    "|  \t\t\t\tMWMT \t\t\t                 |  \t\t\t\tMean \t\t\t\twarmest month temperature \t\t\t        |  \t\t\t\t°C \t\t\t    |\n",
    "|  \t\t\t\tMCMT \t\t\t                 |  \t\t\t\tMean \t\t\t\tcoldest month temperature \t\t\t        |  \t\t\t\t°C \t\t\t    |\n",
    "|  \t\t\t\tTD \t\t\t                   |  \t\t\t\tContinentality \t\t\t                        |  \t\t\t\t°C \t\t\t    |\n",
    "|  \t\t\t\tAHM \t\t\t                  |  \t\t\t\tAnnual \t\t\t\theat:moisture index \t\t\t            |  \t\t\t\t°C/mm \t\t\t |\n",
    "|  \t\t\t\tSHM \t\t\t                  |  \t\t\t\tSummer \t\t\t\theat:moisture index \t\t\t            |  \t\t\t\t°C/mm \t\t\t |\n",
    "|  \t\t\t\tDDbelow0 \t\t\t             |  \t\t\t\tDegree-days \t\t\t\tbelow 0°C \t\t\t                 |  \t\t\t\t°C \t\t\t    |\n",
    "|  \t\t\t\tDDabove5 \t\t\t             |  \t\t\t\tDegree-days \t\t\t\tabove 5°C \t\t\t                 |  \t\t\t\t°C \t\t\t    |\n",
    "|  \t\t\t\tDDbelow18 \t\t\t            |  \t\t\t\tDegree-days \t\t\t\tbelow 18°C \t\t\t                |  \t\t\t\t°C \t\t\t    |\n",
    "|  \t\t\t\tDDabove18 \t\t\t            |  \t\t\t\tDegree-days \t\t\t\tabove 18°C \t\t\t                |  \t\t\t\t°C \t\t\t    |\n",
    "|  \t\t\t\tNFFD \t\t\t                 |  \t\t\t\tNumber \t\t\t\tof frost-free days \t\t\t             |  \t\t\t\t- \t\t\t     |\n",
    "|  \t\t\t\tFFP \t\t\t                  |  \t\t\t\tLongest \t\t\t\tfrost-free period \t\t\t             |  \t\t\t\tdays \t\t\t  |\n",
    "|  \t\t\t\tbFFP \t\t\t                 |  \t\t\t\tBegining \t\t\t\tof FFP \t\t\t                       |  \t\t\t\tday \t\t\t   |\n",
    "|  \t\t\t\teFFP \t\t\t                 |  \t\t\t\tEnd \t\t\t\tof FFP \t\t\t                            |  \t\t\t\tday \t\t\t   |\n",
    "|  \t\t\t\tEMT \t\t\t                  |  \t\t\t\tExtreme \t\t\t\tminimum temperature  \t\t\t\t \t\t\t         |  \t\t\t\t°C \t\t\t    |\n",
    "|  \t\t\t\tMAT \t\t\t                  |  \t\t\t\tAnnual \t\t\t\tmean temperaure \t\t\t                |  \t\t\t\t°C \t\t\t    |\n",
    "|  \t\t\t\tMAP \t\t\t                  |  \t\t\t\tAnnual \t\t\t\ttotal precipitation  \t\t\t\t \t\t\t          |  \t\t\t\tmm \t\t\t    |\n",
    "|  \t\t\t\tTmin_an \t\t\t              |  \t\t\t\tAnnual \t\t\t\tmean of minimum temperature  \t\t\t\t \t\t\t  |  \t\t\t\t°C \t\t\t    |\n",
    "|  \t\t\t\tTmax_an \t\t\t              |  \t\t\t\tAnnual \t\t\t\tmean  of maximum temperature  \t\t\t\t \t\t\t |  \t\t\t\t°C \t\t\t    |\n",
    "|  \t\t\t\tTmax_01 \t\t\t\tto  \t\t\t\t Tmax_12 \t\t\t |  \t\t\t\tMaximum \t\t\t\tmonthly temperatures \t\t\t          |  \t\t\t\t°C \t\t\t    |\n",
    "|  \t\t\t\tTmin_01 \t\t\t\tto Tmin_12 \t\t\t   |  \t\t\t\tMinimum \t\t\t\tmonthly temperatures \t\t\t          |  \t\t\t\t°C \t\t\t    |\n",
    "|  \t\t\t\tTave_01 \t\t\t\tto Tave_12 \t\t\t   |  \t\t\t\tMean \t\t\t\tmonthly temperatures \t\t\t             |  \t\t\t\t°C \t\t\t    |\n",
    "|  \t\t\t\tTave_at \t\t\t              |  \t\t\t\tMean \t\t\t\tautumn temperature \t\t\t               |  \t\t\t\t°C \t\t\t    |\n",
    "|  \t\t\t\tTave_sm \t\t\t              |  \t\t\t\tMean \t\t\t\tsummer temperature \t\t\t               |  \t\t\t\t°C \t\t\t    |\n",
    "|  \t\t\t\tTave_sp \t\t\t              |  \t\t\t\tMean \t\t\t\tspring temperature \t\t\t               |  \t\t\t\t°C \t\t\t    |\n",
    "|  \t\t\t\tTave_wt \t\t\t              |  \t\t\t\tMean \t\t\t\twinter temperature \t\t\t               |  \t\t\t\t°C \t\t\t    |\n",
    "|  \t\t\t\tTmax_at \t\t\t              |  \t\t\t\tMaximum \t\t\t\tautumn temperature \t\t\t            |  \t\t\t\t°C \t\t\t    |\n",
    "|  \t\t\t\tTmax_sm \t\t\t              |  \t\t\t\tMaximum \t\t\t\tsummer temperature \t\t\t            |  \t\t\t\t°C \t\t\t    |\n",
    "|  \t\t\t\tTmax_sp \t\t\t              |  \t\t\t\tMaximum \t\t\t\tspring temperature \t\t\t            |  \t\t\t\t°C \t\t\t    |\n",
    "|  \t\t\t\tTmax_wt \t\t\t              |  \t\t\t\tMaximum \t\t\t\twinter temperature \t\t\t            |  \t\t\t\t°C \t\t\t    |\n",
    "|  \t\t\t\tTmin_at \t\t\t              |  \t\t\t\tMinimum \t\t\t\tautumn temperature \t\t\t            |  \t\t\t\t°C \t\t\t    |\n",
    "|  \t\t\t\tTmin_sm \t\t\t              |  \t\t\t\tMinimum \t\t\t\tsummer temperature \t\t\t            |  \t\t\t\t°C \t\t\t    |\n",
    "|  \t\t\t\tTmin_sp \t\t\t              |  \t\t\t\tMinimum \t\t\t\tspring temperature \t\t\t            |  \t\t\t\t°C \t\t\t    |\n",
    "|  \t\t\t\tTmin_wt \t\t\t              |  \t\t\t\tMinimum \t\t\t\twinter temperature \t\t\t            |  \t\t\t\t°C \t\t\t    |\n",
    "|  \t\t\t\tPPT_at \t\t\t               |  \t\t\t\tMean \t\t\t\tautumn precipitation \t\t\t             |  \t\t\t\tmm \t\t\t    |\n",
    "|  \t\t\t\tPPT_sm \t\t\t               |  \t\t\t\tMean \t\t\t\tsummer precipitation \t\t\t             |  \t\t\t\tmm \t\t\t    |\n",
    "|  \t\t\t\tPPT_sp \t\t\t               |  \t\t\t\tMean \t\t\t\tspring precipitation \t\t\t             |  \t\t\t\tmm \t\t\t    |\n",
    "|  \t\t\t\tPPT_wt \t\t\t               |  \t\t\t\tMean \t\t\t\twinter precipitation \t\t\t             |  \t\t\t\tmm \t\t\t    |\n",
    "|  \t\t\t\tPPT_01 \t\t\t\tto PPT_12 \t\t\t     |  \t\t\t\tMean \t\t\t\tmonthly precipitation \t\t\t            |  \t\t\t\tmm \t\t\t    |"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The data is structured like this:\n",
    "| Feature                 | Description  |\n",
    "|----------------------------|------------|\n",
    "| Resolution                 | 30 arcsec  |\n",
    "| Coordinate System          | WGS 84     |\n",
    "| Projection                 | CRS (\"+proj=longlat +datum=WGS84\"),  (EPSG:4326)|\n",
    "| Data Format                | GeoTIFF|\n",
    "| Extent                     | -32.65000, 69.44167, 30.87892, 71.57893  (xmin, xmax, ymin, ymax) |\n",
    "| Temporal scale             |  Past climate: mean of 1961-1990 & 1991-2010 |\n",
    "|                            | Future periods: Means of 2011-2020, 2021-2040,  2041-2060, 2061-2080,2081-2100 |\n",
    "| Climate forcing scenarios  | RCP 8.5 and RCP 4.5 |\n",
    "| Number of variables | 80 |"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The data I want is the following set of variables:\n",
    "\n",
    "- MWMT, Mean warmest month temperature\n",
    "- TD, Continentality\n",
    "- AHM, Annual Heat-Moisture Index\n",
    "- SHM, Summer Heat-Moisture Index\n",
    "- DDbelow0, Degree-days below 0°C\n",
    "- DDabove18, Degree-days above 18°C\n",
    "- MAT, Annual mean temperaure\n",
    "- MAP, Annual total precipitation\n",
    "- Tave_sm, Mean summer temperature\n",
    "- Tmax_sm, Maximum summer temperature\n",
    "- PPT_at, Mean autumn precipitation\n",
    "- PPT_sm, Mean summer precipitation\n",
    "- PPt_sp, Mean spring precipitation\n",
    "- PPT_wt, Mean winter precipitation\n",
    "\n",
    "With this descriptors I can catch some features related to fuel availability and fire danger.\n",
    "MWMT, TD, DDbelow0, DDabove18, MAT, Tave_sm, Tmax_sm are related to temperature, with a particular focus on the summer season.\n",
    "AHM, SHM, are related to the interplay between heat and moisture, also with a focus on summer period.\n",
    "MAP, PPT_at, PPT_sm, PPt_sp, PPT_wt are related to precipitation, with a particular focus on the seasonality.\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.1"
  },
  "orig_nbformat": 4
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
