# Risk assessment for wildfire (machine learning approach)

In this workflow, hazard and risk maps, related to aggregated time windows of 20 years, are given. This workflow relies on machine learning approach to understand the drivers of the fire activity, crossing past wildfires with geophysical, vegetation and climatic information; such knowledge is used to produce susceptibility (likelihood) maps who will depend on the climatic and vegetation characteristics, who can change in time according to different scenarios.  


## Hazard assessment methodology

This analysis is based on the hazard mapping works of Tonini et al. 2020, Trucchia et al. 2023.

The workflow is based on the following steps:

* Data gathering and preprocessing.
* Building a model for wildfire susceptibility using present climate conditions and synoptic wildfire events.
* Projecting the model to future climate conditions.

For both present and future conditions, susceptibility can be evolved to hazard by considering the different plant functional types, which are a proxy for the intensity of potential wildfires. Hazard is given as a discrete indicator ranging from 1 to 6.

Regarding **climate**, the analysis revolves around a High-resolution gridded climate data for Europe based on bias-corrected EURO-CORDEX: the ECLIPS-2.0 dataset. ECLIPS (European CLimate Index ProjectionS) dataset contains gridded data for 80 annual, seasonal, and monthly climate variables for two past (1961-1990, 1991-2010) and five future periods (2011-2020, 2021-2140, 2041-2060, 2061-2080, 2081-2100). The future data are based on five Regional Climate Models (RCMs) driven by two greenhouse gas concentration scenarios, RCP 4.5 and 8.5. See _(Chakraborty et al. 2021)_ for more details.

Regarding **vegetation**, the standard approach if no special input is given adopts the [CORINE land cover](https://land.copernicus.eu/en/products/corine-land-cover) at the third level of aggregation. With some preprocessing, it is possible to use other fuel cover - land cover raster or vector maps as input, as long as the user specifies a new conversion table between the land use/land cover classes and the plant functional types (grassland, shrubs, broadleaves, conifersm etc.) and changes the default routines for the calculation of wildland urban interface from anthropic land use patches.

Regarding past **history of wildfires**, if no special input is given, the standard approach is to use the EFFIS database, which contains information on the location, date, and size of wildfires in the pan-European region. However, the approach is of course compatible with any given repository of vector data for wildfires. With a little preprocessing, it is possible to use wildfire databases in raster or netcdf format, while the presented algorithm is based on the use of vector data.

:::{list-table}
:header-rows: 1

* - Required data
  - Format
  - Sample data (Catalonia)
* - Boundaries of your area of interest
  - Vector data, e.g., shapefile or geojson.
  - `data_Catalonia/boundaries`
* - Digital elevation model (DEM)
  - Raster, e.g., GeoTIFF. A resolution between 50 and 500 m is recommended.
  - `data/dem2_3035.tif` (Spain, 100 m resolution)
* - Land cover
  - Raster, e.g., GeoTIFF. Automatic download included in CHELSA notebook.
  - `data/veg_corine_reclass.tif`
* - Historical fires database
  - Vector or raster. The workflow includes a step to rasterize vector data.
  - `data_Catalonia/fires`
:::


:::{tip}

Use the [data request form from EFFIS](https://forest-fire.emergency.copernicus.eu/apps/data.request.form/) to download a fire database (burnt area) based on satellite information compatible with the workflow.
:::


## Risk Assessment Methodology

The risk assessment methodology is based on the hazard maps produced in the previous step and the exposure data. The exposure data is based on primary layers, and derived layers. In the first category we use exposed elements from Open Street Map (roads, hospitals, schools) and polygons of wildland-urban interfaces, derived from the land use input (the default one is again the CORINE). In the second category we use the different rasters of vulnerability (that is, which include the exposed element densities and their vulnerability to fire) made available by EFFIS: population vulnerability, ecological vulnerability, and economic vulnerability. In both cases, risk is obtained by crossing hazard and exposure  (or exposure times vulnerability) data by the means of contingency tables. The risk is given as a discrete indicator ranging from 1 to 4.

The produced risk maps can be used to identify the most critical areas and to plan the most effective risk mitigation strategies. In the postprocesing phase, the risk can be aggregated according to NUTS regions, administrative boundaries, or any other user-defined area. The spatialised risk trends with respect to present climate risk map can be analysed, to understand where intensification of risk can be expected under several climate scenarios.



## References

(Chakraborty et al. 2021) Chakraborty D.; Dobor L.; Zolles A.; Hlásny T.; Schueler S. High-resolution gridded climate data for Europe based on bias-corrected EURO-CORDEX: The ECLIPS dataset. Geosci Data J. 2021, 8, 121–131. https://doi.org/10.1002/gdj3.110 

(EC-JRC et al. 2020) European Commission, Joint Research Centre, Costa, H.; De Rigo, D.; Libertà, G.; et al. European wildfire danger and vulnerability in a changing climate – Towards integrating risk dimensions – JRC PESETA IV project – Task 9 - forest fires. Publications Office of the European Union. 2020. https://data.europa.eu/doi/10.2760/46951  

(Oom et al., 2022) Oom, D.; de Rigo, D.; Pfeiffer, H.; Branco, A.; Ferrari, D.; Grecchi, R.; Artés-Vivancos, T.; Houston Durrant, T.; Boca, R.;
Maianti, P.; Libertá, G.; San-Miguel-Ayanz, J.; et a.l. Pan-European wildfire risk assessment, EUR 31160 EN, Publications Office of
the European Union, Luxembourg. 2022. ISBN 978-92-76-55137-9. doi:10.2760/9429. JRC130136

(Tonini et al. 2020) Tonini, M.; D’Andrea, M.; Biondi, G.; Degli Esposti, S.; Trucchia, A.; Fiorucci, P. A Machine Learning-Based Approach for Wildfire Susceptibility Mapping. The Case Study of the Liguria Region in Italy. Geosciences 2020, 10, 105. https://doi.org/10.3390/geosciences10030105 

(Trucchia et al. 2022) Trucchia, A.; Meschi, G.; Fiorucci, P.; Gollini, A.; Negro, D. Defining Wildfire Susceptibility Maps in Italy for Understanding Seasonal Wildfire Regimes at the National Level. Fire 2022, 5, 30. https://doi.org/10.3390/fire5010030 

(Trucchia et al. 2023) Trucchia, A.; Meschi, G.; Fiorucci, P.; Provenzale, A.; Tonini, M.; Pernice, U.; Wildfire hazard mapping in the eastern Mediterranean landscape. International Journal of Wildland Fire 2023, 32, 417-434.


## Contributors 

- Farzad Ghasemiazma
- Andrea Trucchia 
- Giorgio Meschi
- Guido Biondi 
- Nicola Rebora
- Paolo Fiorucci
