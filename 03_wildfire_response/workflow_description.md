# Wildfire workflow: FWI response to climate change and affected population

:::{warning}

This workflow is under construction.
:::


This workflow is designed to support the regional estimation of wildfire hazard and the affected population in the context of climate change.

Wildfire hazard depends on the local climatic conditions of a region, among other factors.
In a changing climate, we can therefore expect changes to the potential for fire development.
While an increase in mean temperature is likely to extend the length of the traditional fire season in already susceptible regions, the atmosphere is also able to hold and carry additional moisture in a warmer climate, potentially offseting an increase of fire hazard based on temperature alone.
It is therefore important to consider multiple influential variables in combination to get a reliable picture of the response of wildfire susceptibility to climate change.


## Hazard assessment methodology

In this workflow, the potential for wildfire development is quantified by the **Fire Weather Index (FWI)**, which combines multiple atmospheric variables into an assessment of fire danger.
The response of local FWI statistics is modelled as a function of changes to both mean temperature and precipitation (relative to a baseline climate), using the methodology and simulations of [El Garroussi et al. (2024)](https://doi.org/10.1038/s41612-024-00575-8).
The constructed response model can be applied to projections of temperature and precipitation from climate models to estimate future changes to regional FWI statistics and their sensitivity to model uncertainty.

Hazard assessment in three parts (one notebook each):

1. [Historical climate](hazard_assessment_historical.ipynb)
2. [Response surface methodology](hazard_assessment_response.ipynb)
3. [Climate projections](hazard_assessment_projections.ipynb)


### Advantages

- Once constructed, the response model can be used to estimate FWI statistics for a wide range of temperature and precipitation change scenarios.
  Projections of temperature and precipitation change can be used for which no corresponding FWI data are available.
- An impact response surface can easily be visualized and analysed in the space of mean temperature and precipitation changes, e.g. by plotting the yearly probability of exceedance for a given FWI threshold.
- The response model predicts statistics of the FWI, even if the temperature and precipitation change under consideration would otherwise not be suitable to reliably compute statistics, e.g., due to limited temporal coverage.
- Sensitivity to uncertainty along a given temperature-precipitation change trajectory can be estimated with the response model.

### Limitations

- The construction of the FWI aims to account for the effects of fuel moisture and wind on fire behaviour.
  Values are a function of multiple atmospheric variables and their integrated temporal evolution, thereby assessing the *potential* for wildfire.
  The index does not take the actual land use and availability of burnable material into account.
- Computing a the response of FWI statistics as a function of temperature and precipitation only is a simplification of climate change and the FWI, which, e.g., also depends on wind and moisture conditions.


### Important concepts

- **Fire Weather Index** (FWI):
  The FWI is a component of the Canadian Forest Fire Weather Index System, serving as a numerical measure of fire intensity.
  It is derived from weather observations including temperature, humidity, wind speed, and precipitation.
  The FWI is instrumental in estimating the potential land flammability and severity of wildfires, thereby helping in fire management and mitigation efforts.
- **Probability of exceedance**:
  This statistical measure indicates the likelihood that a certain variable, in our case the FWI, will surpass a predefined threshold.
  It is expressed as a probability, ranging from 0 to 1, where a higher value signifies a greater chance of exceeding the specified FWI threshold.
  Understanding the probability of exceedance is essential for assessing fire risk under various climate scenarios and making informed decisions for fire prevention and safety planning.
- **Impact response surface** (IRS):
  A model for the response of a dependent variable (here: the probability of exceedance of a FWI threshold) to changes in a set of explanatory variables (here: mean temperature and precipitation).
  The IRS is constructed empirically from a reference dataset of both the explanatory and corresponding dependent variables.
  Once constructed, it can be used to evaluate a response and the associated sensitivity to uncertainty for given values of the explanatory variables within the range covered by the reference dataset.


## Risk assessment methodology

Count the population affected at specific levels of hazard.

- [Affected population](risk_assessment_population.ipynb)


## Datasets

### Simulations of FWI with CMIP6-informed perturbations

The reference dataset from which the response model is built consists of a set of simulations of historical FWI, based on ERA5 reanalysis data.
Simulations in the set are perturbed such that the mean temperature change covers a range from 0 to +5°C and the mean precipitation change a range from -40% to +60% relative to the actual historical climate as represented by the reanalysis data.
The perturbations imposed on the simulations are informed by CMIP6 model runs such that they represent realistic patterns of climate change.

The output of the simulations is available from Zenodo ([El Garroussi 2024](https://doi.org/10.5281/zenodo.10458186)).
For this workflows, we offer to possibility to access the dataset through a mirror that is provided by the CLIMAAX project.
The mirror provides a more convenient data layout for a local analysis as carried out in this workflow.

### Yearly climate indicators from reanalysis and EURO-CORDEX

TODO: [Climate indicators for Europe from 1940 to 2100 derived from reanalysis and climate projections](https://cds.climate.copernicus.eu/datasets/sis-ecde-climate-indicators?tab=overview) from CDS


### Opportunities to include local data

The impact response surface methodology allows for the use of a large variety of mean temperature and precipitation projections for the future climate.
Workflow users are encouraged to supply own projections that have been tailored to their specific region of interest, e.g., due to a high(er)-resolution regional modelling effort and/or bias correction.

### Global population projections for 5 SSP scenarios

TODO: [Wang et al. (2022)](https://doi.org/10.6084/m9.figshare.19608594.v3).


## Workflow outputs

The workflow produces an impact response surface for the probability of exceedance of a selected FWI threshold for a region of interest and as a function of temperature and precipitation.
Projections of mean temperature and precipitation change are overlaid on the IRS to estimate changes in FWI statistics and their sensitivity to projection uncertainty.

TODO: show figures with example outputs


## References

- El Garroussi, S., Di Giuseppe, F., Barnard, C. et al. Europe faces up to tenfold increase in extreme fires in a warming climate. npj Clim Atmos Sci 7, 30 (2024). https://doi.org/10.1038/s41612-024-00575-8
- El Garroussi, S. (2024). 30-Year Canadian Fire Weather Index Simulations over Europe: CMIP6-Informed Temperature and Precipitation Perturbations [Data set]. Zenodo. https://doi.org/10.5281/zenodo.10458186
