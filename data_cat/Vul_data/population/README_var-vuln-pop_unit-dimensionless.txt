Population vulnerability in the European Wildfire Risk Assessment

Aggregated index of population vulnerability.  Based on the population exposed in the vulnerable Wildland Urban Interface derived after https://doi.org/10.2760/46951 [preliminary prototype approximation].
The maximum extent of possible values ranges from 0 to 1 (100 %), continuous values, so that the data satisfy the array-based semantic constraint ::proportion:: (semantic array programming).
In each spatial cell c, the variable is expressed as percentile (%), i.e. as percentage of the other cells whose value is less than in c.

Layer name : Population vulnerability
Layer group: Vulnerability
Layer unit : Percentile (%)
CSV file   : var-vuln-pop_unit-dimensionless.csv
Projection : the native geospatial projection is EURO-CORDEX
             (https:euro-cordex.net).  European grid of the Coordinated
             Regional Climate Downscaling Experiment (CORDEX) by World Climate
             Research Programme (WCRP), https://purl.org/INRMM-MiD/z-THSLFEYQ .
             The standard grid EUR-11 is used (grid cell resolution of
             approximately 0.11 degrees, or about 12.5 km).

             Available reprojection information for WGS84
             and Lambert Azimuthal Equal-Area projection (ETRS89-LAEA).


CSV fields:
- id                        : unique id of each EURO-CORDEX cell.
- lon-1, lon-2, lon-3, lon-4: longitude (WGS84 projection) of each corner
                              in the ‹id›-th EURO-CORDEX cell.
- lat-1, lat-2, lat-3, lat-4: latitude (WGS84 projection) of each corner
                              in the ‹id›-th EURO-CORDEX cell.
- x-pos, y-pos              : position of the ‹id›-th EURO-CORDEX cell
                              in the EURO-CORDEX matrix.
- the last column lists the values of the variable (NaN: missing data).


A GeoTIFF version of the CSV file is also enclosed, rasterised in ETRS89-LAEA.  Please, note that the EURO-CORDEX spatial cells (grid cell resolution of approximately 0.11 degrees, or about 12.5 km, fully described in the CSV file) remain the reference mapping unit even for the GeoTIFF reprojected version, although for reprojecting this version in ETRS89-LAEA the native EURO-CORDEX cells were subject to a technical upsampling in contiguous clusters of pixels with higher resolution.



© European Union, 2019-2022

The Commission's reuse policy is implemented by the Commission Decision of 12 December 2011 on the reuse of Commission documents.
   European Commission, 2011. Commission Decision of 12 December 2011 on the
   reuse of Commission documents (2011/833/EU).
   Official Journal of the European Union 54 (L330), 39-42.
   http://data.europa.eu/eli/dec/2011/833/oj

Unless otherwise indicated (e.g. in individual copyright notices), content owned by the EU on this website is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) licence (https://creativecommons.org/licenses/by/4.0/). This means that reuse is allowed, provided appropriate credit is given and changes are indicated.

When using these data and information, please cite the relevant sources to acknowledge the appropriate credit (see the references below).


The European Commission shall not be liable for any consequence stemming from the reuse of this publication.
Neither the European Commission nor any person acting on behalf of the Commission is responsible for the use that might be made of the data and information provided in this archive.

European Commission legal notice: https://ec.europa.eu/info/legal-notice_en




References


[1] Oom, D., de Rigo, D., San-Miguel-Ayanz, J., Artes-Vivancos, T., Boca, R., Branco, A., Campanharo, W.A., Grecchi, R., Houston Durrant, T., Ferrari, D., Libertà, G., Maianti, P., Pfeiffer, H., 2021. Wildfires. In: Poljanšek, K., Valles, A.C., Ferrer, M.M. (Eds.), Recommendations for National Risk Assessment for Disaster Risk Management in EU: Where Science and Policy Meet - Version 1. Publications Office of the European Union, Luxembourg, pp. 93-105. ISBN: 978-92-76-30256-8 https://doi.org/10.5281/zenodo.6045338

[2] Oom, D., de Rigo, D., Pfeiffer, H., San-Miguel-Ayanz, J., Grecchi, R., Durrant, T.H., Libertà, G., Artes-Vivancos, T., Boca, R., Maianti, P., Branco, A., Ferrari, D., 2020. Developing the European wildfire risk assessment (WRA). In: Atlas of the Human Planet 2020. Publications Office of the European Union, Luxembourg, pp. 37-38. ISBN: 978-92-76-27388-2 https://purl.org/INRMM-MiD/z-QW8I46GC

[3] San-Miguel-Ayanz, J., Costa, H., de Rigo, D., Libertà, G., Artés Vivancos, T., Houston Durrant, T., Nuijten, D., Löffler, P., Moore, P., Baetens, J., Konstantinov, V., Duche, Y., Joannelle, P., Debreceni, P., Nagy, D., Zaken, A.B., Mitri, G., Assali, F., Alaoui, H.M., Piwnicki, J., Szczygieł, R., Almeida, R., Mara, S., Eritsov, A., Sandahl, L., Moffat, A., Gazzard, R., 2019. Basic criteria to assess wildfire risk at the pan-European level. Publications Office of the European Union, Luxembourg. ISBN: 978-92-79-98200-2 https://doi.org/10.2760/052345

[4] San-Miguel-Ayanz, J., Chuvieco, E., Handmer, J., Moffat, A., Montiel-Molina, C., Sandahl, L., Viegas, D., 2017. Climatological risk: wildfires. In: Poljanšek, K., Marín Ferrer, M., De Groeve, T., Clark, I. (Eds.), Science for disaster risk management 2017: knowing better and losing less. Publications Office of the European Union, pp. 294-305. ISBN: 978-92-79-60679-3 , https://purl.org/INRMM-MiD/c-14445352

