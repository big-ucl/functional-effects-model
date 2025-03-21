# TITLE ( "panel data" OR "panel effects" OR "longitudinal" OR "panel" OR "mixed effects" OR "random effects" ) AND EXACTSRCTITLE ( "choice modelling" OR "Transportation research" OR "travel behaviour" OR "transportation letters" ) AND ALL ( "open data" OR "open-data" OR "github" )

1. Longitudinal and spatial analysis of Americans’ travel distances following COVID-19
    * data not available
2. Review of Learning-Based Longitudinal Motion Planning for Autonomous Vehicles: Research Gaps Between Self-Driving and Traffic Congestion
    * not relevant, motion of scope for autonomous vehicle
3. Longitudinal analysis of transit-integrated ridesourcing users and their trips
    * data not available
4. Data selection in machine learning for identifying trip purposes and travel modes from longitudinal GPS data collection lasting for seasons
    * Not panel data, GPS trajectory data
5. Predictors of driving cessation in community-dwelling older adults: A 3-year longitudinal study
    * Data not directly openly available. May be if contact the Korea Institute for Health and Social Affairs but not clear if it would be free or not

# TITLE ( "panel data" OR "panel effects" OR "longitudinal" ) AND EXACTSRCTITLE ( "choice modelling" )

1. Longitudinal investigation of skeletal activity episode timing decisions – A copula approach
    * Not applicable, investigate correlation between activity start time and duration.
2. What sort of Brexit do the British people want? A longitudinal study examining the ‘trade-offs’ people would be willing to make in reaching a Brexit deal
    * Data not available
3. Estimating installed-base effects in product adoption: Borrowing IVs from the dynamic panel data literature
    * Simulated data
4.  Using panel data for modelling duration dynamics of outdoor leisure activities
    *  Relevant - waiting to access data to dutch mobility panel
    *  OK
5. Estimation of an unbalanced panel data Tobit model with interactive effects
    * Potentially ok but need to register to the Chinese Family Panel Studies
6.  Triggers of behavioral change: Longitudinal analysis of travel behavior, household composition and spatial characteristics of the residence
    * Data old and doesn't seem to be easily accessible. Potentially ok.
7. Location choice with longitudinal WiFi data
    * Data not available, not socio-demographic characteristics

# ALL("easyshare" AND ("panel" or "longitudinal"))

1. Look before you leap: Earnings gaps and elderly self-employment
    * Not panel.
2. Exposome Determinants of Quality of Life in Adults Over 50: Personality Traits, Childhood Conditions, and Long-Term Unemployment in SHARELIFE Retrospective Panel
    * Not using easySHARE.
3. ProInsight: A Tool for Risk Prediction and Impact Evaluation of Digital Health Solution Implementations
    * Not choice related. Extrapolation of longitudinal factors over the waves with cubic B-splines.
4. Functional Graph Convolutional Networks: A Unified Multi-task and Multi-modal Learning Framework to Facilitate Health and Social-Care Insights
    * Use og GCN to do regression/classification or forecast on multimlongitudinal factors. Not really relevant in our case
5. A New Computationally Efficient Algorithm to solve Feature Selection for Functional Data Classification in High-dimensional Spaces
    * Not applicable in our case, feature selection algorithm
6. Shielded by Education? The Buffering Role of Education in the Relationships Between Changes in Mental Health and Physical Functioning Through Time Among Older Europeans
    * OK
    * Use of wave 5 and 6 to analyse mental health. Some variables from SHARE, some from easySHARE, seems reproducible and used a mixed effect model so relevant
7. The effects of COVID-19-era unemployment and business closures upon the physical and mental health of older Europeans: Mediation through financial circumstances and social activity
    * Not interested in COVID-19, not the focus of our model.
8. Conditional on the Environment? The Contextual Embeddedness of Age, Health, and Socioeconomic Status as Predictors of Remote Work among Older Europeans through the COVID-19 Pandemic
    * Not interested in COVID-19, not the focus of our model. 
9. Symptom Network Analysis Tools for Applied Researchers With Cross-Sectional and Panel Data – A Brief Overview and Multiverse Analysis
    * Comparison of methodologies to investigate mental health on longitudinal data. Use of easySHARE quality of life index, depressive symptoms, and amount of alcohol.
10. Prospective associations between hand grip strength and subsequent depressive symptoms in men and women aged 50 years and older: insights from the Survey of Health, Aging, and Retirement in Europe
    * investigate the relation between hand grip and depression.
11. Predictors of loneliness onset and maintenance in European older adults during the COVID-19 pandemic
    * Not using easySHARE
12. Who uses technical aids in old age? Exploring the implementation of technology-based home modifications in Europe
    * Not panel model
13. Healthy Life Expectancy of People Over Age 65: Results of the Russian Epidemiological Study EVCALIPT
    * Not panel and not using easySHARE
14. The Role of Country-Level Availability and Generosity of Healthcare Services, and Old-Age Ageism for Missed Healthcare during the COVID-19 Pandemic Control Measures in Europe
    * Not panel model
15. The Health Effects of Workforce Involvement and Transitions for Europeans 50-75 Years of Age: Heterogeneity by Financial Difficulties and Gender
    * model self-perc3eived health but not exactly panel model (only variables from w6 to predict w7 self-perceived health)
16. The effect of job quality on quality of life and wellbeing in later career stages: A multilevel and longitudinal analysis on older workers in Europe
    * model the wellbeing (CASP-12). Panel data but not with easySHARE.
17. The role of migration status in the link between ADL/IADL and informal as well as formal care in Germany: Findings of the Survey of Health, Aging and Retirement in Europe
    * Not panel, only germany, model informal and formal care
18. New Variations of Random Survival Forests and Applications to Age-Related Disease Data
    * Survival analysis on the disease with Ranfom Forest variation. Methodological
19. Age and sex trends in depressive symptoms across middle and older adulthood: Comparison of the Canadian Longitudinal Study on Aging to American and European cohorts
    * Not panel
20. Inequality of educational opportunity at time of schooling predicts cognitive functioning in later adulthood
    * Mixed-effect model to predict from the inequality of educational opportunity (also form parents) the cognitive ability
    * Maybe OK
    * Not easySHARE
21. Does Social Isolation Affect Medical Doctor Visits? New Evidence Among European Older Adults
    * Model the number of visits of the GP. OK. waves 1-6,. Static and dynamic model with individual model. Really relevant.
22. Job status and depressive symptoms in older employees: An empirical analysis with SHARE (Survey of health, ageing and retirement in Europe) data
    * Not panel model
23. Siblings caring for their parents across Europe: Gender in cross-national perspective
    * Not panel (only wave 6)
24. Linking mediterranean diet and lifestyle with cardio metabolic disease and depressive symptoms: A study on the elderly in europe
    * OK. modelling chronic CMD, BMI, or level of depressive symptoms from socio-demographic characteristics and mediteranean diet with a fixed effects model.
25. Patterns of healthy aging and household size dynamics in Western Europe
    * Not a modelling paper
26. The Relation of Physical Activity and Self-Rated Health in Older Age - Cross Country Analysis Results from SHARE
    * Not panel (only using wave 6)
27. Self-employment, depression, and older individuals: A cross-country study
    * linking self-employment to depressive symptoms. Not clear what methods they are using. Not reproducible
28. Implementation and Methodological Framework of the SHARE study in Croatia
    * Not relevant. How they conducted the SHARE survey in Croatia.
29. Health-risk behaviours in objective and subjective health among croatians aged 50 and older
    * Not panel, only wave 6
30. Season of birth, health and aging
    * Weird paper where they use season of birth as a predictor of a constructed health index. Not reproducible since the health index is not recompuable. Fixed-effect model.
31. Positive Aging Views inthe General PopulationPredict BetterLong-Term Cognitionfor Elders in EightCountries
    * Not reproducible because of a constructed age status.
32. Associations of childhood health and financial situation with quality of life after retirement-regional variation across Europe
    * Not panel, only wave 5
33. Long-run improvements in human health: Steady but unequal
    * Based on a health deficit index that is hardly reproducible.
34. What is Happening with Quality of Life Among the Oldest People in Southern European Countries? An Empirical Approach Based on the SHARE Data
    * Not panel, only wave 6
35. Hungry children age faster
    * Based on a health index hardly reproducible
36. Inference With Difference-in-Differences With a Small Number of Groups
    * methodologically not applicable
37. Softcopy quality ruler method: implementation and validation
    * Not the same easySHARE, not relevant at all, camera measurement.