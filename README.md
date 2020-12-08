# Pymaceuticals
# Dependencies and Setup
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
import numpy as np

# Study data files
mouse_metadata_path = "data/Mouse_metadata.csv"
study_results_path = "data/Study_results.csv"

# Read the mouse data and the study results
mouse_metadata = pd.read_csv("../Resources/Mouse_metadata.csv")
study_results = pd.read_csv("../Resources/Study_results.csv")

# Combine the data into a single dataset
merged = mouse_metadata.merge(study_results, how="right", on='Mouse ID')

# Display the data table for preview
merged.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mouse ID</th>
      <th>Drug Regimen</th>
      <th>Sex</th>
      <th>Age_months</th>
      <th>Weight (g)</th>
      <th>Timepoint</th>
      <th>Tumor Volume (mm3)</th>
      <th>Metastatic Sites</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>k403</td>
      <td>Ramicane</td>
      <td>Male</td>
      <td>21</td>
      <td>16</td>
      <td>0</td>
      <td>45.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>k403</td>
      <td>Ramicane</td>
      <td>Male</td>
      <td>21</td>
      <td>16</td>
      <td>5</td>
      <td>38.825898</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>k403</td>
      <td>Ramicane</td>
      <td>Male</td>
      <td>21</td>
      <td>16</td>
      <td>10</td>
      <td>35.014271</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>k403</td>
      <td>Ramicane</td>
      <td>Male</td>
      <td>21</td>
      <td>16</td>
      <td>15</td>
      <td>34.223992</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>k403</td>
      <td>Ramicane</td>
      <td>Male</td>
      <td>21</td>
      <td>16</td>
      <td>20</td>
      <td>32.997729</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Checking the number of mice.
len(merged['Mouse ID'].unique())
```




    249




```python
# Getting the duplicate mice by ID number that shows up for Mouse ID and Timepoint. 
duplicated = merged.loc[merged.duplicated(subset=["Mouse ID", "Timepoint"]), "Mouse ID"].unique()
duplicated

```




    array(['g989'], dtype=object)




```python
# Optional: Get all the data for the duplicate mouse ID. 


```


```python
# Create a clean DataFrame by dropping the duplicate mouse by its ID.
clean = merged.drop_duplicates(subset=None, keep='first', inplace=False)
clean.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mouse ID</th>
      <th>Drug Regimen</th>
      <th>Sex</th>
      <th>Age_months</th>
      <th>Weight (g)</th>
      <th>Timepoint</th>
      <th>Tumor Volume (mm3)</th>
      <th>Metastatic Sites</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>k403</td>
      <td>Ramicane</td>
      <td>Male</td>
      <td>21</td>
      <td>16</td>
      <td>0</td>
      <td>45.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>k403</td>
      <td>Ramicane</td>
      <td>Male</td>
      <td>21</td>
      <td>16</td>
      <td>5</td>
      <td>38.825898</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>k403</td>
      <td>Ramicane</td>
      <td>Male</td>
      <td>21</td>
      <td>16</td>
      <td>10</td>
      <td>35.014271</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>k403</td>
      <td>Ramicane</td>
      <td>Male</td>
      <td>21</td>
      <td>16</td>
      <td>15</td>
      <td>34.223992</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>k403</td>
      <td>Ramicane</td>
      <td>Male</td>
      <td>21</td>
      <td>16</td>
      <td>20</td>
      <td>32.997729</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Checking the number of mice in the clean DataFrame.
count_merged = clean['Mouse ID'].count()
count_merged
```




    1892



## Summary Statistics


```python
# Generate a summary statistics table of mean, median, variance, standard deviation, and SEM of the tumor volume for each regimen

# Use groupby and summary statistical methods to calculate the following properties of each drug regimen: 
# mean, median, variance, standard deviation, and SEM of the tumor volume. 
# Assemble the resulting series into a single summary dataframe.
means = merged.groupby('Drug Regimen').mean()["Tumor Volume (mm3)"]
medians = merged.groupby('Drug Regimen').median()["Tumor Volume (mm3)"]
var = merged.groupby('Drug Regimen').var()["Tumor Volume (mm3)"]
std = merged.groupby('Drug Regimen').std()["Tumor Volume (mm3)"]
sem = merged.groupby('Drug Regimen').sem()["Tumor Volume (mm3)"]

# # Assemble the resulting series into a single summary dataframe.
Regimen = pd.DataFrame({"Mean Tumor Vol": means,
                       "Median": medians,
                       "Variance": var,
                       "Std": std,
                       "Standard Error": sem})
Regimen
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mean Tumor Vol</th>
      <th>Median</th>
      <th>Variance</th>
      <th>Std</th>
      <th>Standard Error</th>
    </tr>
    <tr>
      <th>Drug Regimen</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Capomulin</th>
      <td>40.675741</td>
      <td>41.557809</td>
      <td>24.947764</td>
      <td>4.994774</td>
      <td>0.329346</td>
    </tr>
    <tr>
      <th>Ceftamin</th>
      <td>52.591172</td>
      <td>51.776157</td>
      <td>39.290177</td>
      <td>6.268188</td>
      <td>0.469821</td>
    </tr>
    <tr>
      <th>Infubinol</th>
      <td>52.884795</td>
      <td>51.820584</td>
      <td>43.128684</td>
      <td>6.567243</td>
      <td>0.492236</td>
    </tr>
    <tr>
      <th>Ketapril</th>
      <td>55.235638</td>
      <td>53.698743</td>
      <td>68.553577</td>
      <td>8.279709</td>
      <td>0.603860</td>
    </tr>
    <tr>
      <th>Naftisol</th>
      <td>54.331565</td>
      <td>52.509285</td>
      <td>66.173479</td>
      <td>8.134708</td>
      <td>0.596466</td>
    </tr>
    <tr>
      <th>Placebo</th>
      <td>54.033581</td>
      <td>52.288934</td>
      <td>61.168083</td>
      <td>7.821003</td>
      <td>0.581331</td>
    </tr>
    <tr>
      <th>Propriva</th>
      <td>52.322552</td>
      <td>50.854632</td>
      <td>42.351070</td>
      <td>6.507770</td>
      <td>0.512884</td>
    </tr>
    <tr>
      <th>Ramicane</th>
      <td>40.216745</td>
      <td>40.673236</td>
      <td>23.486704</td>
      <td>4.846308</td>
      <td>0.320955</td>
    </tr>
    <tr>
      <th>Stelasyn</th>
      <td>54.233149</td>
      <td>52.431737</td>
      <td>59.450562</td>
      <td>7.710419</td>
      <td>0.573111</td>
    </tr>
    <tr>
      <th>Zoniferol</th>
      <td>53.236507</td>
      <td>51.818479</td>
      <td>48.533355</td>
      <td>6.966589</td>
      <td>0.516398</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Generate a summary statistics table of mean, median, variance, standard deviation, and SEM of the tumor volume for each regimen

# Using the aggregation method, produce the same summary statistics in a single line
agg = merged.groupby('Drug Regimen').agg(['mean', 'median', 'var', 'std', 'sem'])['Tumor Volume (mm3)']
agg
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>median</th>
      <th>var</th>
      <th>std</th>
      <th>sem</th>
    </tr>
    <tr>
      <th>Drug Regimen</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Capomulin</th>
      <td>40.675741</td>
      <td>41.557809</td>
      <td>24.947764</td>
      <td>4.994774</td>
      <td>0.329346</td>
    </tr>
    <tr>
      <th>Ceftamin</th>
      <td>52.591172</td>
      <td>51.776157</td>
      <td>39.290177</td>
      <td>6.268188</td>
      <td>0.469821</td>
    </tr>
    <tr>
      <th>Infubinol</th>
      <td>52.884795</td>
      <td>51.820584</td>
      <td>43.128684</td>
      <td>6.567243</td>
      <td>0.492236</td>
    </tr>
    <tr>
      <th>Ketapril</th>
      <td>55.235638</td>
      <td>53.698743</td>
      <td>68.553577</td>
      <td>8.279709</td>
      <td>0.603860</td>
    </tr>
    <tr>
      <th>Naftisol</th>
      <td>54.331565</td>
      <td>52.509285</td>
      <td>66.173479</td>
      <td>8.134708</td>
      <td>0.596466</td>
    </tr>
    <tr>
      <th>Placebo</th>
      <td>54.033581</td>
      <td>52.288934</td>
      <td>61.168083</td>
      <td>7.821003</td>
      <td>0.581331</td>
    </tr>
    <tr>
      <th>Propriva</th>
      <td>52.322552</td>
      <td>50.854632</td>
      <td>42.351070</td>
      <td>6.507770</td>
      <td>0.512884</td>
    </tr>
    <tr>
      <th>Ramicane</th>
      <td>40.216745</td>
      <td>40.673236</td>
      <td>23.486704</td>
      <td>4.846308</td>
      <td>0.320955</td>
    </tr>
    <tr>
      <th>Stelasyn</th>
      <td>54.233149</td>
      <td>52.431737</td>
      <td>59.450562</td>
      <td>7.710419</td>
      <td>0.573111</td>
    </tr>
    <tr>
      <th>Zoniferol</th>
      <td>53.236507</td>
      <td>51.818479</td>
      <td>48.533355</td>
      <td>6.966589</td>
      <td>0.516398</td>
    </tr>
  </tbody>
</table>
</div>



## Bar and Pie Charts


```python
# Generate a bar plot showing the total number of unique mice tested on each drug regimen using pandas.

# Generate a bar plot showing number of data points for each treatment regimen using pandas
reg = merged.groupby(["Drug Regimen"]).count()["Mouse ID"]
reg.plot(kind="bar", figsize = (10,5))

plt.title("Data Points")
plt.xlabel("Drug Regimen")
plt.ylabel("Data Points")
```




    Text(0,0.5,'Data Points')




![png](output_11_1.png)



```python
# Generate a bar plot showing the total number of unique mice tested on each drug regimen using pyplot.
u = [230, 178, 178, 188, 186, 181, 161, 228, 181, 182]

#Set the x_axis to be the amount of the Data Regimen
x_axis = np.arange(len(reg))

plt.bar(x_axis, u, color='b', alpha=0.75, align='center')

tick_locations = [value for value in x_axis]
plt.xticks(tick_locations, ['Capomulin', 'Ceftamin', 'Infubinol', 'Ketapril', 'Naftisol', 'Placebo', 'Propriva', 'Ramicane', 'Stelasyn', 'Zoniferol'],  rotation='vertical')

plt.xlim(-0.75, len(x_axis)-0.25)

plt.ylim(0, max(u)+10)

plt.title("Data Points Visual")
plt.xlabel("Drug Regimen")
plt.ylabel("Data Points")

```




    Text(0,0.5,'Data Points')




![png](output_12_1.png)



```python
Observation:
    
    I noticed that the data was not even, especially because Capomulin and Ramicane had the most data points.
```


```python
gen = merged.groupby(["Mouse ID","Sex"])
gen

mouse_gen = pd.DataFrame(gen.size())

#Create the dataframe with total count of Female and Male mice
mouse_gen = pd.DataFrame(mouse_gen.groupby(["Sex"]).count())
mouse_gen.columns = ["Total Count"]

#create and format the percentage of female vs male
mouse_gen["Percentage of Sex"] = (100*(mouse_gen["Total Count"]/mouse_gen["Total Count"].sum()))

#format the "Percentage of Sex" column
mouse_gen["Percentage of Sex"] = mouse_gen["Percentage of Sex"]

#gender_df
mouse_gen

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total Count</th>
      <th>Percentage of Sex</th>
    </tr>
    <tr>
      <th>Sex</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Female</th>
      <td>124</td>
      <td>49.799197</td>
    </tr>
    <tr>
      <th>Male</th>
      <td>125</td>
      <td>50.200803</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Generate a pie plot showing the distribution of female versus male mice using pandas
colors = ['green', 'orange']
explode = (0.1, 0)
plot = mouse_gen.plot.pie(y='Total Count',figsize=(5,5), colors = colors, startangle=140, explode = explode, shadow = True, autopct="%1.1f%%")
```


![png](output_15_0.png)



```python
# Generate a pie plot showing the distribution of female versus male mice using pyplot

# Create Labels for the sections of the pie
labels = ["Female","Male"]

#List the values of each section of the pie chart
sizes = [49.799197,50.200803]

#Set colors for each section of the pie
colors = ['yellow', 'purple']

#Determine which section of the circle to detach
explode = (0.1, 0)

#Create the pie chart based upon the values 
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct="%1.1f%%", shadow=True, startangle=140)

#Set equal axis
plt.axis("equal")

```




    (-1.1879383453817904,
     1.111754351424799,
     -1.1987553745848882,
     1.1126035084692154)




![png](output_16_1.png)



```python
Observation:
    
    There are slightly more males than females in this data set.
```

## Quartiles, Outliers and Boxplots


```python
# Calculate the final tumor volume of each mouse across four of the treatment regimens:  
# Capomulin, Ramicane, Infubinol, and Ceftamin
regimes = merged[merged["Drug Regimen"].isin(["Capomulin", "Ramicane", "Infubinol", "Ceftamin"])]
regimes = regimes.sort_values(["Timepoint"], ascending=True)

# Start by getting the last (greatest) timepoint for each mouse


# Merge this group df with the original dataframe to get the tumor volume at the last timepoint
sort_regimes = regimes.groupby(['Drug Regimen', 'Mouse ID']).last()['Tumor Volume (mm3)']
sort_regimes

```




    Drug Regimen  Mouse ID
    Capomulin     b128        38.982878
                  b742        38.939633
                  f966        30.485985
                  g288        37.074024
                  g316        40.159220
                  i557        47.685963
                  i738        37.311846
                  j119        38.125164
                  j246        38.753265
                  l509        41.483008
                  l897        38.846876
                  m601        28.430964
                  m957        33.329098
                  r157        46.539206
                  r554        32.377357
                  r944        41.581521
                  s185        23.343598
                  s710        40.728578
                  t565        34.455298
                  u364        31.023923
                  v923        40.658124
                  w150        39.952347
                  w914        36.041047
                  x401        28.484033
                  y793        31.896238
    Ceftamin      a275        62.999356
                  b447        45.000000
                  b487        56.057749
                  b759        55.742829
                  f436        48.722078
                                ...    
    Infubinol     v766        51.542431
                  w193        50.005138
                  w584        58.268442
                  y163        67.685569
                  z581        62.754451
    Ramicane      a411        38.407618
                  a444        43.047543
                  a520        38.810366
                  a644        32.978522
                  c458        38.342008
                  c758        33.397653
                  d251        37.311236
                  e662        40.659006
                  g791        29.128472
                  i177        33.562402
                  i334        36.374510
                  j913        31.560470
                  j989        36.134852
                  k403        22.050126
                  m546        30.564625
                  n364        31.095335
                  q597        45.220869
                  q610        36.561652
                  r811        37.225650
                  r921        43.419381
                  s508        30.276232
                  u196        40.667713
                  w678        43.166373
                  y449        44.183451
                  z578        30.638696
    Name: Tumor Volume (mm3), Length: 100, dtype: float64




```python
# Put treatments into a list for for loop (and later for plot labels)
top_4 = ['Capomulin', 'Ramicane', 'Infubinol','Ceftamin']


# Create empty list to fill with tumor vol data (for plotting)
tvoldata = []


# Calculate the IQR and quantitatively determine if there are any potential outliers. 
quartiles = sort_regimes.quantile([.25,.5,.75])
lowerq = quartiles[0.25]
upperq = quartiles[0.75]
iqr = upperq-lowerq
    
    # Locate the rows which contain mice on each drug and get the tumor volumes
    
    
    # add subset 
    
    
    # Determine outliers using upper and lower bounds
    
```


```python
# Generate a box plot of the final tumor volume of each mouse across four regimens of interest
df = best_regimen_df.reset_index()
tumor_lists = final_df.groupby('Drug Regimen')['Tumor Volume (mm3)'].apply(list)
tumor_list_df = pd.DataFrame(tumor_lists)
tumor_list_df = tumor_list_df.reindex(top_4)
tumor_vols = [vol for vol in tumor_list_df['Tumor Volume (mm3)']]
plt.boxplot(tumor_vols, labels=top_4)
plt.ylim(10, 80)
plt.show()

```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-20-55223ea2f1f2> in <module>()
          1 # Generate a box plot of the final tumor volume of each mouse across four regimens of interest
    ----> 2 df = best_regimen_df.reset_index()
          3 tumor_lists = final_df.groupby('Drug Regimen')['Tumor Volume (mm3)'].apply(list)
          4 tumor_list_df = pd.DataFrame(tumor_lists)
          5 tumor_list_df = tumor_list_df.reindex(top_4)


    NameError: name 'best_regimen_df' is not defined


## Line and Scatter Plots


```python
# Generate a line plot of tumor volume vs. time point for a mouse treated with Capomulin
tumor_v_t = merged[merged["Mouse ID"].isin(["j119"])]
tumor_v_t

time_v_t_data = tumor_v_t[["Mouse ID", "Timepoint", "Tumor Volume (mm3)"]]
time_v_t_data

line_plot_df = time_v_t_data.reset_index()
line_plot_df

line_plot_final = line_plot_df[["Mouse ID", "Timepoint", "Tumor Volume (mm3)"]]
line_plot_final

lines = line_plot_final.plot.line()
lines
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11a284550>




![png](output_23_1.png)



```python
Observation:
    
    There seems to be a low correlation between the two. 
    As the timepoint steadily increases, it appears that the Tumor Volume fluctuates. 
```


```python
# Generate a scatter plot of average tumor volume vs. mouse weight for the Capomulin regimen
x_axis = np.arange(0, 10, 0.1)
times = []
for x in x_axis:
    times.append(x * x + np.random.randint(0, np.ceil(max(x_axis))))
```


```python
plt.title("Average Tumor Volume vs. Mouse weight ")
plt.xlabel("Average Tumor Volume")
plt.ylabel("Mouse Weight")

plt.scatter(x_axis, times, marker="o", color="red")
plt.show()
```


![png](output_26_0.png)


## Correlation and Regression


```python
# Calculate the correlation coefficient and linear regression model 
# for mouse weight and average tumor volume for the Capomulin regimen

```
