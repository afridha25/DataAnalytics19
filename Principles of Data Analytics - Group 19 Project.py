#!/usr/bin/env python
# coding: utf-8

# ## BUSINESS STATEMENT
# 
# 
# Our analysis will be on Rental car companies and how Enterprise needs to switch towards more fuel-efficient cars in terms of CO2 emission. 
# This research will mostly be on which fuel type produces the least CO2.
# 
# We believe fuel type to be the most important variable in reducing CO2 emissions, having recently been displayed as significant by the UK’s switch from E5 to E10 petrol.
# Which ‘cut transport CO2 emissions by 750,000 tonnes per year, which is the equivalent of taking 350,000 cars off of UK roads.’ 
# 
# (https://www.gov.uk/government/news/fuelling-a-greener-future-e10-petrol-available-at-pumps-from-today) proving the validity of our argument.
# 
# This increased eco-friendly policy is not only for helping the planet, but also ensures that Enterprise remains relevant in the current market. 
# Customers have lexicographical preferences when it comes to car rental, forcing these companies to compete on all fronts against each other for the customers. 
# If Enterprise gets left behind on the Eco-friendly issue, this will result in them being seen as a dirty company, 
# with customers preferring their competitors.
# Increased eco-friendly cars in their fleet also have large tax implications 
# 
# (https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/709655/ultra-low-emission-vehicles-tax-benefits.pdf). 
# 
# In short, CO2 emissions is an important topic Enterprise needs to analyse.
# 
# 

# In[1]:


#Importing Libraries
import math
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as sts
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# In[2]:


#Reading the dataset
CO_NO_Emission = pd.read_csv('Data_CO.csv')


# In[3]:


#Description about the data
CO_NO_Emission.describe()


# In[4]:


CO_NO_Emission.info()


# In[5]:


#Cleaning the data
#Locking the columns which are required for the problem
Emission_data=CO_NO_Emission.loc[:,['Manufacturer','Model','Description','Transmission','Manual or Automatic','Engine Capacity',
                    'Fuel Type','Powertrain','Emissions CO [mg/km]','Emissions NOx [mg/km]','Date of change','THC Emissions [mg/km]','THC + NOx Emissions [mg/km]','WLTP CO2','Noise Level dB(A)']]
Emission_data.head()



# In[6]:


Emission_data.tail()


# In[7]:


Emission_data.describe()


# In[8]:


Emission_data.isnull().values.any()


# In[9]:


Emission_data.isnull().sum()


# In[10]:


def handling_outlier(variable):
  # See observation outliers on the variables
  Q1 = Emission_data[variable].quantile(0.25)
  Q3 = Emission_data[variable].quantile(0.75)
  IQR = Q3 - Q1
  lower_range = Q1 - (1.5 * IQR)
  upper_range = Q3 + (1.5 * IQR)
  
  # Replace outlier observations with upper bound and lower bound
  Emission_data.loc[(Emission_data[variable]>upper_range), variable] = upper_range
  Emission_data.loc[(Emission_data[variable]<lower_range), variable] = lower_range
  return Emission_data


# In[11]:


high_correlated_variables = Emission_data.corr()['WLTP CO2'].sort_values(ascending=False)[0:5].index.values.tolist() 

# Remove outlier for each high correlated variables
for variable in high_correlated_variables:
  handling_outlier(variable)


# In[12]:


Emission_data.hist(figsize=(12,8), color='green', bins=30, edgecolor='black')
plt.tight_layout()
plt.show()


# We can notice that the histogram of the Emission CO, Emission NOx and THC Emission follows exponentital distribution and this distribution can be used to approximate the data for future measurements.
# WLTP CO2 Emission follows Normal Distribution which indicates that most values falls between 100 and 200mg/km. Most values are closer to the average mean of WLTP Co2
# 

# In[13]:


Emission_data.plot(kind = 'box', subplots = True, layout = (4, 3), figsize = (15, 15));


# In[14]:


Brand_data=Emission_data['Manufacturer'].value_counts().reset_index().rename(columns={'index':'Manufacturer','Manufacturer':'Count'})[0:25]
Brand_data
fig = go.Figure(go.Bar(
    x=Brand_data['Manufacturer'],y=Brand_data['Count'],
    marker={'color': Brand_data['Count'], 
    'colorscale': 'sunset'},  
    text=Brand_data['Count'],
    textposition = "outside",
))
fig.update_layout(title_text='Top 25 Company',xaxis_title="Company ",yaxis_title="Number Of Vehicles ",title_x=0.5)
fig.show()


# In[15]:


df_FuelType=Emission_data['Fuel Type'].value_counts().reset_index().rename(columns={'index':'Fuel Type','Fuel Type':'Count'})
df_FuelType
fig = px.pie(df_FuelType, values='Count', names='Fuel Type')

fig.update_traces(hoverinfo='label+percent', textinfo='value+percent', textfont_size=12,insidetextorientation='radial')
fig.update_layout(title="Fuel Type",title_x=0.5)
fig.show()


# In[16]:


df_Transmission=Emission_data['Transmission'].value_counts().reset_index().rename(columns={'index':'Transmission','Transmission':'Count'})

fig = go.Figure(go.Bar(
    x=df_Transmission['Transmission'],y=df_Transmission['Count'],
    marker={'color': df_Transmission['Count'], 
    'colorscale': 'burg'},  
    text=df_Transmission['Count'],
    textposition = "outside",
))
fig.update_layout(title_text='Transmission Distribution ',xaxis_title="Transmission ",yaxis_title="Number Of Vehicles ",title_x=0.5)


# In[17]:


print("Correlation Matrix")
plt.rcParams['figure.figsize']=(8,6)
sns.heatmap(Emission_data.corr(),cmap='coolwarm',linewidths=.5,fmt=".2f",annot = True);


# ## HYPOTHESIS TESTING 
# 
# 
#     
# Our test statistic does not fall in the critical region -2.1788 < test statistic(0.3667) < 2.1788. Therefore, H0 is retained.
# there is no statistically significant difference between the CO2 emissions of Diesel and Petrol. 
# 
# 
# The CO2 emissions due to electricity production required for Electric cars is around ....... Our board of directors believes that "Electric cars require much more energy to produce than traditional cars, especially on account of the batteries." Our hypothesis is that electric cars emits the same or more amount of CO2 as of traditional cars. We perform a hypothesis test to compare the CO2 emission of both electric and traditional cars. 
# H0: sample mean of electric car CO2 emission = sample mean of traditional car CO2 emission
# H1: sample mean of electric car CO2 emission != sample mean of traditional car CO2 emission
# 
# Electric cars must therefore be driven longer before they can be considered zero-emission cars."
# 
# 
# Average CO2 Emission for Electric Cars(For Uk) -  g/km
# 
# 
# 
# 

# https://ev-database.uk/cheatsheet/energy-consumption-electric-car
# 
# https://www.eia.gov/tools/faqs/faq.php?id=74&t=11
# 
# 
# 

# In[22]:


import scipy.stats as sts
import matplotlib.pyplot as plt
import numpy as np
import math
x = np.linspace(-8, 8, 100) #the range of t distribution should start from -infinity to +infinity so it starts from -8 to +8
plt.plot(x, sts.t.pdf(x, 4655), 'r-', alpha=0.6)
plt.title('PDF of t-Distribution')
plt.ylabel('f(x)')
plt.xlabel('x')


C_value=sts.t.ppf(0.05, 4655, loc=0, scale=1)
print(C_value)
x2= np.linspace(-8,C_value, 100)
plt.fill_between(x2, sts.t.pdf(x2, 4655), color='purple',alpha=0.6) #n = 19 - degree of freedom 
plt.annotate('Critical Value is -1.729',(C_value,0),(C_value-2,0.25),arrowprops=dict(arrowstyle="fancy",
                            fc="0.3", ec="none",
                            connectionstyle="angle3,angleA=0,angleB=-90")) #to highlight the text value
test_statistic_value=((149.067454-2)/3730.62)*math.sqrt(4657)
print('The test statistic is', test_statistic_value)
plt.annotate('Test Statistic is -1.601',(test_statistic_value,0),(test_statistic_value+4,0.15),arrowprops=dict(arrowstyle="wedge,tail_width=0.5", alpha=0.3))
print('The test statistic is not in the critical region, null hypothesis is retained ')


# In[ ]:




