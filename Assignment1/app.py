import pandas as pd
import plotly.express as px
import utils as ut

LIFE_EXPECTANCY_TITLE="Life expectancy (period) at birth - Sex: all - Age: 0"
GDP_TITLE="GDP per capita, PPP (constant 2017 international $)"
SCATTER_PLOT_NAME="GDP per capita vs Life Expectancy"

gdp_data = pd.read_csv("gdp-per-capita-worldbank.csv")
gdp_data_data_frame = pd.DataFrame(gdp_data)
life_expectancy = pd.read_csv("life-expectancy.csv")
life_expectancy_data_frame = pd.DataFrame(life_expectancy)

displayByYear = ut.get_year_or_country()

if(displayByYear):
    displayNames = ut.get_display_name()
    year = int(input("Enter desired year (2002-2021): "))
else:
    entity = input("Enter desired country (starting with capital letter): ")

if(displayByYear):
    aggregated_data = pd.merge(gdp_data_data_frame[gdp_data_data_frame["Year"] == year], life_expectancy_data_frame[life_expectancy_data_frame["Year"] == year], on="Entity")
else:
    aggregated_data = pd.merge(gdp_data_data_frame[gdp_data_data_frame["Entity"] == entity], life_expectancy_data_frame[life_expectancy_data_frame["Entity"] == entity], on="Year")

# print(aggregated_data)

if (displayByYear and displayNames):
    fig = px.scatter(aggregated_data, x=GDP_TITLE, y=LIFE_EXPECTANCY_TITLE, text='Entity', title=SCATTER_PLOT_NAME, labels={'X': 'X-axis Label', 'Y': 'Y-axis Label'})
elif(displayByYear and not displayNames):
    fig = px.scatter(aggregated_data, x=GDP_TITLE, y=LIFE_EXPECTANCY_TITLE, title=SCATTER_PLOT_NAME, labels={'X': 'X-axis Label', 'Y': 'Y-axis Label'})
elif(not displayByYear):
    fig = px.scatter(aggregated_data, x=GDP_TITLE, y=LIFE_EXPECTANCY_TITLE, text='Year', title=SCATTER_PLOT_NAME, labels={'X': 'X-axis Label', 'Y': 'Y-axis Label'})

fig.update_traces(textposition='top right')

fig.show()

if (displayByYear):
    # Calculate the mean and standard deviation of life expectancy
    mean_life_expectancy = aggregated_data[LIFE_EXPECTANCY_TITLE].mean()
    std_deviation_life_expectancy = aggregated_data[LIFE_EXPECTANCY_TITLE].std()

    # Calculate the threshold for higher life expectancy (mean + 1 * standard deviation)
    threshold = mean_life_expectancy + std_deviation_life_expectancy

    # Filter the data for countries with life expectancy higher than the threshold
    high_life_expectancy_countries = aggregated_data[aggregated_data[LIFE_EXPECTANCY_TITLE] > threshold]

    # Display the countries with high life expectancy
    print("Countries with life expectancy higher than one standard deviation above the mean:")
    print(high_life_expectancy_countries)