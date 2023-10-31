## Intro to Data Science and AI

# Assignment 1

---

## Application description

The application creates a scatter plot that compares GDP per capita with Life expectancy.

---

## Running the application

### Prerequisites:

* Installed Anaconda distribution of Python (https://www.anaconda.com/download)

### Starting the app

Run the following command **in the directory of app.py**:

> python app.py

---

## Using the application

The application is capable of creating a scatter plot with the data for each country and a year provided by the user
or for all years for a single country provided by the user.

### Workflow for displaying data for each country and a year provided by the user

The app prompts the user with message: ***Show data by country or by year (c/y):*** the user must type ***y*** for
data for each country for the year to be displayed. After that the user is prompted with: ***Do you want country names
to be displayed on the chart (y/n):***, typing ***y*** will add the name of the country to each dot on the plot
and typing ***n*** will not do it. After that the prompt is: ***Enter desired year (2002-2021):*** the user enters
a year for which data to be displayed. Countries with life expectancy higher than one standard deviation above the mean
are logged in the console and browser opens and the scatter plot is displayed.

### Workflow for displaying data for a country provided by the user for each year

The app prompts the user with message: ***Show data by country or by year (c/y):*** the user must type ***c*** for data
for a country provided by the user for each year  to be displayed. After that the user is prompted with:  ***Enter
desired country (starting with capital letter):***, the user enters the name of a country, a browser opens and
the scatter plot is displayed.

---