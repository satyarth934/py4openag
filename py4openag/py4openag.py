import random
import codecs
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.linear_model import LinearRegression

from ipyleaflet import (
    Map,
    basemaps,
    Marker,
    FullScreenControl,
    AwesomeIcon,
)

from typing import List, Tuple


class functions:
    def __init__(self, data: pd.DataFrame) -> None:
        """Performs some checks on the input dataself.

        Args:
                data (pd.DataFrame): Input data
        """
        print("Using cleaned and documented code...")

        validation = self.data_validation(data)
        if validation[0]:
            columns = self.has_columns(data)
            if columns[0]:
                self.data = data
                print("Successfully imported the data!\n")
            else:
                print(columns[1])  # Printing the error message from method.
        else:
            print(validation[1])  # Printing the error message from method.

    def has_columns(self, data: pd.DataFrame) -> Tuple[bool, object]:
        """Ensures that the input data contains all the necessary columns.

        Args:
                data (pd.DataFrame): Input data

        Returns:
                Tuple[bool, object]: Returns True if the necessary columns exist, False otherwise along with an error message.
        """

        if set(
            [
                "mean_2m_air_temperature",
                "minimum_2m_air_temperature",
                "maximum_2m_air_temperature",
                "total_precipitation",
                "Date",
            ]
        ).issubset(data.columns):
            return (True, None)
        else:
            return (False, "Data is missing columns")

    def data_validation(self, data: pd.DataFrame) -> Tuple[bool, object]:
        """Validate the format of the Date column of the input data.
        The format must be of the type 'datetime64[ns]'.

        Args:
                data (pd.DataFrame): Input data

        Returns:
                Tuple[bool, object]: Returns True if the format of the input data is correct. False otherwise along with an error message.
        """

        if data["Date"].dtype == "datetime64[ns]":
            return (True, None)
        else:
            return (False, "Date has to be in datetime format")

    def extreme_degree_days(
        self,
        data: pd.DataFrame,
        thresholdtemp: float,
        year: int,
        months: List[int, int] = [1, 12],
    ) -> float:
        """Calculates mean temperature for days with a temperature rise.

        Args:
                data (pd.DataFrame): Input data
                thresholdtemp (float): Threshold temperature
                year (int): Year to query from the input data.
                months (List[int, int], optional): Specifies the range of months using the start and end month. Defaults to [1, 12].

        Returns:
                float: Returns the mean temperature for days with a temperature rise.
        """

        sum = self.integral_time(data, thresholdtemp, "above", year, months)
        new_df = pd.DataFrame()
        tempdata = data[data["Date"].dt.year == year]
        for i in range(months[0], months[1] + 1):
            new_df = new_df.append(
                tempdata[tempdata["Date"].dt.month == i], ignore_index=True
            )
        index = new_df.index
        extreme_degree_days = sum / len(index)
        return extreme_degree_days

    def integral_time(
        self,
        data: pd.DataFrame,
        threshold: float,
        area: str,
        year: int,
        months: List[int, int] = [1, 12],
    ) -> float:
        """Calculates the sum of rise or fall in temperature for the specified months.

        Args:
                data (pd.DataFrame): Input data
                threshold (float): Threshold to calculate the temperature rise or fall.
                area (str): To specify rise: "above" or fall: "below"
                year (int): Year to query from the input data.
                months (List[int, int], optional): Specifies the range of months using the start and end month. Defaults to [1, 12].

        Returns:
                float: Returns the sum of rise or fall in temperature for the specified months.
        """

        sum = 0
        new_df = pd.DataFrame()
        tempdata = data[data["Date"].dt.year == year]
        for i in range(months[0], months[1] + 1):
            new_df = new_df.append(
                tempdata[tempdata["Date"].dt.month == i], ignore_index=True
            )
        if area == "above":
            for _, j in new_df.iterrows():
                sum += max((j["mean_2m_air_temperature"] - threshold), 0)
        if area == "below":
            for _, j in new_df.iterrows():
                sum += max((threshold - j["mean_2m_air_temperature"]), 0)
        return sum

    def growing_degree_days(
        self, data: pd.DataFrame, year: int, basetemp: float
    ) -> float:
        """Calculates the mean per day rise in temperature.

        Args:
                data (pd.DataFrame): Input data
                year (int): Year to query from the input data.
                basetemp (float): Base temperature to estimate the temperature rise.

        Returns:
                float: Returns the mean per day rise in temperature.
        """

        sum = 0
        k = 0
        new_df = data[data["Date"].dt.year == year]

        for _, j in new_df.iterrows():
            temp = (
                (j["minimum_2m_air_temperature"] + j["maximum_2m_air_temperature"]) / 2
            ) - basetemp

            if temp > 0:
                sum += temp
            k += 1

        gdd = sum / k
        return gdd

    def growingdays_basetemp(self, crop: str) -> object:
        """Returns the base temperature for some specified cropsself.

        Args:
                crop (str): Crop name

        Returns:
                object: Returns base temperature (float) for some specified crops and None for others.
        """

        if crop in [
            "wheat",
            "barley",
            "rye",
            "oats",
            "flaxseed",
            "lettuce",
            "asparagus",
        ]:
            return 4.5
        elif crop in ["sunflower", "potato"]:
            return 8
        elif crop in ["maize", "sorghum", "rice", "soybeans", "tomato", "coffee"]:
            return 10
        else:
            print(
                "The crop is not present. Look up base temperature for: [wheat,barley,rye,oats,flaxseed,lettuce,asparagus,sunflower,potato,maize,sorghum,rice,soybeans,tomato,coffee] instead"
            )
            return None

    def average_temperature(
        self, data: pd.DataFrame, year: int, months: List[int, int] = [1, 12]
    ) -> float:
        """Calculates the mean of the air temperature over all the months mentioned for the specified yearself.

        Args:
                data (pd.DataFrame): Input data
                year (int): Year to query from the input data.
                months (List[int, int], optional): Specifies the range of months using the start and end month. Defaults to [1, 12].

        Returns:
                float: Returns the mean of the air temperature over all the months for the specified year.
        """

        new_df = pd.DataFrame()
        tempdata = data[data["Date"].dt.year == year]
        for i in range(months[0], months[1] + 1):
            new_df = new_df.append(
                tempdata[tempdata["Date"].dt.month == i], ignore_index=True
            )
        avg = new_df["mean_2m_air_temperature"].mean()
        return avg

    def total_precipitation(
        self, data: pd.DataFrame, year: int, months: List[int, int] = [1, 12]
    ) -> float:
        """Calculates the total precipitation over all the months mentioned for a specified year.

        Args:
                data (pd.DataFrame): Input data
                year (int): Year to query from the input data
                months (List[int, int], optional): Specifies the range of months using the start and end month. Defaults to [1, 12].

        Returns:
                float: Returns the sum of total_precipitation over all the months for the specified year.
        """

        new_df = pd.DataFrame()
        tempdata = data[data["Date"].dt.year == year]
        for i in range(months[0], months[1] + 1):
            new_df = new_df.append(
                tempdata[tempdata["Date"].dt.month == i], ignore_index=True
            )
        sum = new_df["total_precipitation"].sum()
        return sum

    def temptrend(
        self, data: pd.DataFrame, years: List[int, int]
    ) -> Tuple[float, float, float]:
        """Calculates the average temperature trend over years using linear regression and pearson correlation.

        Args:
                data (pd.DataFrame): Input data
                years (List[int, int]): Range of years defined by starting year and ending year.

        Returns:
                Tuple[float, float, float]: Returns the captured trend using three values: pearson correlation coefficient, two-tailed p-value, linear regression coefficient.
        """

        pvalT = []
        yearavg = []
        for year in range(years[0], years[1] + 1):
            new_df = data[data["Date"].dt.year == year]
            avg = new_df["mean_2m_air_temperature"].mean()
            yearavg.append(avg)
        x = np.array(yearavg)
        t = np.array([i for i in range(len(yearavg))])
        reg = LinearRegression().fit(t.reshape(-1, 1), x)
        p = stats.pearsonr(t, x)
        pvalT = p
        r = p[0]
        return r, pvalT[1], reg.coef_[0]

    def preciptrend(
        self, data: pd.DataFrame, years: List[int, int]
    ) -> Tuple[float, float, float]:
        """Calculates the average precipitation trend over years using linear regression and pearson correlation.

        Args:
                data (pd.DataFrame): Input data
                years (List[int, int]): Range of years defined by starting year and ending year.

        Returns:
                Tuple[float, float, float]: Returns the captured trend using three values: pearson correlation coefficient, two-tailed p-value, linear regression coefficient.
        """

        pvalP = []
        yearavg = []
        for year in range(years[0], years[1] + 1):
            new_df = data[data["Date"].dt.year == year]
            avg = new_df["total_precipitation"].mean()
            yearavg.append(avg)
        x = np.array(yearavg)
        t = np.array([i for i in range(len(yearavg))])
        reg = LinearRegression().fit(t.reshape(-1, 1), x)
        p = stats.pearsonr(t, x)
        pvalP = p
        r = p[0]
        return r, pvalP[1], reg.coef_[0]

    def plotmap(
        self,
        metric: str,
        climatedf: pd.DataFrame,
        coorddf: pd.DataFrame,
        filepath: str,
        filename: str = "Map",
    ) -> Tuple[ipyleaflet.Map, plt.figure]:
        """Plots the coordinates on the world map.

        Args:
                metric (str): Metric column to be used.
                climatedf (pd.DataFrame): Climate data frame.
                coorddf (pd.DataFrame): Coordinates data frame.
                filepath (str): Path for the output file.
                filename (str, optional): Name of the output file. Defaults to "Map".

        Returns:
                Tuple[ipyleaflet.Map, plt.figure]: Returns the map and the figure object.
        """

        sel_cols = ["Location", "Year", metric]
        climatedf = climatedf[sel_cols]
        climatedf = climatedf.reindex(columns=climatedf.columns.tolist() + ["color"])
        color = []
        for (i, j) in climatedf.iterrows():
            value = (j[metric] - climatedf[metric].min()) / (
                climatedf[metric].max() - climatedf[metric].min()
            )
            if value > 0 and value <= (1 / 6):
                color.append("darkblue")
            elif value > (1 / 6) and value <= (2 / 6):
                color.append("blue")
            elif value > (2 / 6) and value <= (3 / 6):
                color.append("green")
            elif value > (3 / 6) and value <= (4 / 6):
                color.append("orange")
            elif value > (4 / 6) and value <= (5 / 6):
                color.append("red")
            else:
                color.append("darkred")
        climatedf["color"] = color
        gps_color = pd.merge(climatedf, coorddf, on=["Location"])
        gps_color.head()
        newdata = pd.DataFrame([])
        for (index, row) in gps_color.iterrows():
            row["Latitude"] += random.uniform(0.1, 0.9)
            row["Longitude"] += random.uniform(0.1, 0.9)
            newdata = newdata.append(row)
        center = [39.0119, -98.4842]
        zoom = 3
        i = 0
        m = Map(basemap=basemaps.Esri.WorldImagery, center=center, zoom=zoom)
        for (index, row) in newdata.iterrows():
            icon = AwesomeIcon(
                name="tint",
                marker_color=row.loc["color"],  #'#583470'
                icon_color="black",
                spin=False,
            )
            loc = [row.loc["Latitude"], row.loc["Longitude"]]
            marker = Marker(location=loc, draggable=False, icon=icon)
            m.add_layer(marker)
            i += 1
        m.add_control(FullScreenControl())
        m.save(filepath + "/" + filename + ".html", title=filename)
        mpl.rcParams.update({"font.size": 10})
        fig = plt.figure(figsize=(8, 3))
        ax = fig.add_subplot(111)
        vals = []
        for i in range(7):
            vals.append(
                (((climatedf[metric].max() - climatedf[metric].min()) / 6) * i)
                + climatedf[metric].min()
            )
        cmap = mpl.colors.ListedColormap(
            ["darkblue", "deepskyblue", "limegreen", "orange", "red", "darkred"]
        )
        norm = mpl.colors.BoundaryNorm(vals, cmap.N)
        cb = mpl.colorbar.ColorbarBase(
            ax,
            cmap=cmap,
            norm=norm,
            spacing="uniform",
            orientation="horizontal",
            extend="neither",
            ticks=vals,
        )
        cb.set_label(metric)
        ax.set_position((0.1, 0.45, 0.8, 0.1))
        plt.savefig(filepath + "/" + "legend.jpg", dpi=2000, bbox_inches="tight")
        file = codecs.open(filepath + "/" + filename + ".html", "r", "utf-8")
        file_list = file.read().split("\n")
        file.close()
        print(file_list)
        file_list.insert(
            -3, "<img src='legend.jpg' alt='Plot Legend' style='width:35%;'>"
        )
        file = codecs.open(filepath + "/" + filename + ".html", "w", "utf-8")
        file.write("\n".join(file_list))
        file.close()
        return m, fig

    def heavy_precipitation_days(
        self, data: pd.DataFrame, years: List[int, int]
    ) -> int:
        """Counts the number of heavy precipitation days from the range of the given years.

        Args:
                data (pd.DataFrame): Input data
                years (List[int, int]): Range of years defined by the starting year and ending year.

        Returns:
                int: Returns the count of heavy precipitation days.
        """

        new_df = pd.DataFrame()
        for year in range(years[0], years[1] + 1):
            new_df = new_df.append(
                data[data["Date"].dt.year == year], ignore_index=True
            )
        x = (
            0.99
            * (
                new_df["mean_2m_air_temperature"].max()
                - new_df["mean_2m_air_temperature"].min()
            )
        ) + new_df["mean_2m_air_temperature"].min()
        heavyprecipdays = new_df[new_df["mean_2m_air_temperature"] > x][
            "mean_2m_air_temperature"
        ].count()
        return heavyprecipdays
