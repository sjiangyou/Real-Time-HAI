import h5py
import os
import numpy as np
import pandas as pd
import pathlib
from SHIPS_Data_Collection import ships_extraction
<<<<<<< Updated upstream

os.chdir("Data/H5")
=======
>>>>>>> Stashed changes


def main():
<<<<<<< Updated upstream
    for file in list(pathlib.Path(".").glob("*.RT-H5")):
        h5_convert_csv(file)


=======
    for file in list(pathlib.Path('.').glob('*.RT-H5')):
        h5_convert_csv(file)
        

>>>>>>> Stashed changes
def h5_convert_csv(filename):
    date, time = filename[23:31], filename[33:35]
    year = date[0:4]
    basin, ident, category, latitude, longitude = ships_extraction(date + time)
    latitude, longitude = float(latitude), float(longitude)
    filename, basin, ident, category, date, time = (
        str(filename),
        str(basin),
        str(ident),
        str(category),
        str(date),
        str(time),
    )

    f = h5py.File(filename, "r")
    grid = f["Grid"]

    longitude_values = np.repeat(list(grid["lon"]), 1800)
    latitude_values = list(grid["lat"]) * 3600
    precipitation_values = np.array(list(grid["precipitationCal"])).flatten()

    dataset = pd.DataFrame(
        {
            "lon": longitude_values,
            "lat": latitude_values,
            "precipitation": precipitation_values,
        }
    )
    dataset.columns = [
        grid["lon"].attrs["standard_name"].decode()
        + " ("
        + grid["lon"].attrs["units"].decode()
        + ")",
        grid["lat"].attrs["standard_name"].decode()
        + " ("
        + grid["lat"].attrs["units"].decode()
        + ")",
        "Precipitation (" + grid["precipitationCal"].attrs["units"].decode() + ")",
    ]

    dataset = dataset[dataset["longitude (degrees_east)"] <= (longitude + 6.0)]
    dataset = dataset[dataset["longitude (degrees_east)"] >= (longitude - 6.1)]
    dataset = dataset[dataset["latitude (degrees_north)"] <= (latitude + 6.1)]
    dataset = dataset[dataset["latitude (degrees_north)"] >= (latitude - 6.0)]

    image = dataset.pivot(
        index="latitude (degrees_north)",
        columns="longitude (degrees_east)",
        values="Precipitation (mm/hr)",
    )
    image = image.T
    os.chdir("../../IMERG_CSV")
    image.to_csv(
        f"{basin}_{year}{ident}_{category}_{date}{time}.csv", index=False, header=False
    )
    os.chdir("../Data/H5")


if __name__ == "__main__":
    main()
