import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor



def Modelos(datos1, leMake, leModel, leState, Modelo):
    Year = datos1["Year"]
    Mileage = datos1["Mileage"]
    State = datos1["State"]
    Make = datos1["Make"]
    Model = datos1["Model"]
    ID = 0

    columnas = ['Year', 'Mileage', 'State', 'Make', 'Model', 'ID']
    datos = [[Year, Mileage, State, Make, Model, ID]]

    domain_01 = pd.DataFrame(datos, columns=columnas)
    db_02 = domain_01.set_index('ID')

    db_02["State"] = leState.transform(db_02.State)
    db_02["Make"] = leMake.transform(db_02.Make)
    db_02["Model"] = leModel.transform(db_02.Model)

    # Make prediction
    ypred = Modelo.predict(db_02)
    ypred = str(ypred[0])

    return ypred
