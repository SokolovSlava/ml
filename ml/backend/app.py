from enum import Enum

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import psycopg2
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()

class IrisType(str, Enum):
    SETOSA = "setosa"
    VERSICOLOUR = "versicolour"
    VIRGINICA = "virginica"


class IrisModel(BaseModel):
    petal_height: float = Field(ge=0)
    petal_width: float = Field(ge=0)
    sepal_height: float = Field(ge=0)
    sepal_width: float = Field(ge=0)

def fetch_data() -> pd.DataFrame:
    query = """
    SELECT iris_id, petal_height, petal_width, sepal_height, sepal_width, target
    FROM Iris
    """

    with psycopg2.connect(
        dbname="iris",
        user='admin',
        password='admin',
        host='db'
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            return pd.DataFrame(
                rows,
                columns=[
                    "iris_id",
                    "petal_height",
                    "petal_width",
                    "sepal_height",
                    "sepal_width",
                    "target",
                ],
            ).set_index("iris_id")

def get_predict(data: pd.DataFrame, iris: IrisModel) -> int:
    model = KNeighborsClassifier()
    model.fit(X=data[["petal_height", "petal_width", "sepal_height", "sepal_width"]], y = data["target"])
    return model.predict(pd.DataFrame(data=iris.model_dump(), index=[0]).values)[0]


@app.get("/healthcheck")
def healthcheck():
    return {"status": "OK"}

@app.get("/predict")
def predict(iris: IrisModel):
    data = fetch_data()
    predict = get_predict(data=data, iris=iris)

    iris_type = {
        0: IrisType.SETOSA,
        1: IrisType.VERSICOLOUR,
        2: IrisType.VIRGINICA,
    }[predict]
    return  {"predict": iris_type}