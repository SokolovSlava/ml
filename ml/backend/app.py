from enum import Enum

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

@app.get("/healthcheck")
def healthcheck():
    return {"status": "OK"}

@app.get("/predict")
def predict(iris: IrisModel):
    score = sum([iris.petal_height, iris.petal_width, iris.sepal_height, iris.sepal_width])
    if score > 10:
        iris_type = "foo"
    elif score > 5:
        iris_type = IrisType.VERSICOLOUR
    else:
        iris_type = IrisType.VIRGINICA
    return {"predict": iris_type}