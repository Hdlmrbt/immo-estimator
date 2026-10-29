from pydantic import BaseModel, Field


class HouseInput(BaseModel):
    MedInc: float = Field(gt=0, le=20, description="Revenu médian (en dizaines de milliers de $)")
    HouseAge: float = Field(ge=1, le=60, description="Âge médian des maisons (années)")
    AveRooms: float = Field(gt=0, le=50, description="Nombre moyen de pièces")
    AveBedrms: float = Field(gt=0, le=20, description="Nombre moyen de chambres")
    Population: float = Field(gt=0, le=40000, description="Population du quartier")
    AveOccup: float = Field(gt=0, le=50, description="Nombre moyen d'occupants")
    Latitude: float = Field(ge=32, le=42)
    Longitude: float = Field(ge=-125, le=-114)


class PredictionOut(BaseModel):
    predicted_price: float