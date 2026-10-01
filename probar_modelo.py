import joblib
import xgboost as xgb

paquete = joblib.load("models/modelo_xgboost_semanal.pkl")

preprocesador = paquete["preprocesador"]

modelo = xgb.Booster()
modelo.load_model(bytearray(paquete["booster_raw"]))

print("Preprocesador cargado:", type(preprocesador))
print("Modelo XGBoost cargado correctamente.")