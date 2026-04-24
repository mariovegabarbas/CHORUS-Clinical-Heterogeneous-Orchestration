import json
from pathlib import Path

DEFAULT_JSON="modelos.json"

def cargar_modelos(opcion, json_path=DEFAULT_JSON):
    path=Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"El archivo {json_path} no existe.")
    with open(path,"r",encoding="UTF-8") as file:
        data=json.load(file)
    
    resultado_modelos=[]
    for opt in opcion.split(","):
        opt=opt.strip()
        if opt=="1":
            resultado_modelos.extend(data["LLM"].get("FREE_MODELS",[]))
            print(f"Resultados: {resultado_modelos}")
        elif opt=="2":
            resultado_modelos.extend(data["LLM"].get("PAY_MODELS",[]))
        else:
            print("Opción no reconocida.")

    return resultado_modelos