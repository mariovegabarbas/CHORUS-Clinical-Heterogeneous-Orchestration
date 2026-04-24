from cargador_modelos import cargar_modelos
from Ensambladores.ensamblador_LLM import Ensamblador
from analizador import dataAnalisis
import asyncio

def main_menu():
    print("Bienvenido al Consensuador de Respuestas de Modelos de Lenguaje (CRML)\n¿Quiere usar los modelos gratuitos (1) o los de pago (2)?")
    opcion=input("Teclee 1 o 2: ").strip()
    modelos=cargar_modelos(opcion)
    print(f"Categoría seleccionada: {opcion} -> Cargando {len(modelos)} modelos...\n")
    
    prompt=input("\nEscriba el prompt a evaluar:")
    resultados=asyncio.run(run_ensamblador(modelos,prompt))
    reporte=dataAnalisis(resultados)

async def run_ensamblador(modelos, prompt):
    ensamblador=Ensamblador(modelos=modelos)
    print("\nCargando...")
    resultados= await ensamblador.run(prompt)
    ensamblador.guardar_resultados(resultados)
    print("\nListo")
    return resultados

if __name__ == "__main__":
    main_menu()