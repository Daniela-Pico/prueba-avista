"""
run.py
──────
Punto de entrada CLI del pipeline MLOps.

Uso:
    python run.py                          # usa config por defecto
    python run.py --config path/config.yaml
"""
import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))



# Garantiza que los módulos del proyecto sean encontrados
sys.path.insert(0, os.path.dirname(__file__))

from pipeline import run_pipeline

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(__file__), "config", "config.yaml"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline MLOps — Pronóstico de transacciones bancarias"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"Ruta al archivo de configuración YAML (default: {DEFAULT_CONFIG})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    results = run_pipeline(args.config)

    # Imprimir resumen final en consola
    print("\n" + "=" * 60)
    print("RESUMEN FINAL — Métricas globales por modelo")
    print("=" * 60)
    print(results["summary"].to_string())

    print("\nModelos ganadores:")
    print(results["df_winners"][["idTerminal", "oper", "mejor_modelo", "MAE"]]
          .to_string(index=False))