"""Verifica se as dependências principais do projeto estão instaladas (Issue #1).

Uso:
    python scripts/check_environment.py

Sai com código 0 se todas as dependências puderem ser importadas e com
código 1 caso contrário. Funciona em Linux, macOS e Windows.
"""

import importlib
import importlib.metadata

# nome do pacote (requirements.txt) -> módulo importável
DEPENDENCIES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "jupyter": "jupyter",
    "pytest": "pytest",
    "nltk": "nltk",
}


def main() -> int:
    failures = []
    for package, module in DEPENDENCIES.items():
        try:
            importlib.import_module(module)
        except ImportError as error:
            failures.append(package)
            print(f"[FALHA] {package}: {error}")
        else:
            version = importlib.metadata.version(package)
            print(f"[OK] {package} {version}")

    if failures:
        print(f"\n{len(failures)} dependência(s) ausente(s): {', '.join(failures)}")
        return 1

    print("\nTodas as dependências principais foram importadas com sucesso.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
