import sys
import os
sys.path.append(os.getcwd())

try:
    print("Importing Descriptores...")
    import Descriptores
    print("OK Descriptores")
except Exception as e:
    print(f"FAIL Descriptores: {e}")

try:
    print("Importing nuevas_areas...")
    from Descriptores import nuevas_areas
    print("OK nuevas_areas")
except Exception as e:
    print(f"FAIL nuevas_areas: {e}")

try:
    print("Importing nuevos_geometricos...")
    from Descriptores import nuevos_geometricos
    print("OK nuevos_geometricos")
except Exception as e:
    print(f"FAIL nuevos_geometricos: {e}")

try:
    print("Importing nuevos_factores...")
    from Descriptores import nuevos_factores
    print("OK nuevos_factores")
except Exception as e:
    print(f"FAIL nuevos_factores: {e}")
