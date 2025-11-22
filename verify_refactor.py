import sys
import os

# Add src to sys.path to simulate installation if not installed
sys.path.insert(0, os.path.abspath("src"))

try:
    import legacy.chronos
    print("Successfully imported legacy.chronos")
except ImportError as e:
    print(f"Failed to import legacy.chronos: {e}")

try:
    import chronos2
    print("Successfully imported chronos2")
except ImportError as e:
    print(f"Failed to import chronos2: {e}")

try:
    from legacy.chronos import BaseChronosPipeline
    print("Successfully imported BaseChronosPipeline from legacy.chronos")
except ImportError as e:
    print(f"Failed to import BaseChronosPipeline from legacy.chronos: {e}")

try:
    from chronos2 import Chronos2Pipeline
    print("Successfully imported Chronos2Pipeline from chronos2")
except ImportError as e:
    print(f"Failed to import Chronos2Pipeline from chronos2: {e}")

try:
    from legacy.chronos import Chronos2Pipeline
    print("WARNING: Chronos2Pipeline still available in legacy.chronos (should be removed)")
except ImportError:
    print("Correctly failed to import Chronos2Pipeline from legacy.chronos")

