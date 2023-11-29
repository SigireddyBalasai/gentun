# Make public APIs available at top-level import
from gentun.algorithms import GeneticAlgorithm, RussianRouletteGA
from gentun.populations import Population, GridPopulation
from gentun.server import DistributedPopulation, DistributedGridPopulation
from gentun.client import GentunClient

# xgboost individuals and models
try:
    from gentun.individuals import XgboostIndividual
    from gentun.models.xgboost_models import XgboostModel
except ImportError:
    print("Warning: install xgboost to use XgboostIndividual and XgboostModel.")

# Keras individuals and models
try:
    from gentun.individuals import GeneticCnnIndividual
    from gentun.models.keras_models import GeneticCnnModel
except ImportError:
    print("Warning: install Keras and TensorFlow to use GeneticCnnIndividual and GeneticCnnModel.")
