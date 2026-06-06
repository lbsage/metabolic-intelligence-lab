import numpy as np
from metabolic_intelligence_lab.core.memory import GeometricMemory

def test_memory_retrieve():
    gm = GeometricMemory(dim=2)
    gm.store(np.array([1.0, 0.0]), "x")
    assert gm.retrieve(np.array([1.0, 0.0]), threshold=0.5)[0][1] == "x"
