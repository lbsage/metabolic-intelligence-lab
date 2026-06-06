from metabolic_intelligence_lab.core.salience import SalienceEngine

def test_salience_top():
    s = SalienceEngine()
    s.observe("cold")
    assert s.top(1)[0][0] == "cold"
