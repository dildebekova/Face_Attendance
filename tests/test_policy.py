import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"backend"))
import pytest
from policy import decide

def test_known(): assert decide([(1,.80),(2,.40)],.55,.08)==1
@pytest.mark.parametrize('scores',[[],[(1,.4)],[(1,.7),(2,.69)],[(1,float('nan'))]])
def test_reject(scores):
    with pytest.raises(ValueError): decide(scores,.55,.08)
