from .base import Attack
from .none import NoOpAttack
from .sample_hi_adam import SampleHiAdamAttack
from .gcg import GCGAttack
from .pair import PAIRAttack

ATTACKS: dict[str, Attack] = {
    "none": NoOpAttack(),
    "sample_hi_adam": SampleHiAdamAttack(),
    "gcg": GCGAttack(),
    "pair": PAIRAttack(),
}