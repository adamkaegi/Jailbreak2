from .base import Attack
from .sample_hi_adam import SampleHiAdamAttack
from .none import NoOpAttack

ATTACKS: dict[str, Attack] = {a.name: a for a in (SampleHiAdamAttack(), NoOpAttack())}
