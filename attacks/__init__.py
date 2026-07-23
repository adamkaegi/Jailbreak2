from .base import Attack
from .deepinception import DeepInceptionAttack
from .gcg import GCGAttack
from .none import NoOpAttack
from .pair import PAIRAttack
from .sample_hi_adam import SampleHiAdamAttack

# Available attacks:
# - deepinception: Ryan's nested-scene template attack.
# - template: compatibility alias for deepinception.
# - gcg: static universal adversarial suffix.
# - pair: iterative black-box prompt refinement.
# - sample_hi_adam: original harness wiring example.
# - none: raw-prompt baseline.
_deepinception = DeepInceptionAttack()

ATTACKS: dict[str, Attack] = {
    attack.name: attack
    for attack in (
        _deepinception,
        GCGAttack(),
        PAIRAttack(),
        SampleHiAdamAttack(),
        NoOpAttack(),
    )
}

ATTACKS["template"] = _deepinception
