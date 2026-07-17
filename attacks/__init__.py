from .base import Attack
from .deepinception import DeepInceptionAttack
from .sample_hi_adam import SampleHiAdamAttack
from .none import NoOpAttack

# Available attacks:
# - deepinception: Ryan's nested-scene template attack.
# - template: compatibility alias for deepinception.
# - sample_hi_adam: original harness wiring example.
# - none: raw-prompt baseline.
_deepinception = DeepInceptionAttack()

ATTACKS: dict[str, Attack] = {
    attack.name: attack
    for attack in (_deepinception, SampleHiAdamAttack(), NoOpAttack())
}

ATTACKS["template"] = _deepinception
