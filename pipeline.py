"""Builds the LangChain pipeline that turns one prompt into one model response.

LangChain's "LCEL" (LangChain Expression Language) composes steps with the `|`
operator, like piping shell commands. Every step is a `Runnable` — an object
with `.invoke(x)` that takes one value and returns the next one. `a | b`
gives you a new Runnable that feeds `a`'s output straight into `b`.

Our chain, left to right:

    attack -> [input-stage defenses] -> prompt template -> model -> text parser -> [output-stage defenses]

- `attack.apply` and `defense.apply` are plain Python functions (see
  attacks/base.py, defenses/base.py). Wrapping them in `RunnableLambda` is
  what lets them sit in the `|` chain alongside LangChain's own components.
- `ChatPromptTemplate` turns `{"text": "..."}` into the list of chat messages
  the model expects.
- `ChatOllama` is the actual LLM call.
- `StrOutputParser` unwraps the model's reply object down to a plain string.

Each defense declares a `stage` ("input" or "output"), so adding or removing
a defense never requires touching this file — it just gets spliced in at the
matching point.
"""

from collections.abc import Sequence

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from attacks.base import Attack
from defenses.base import Defense


def _make_model(model_name: str, dry_run: bool) -> Runnable:
    """Return the real Ollama model, or a stub that just echoes the prompt.

    --dry-run lets the rest of the chain be exercised without an Ollama
    server running.
    """
    if dry_run:
        return RunnableLambda(
            lambda prompt_value: f"[dry-run model] prompt was:\n{prompt_value.to_string()}"
        )

    from langchain_ollama import ChatOllama  # imported lazily: not needed for --dry-run

    return ChatOllama(model=model_name)


def _defense_step(defense: Defense) -> Runnable:
    """Wrap a Defense as a Runnable step in the chain.

    `defense=defense` captures the current loop value immediately. Without
    it, every step's lambda would share one `defense` variable and all of
    them would run the *last* defense in the list once the loop finished.
    """
    return RunnableLambda(lambda text, defense=defense: defense.apply(text))


def build_chain(
    attack: Attack,
    defenses: Sequence[Defense],
    model_name: str,
    dry_run: bool = False,
) -> Runnable:
    """Wire one attack, any number of defenses, and a model into one Runnable.

    Call `.invoke(prompt_text)` on the result to run a prompt end to end.
    """
    input_defenses = [d for d in defenses if d.stage == "input"]
    output_defenses = [d for d in defenses if d.stage == "output"]

    prompt_template = ChatPromptTemplate.from_messages([("user", "{text}")])
    model = _make_model(model_name, dry_run)

    chain = RunnableLambda(attack.apply)
    for defense in input_defenses:
        chain = chain | _defense_step(defense)

    # ChatPromptTemplate needs a dict of template variables, so re-wrap the
    # plain string as {"text": ...} before it reaches the template.
    chain = chain | RunnableLambda(lambda text: {"text": text}) | prompt_template | model | StrOutputParser()

    for defense in output_defenses:
        chain = chain | _defense_step(defense)

    return chain
