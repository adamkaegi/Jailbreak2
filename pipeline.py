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
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)

from attacks.base import Attack
from defenses.base import Defense, DefenseResult


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
    """Wrap a Defense as a block-aware Runnable step in the chain.

    `defense=defense` captures the current loop value immediately. Without
    it, every step's lambda would share one `defense` variable and all of
    them would run the *last* defense in the list once the loop finished.
    """

    def apply_defense(state: dict, defense: Defense = defense) -> dict:
        if state["blocked"]:
            return state

        raw_result = defense.apply_with_context(
            state["text"],
            original_prompt=state["original_prompt"],
            model_prompt=state["model_prompt"],
        )
        result = (
            raw_result
            if isinstance(raw_result, DefenseResult)
            else DefenseResult(raw_result)
        )
        if not isinstance(result.text, str):
            raise TypeError(f"Defense '{defense.name}' returned non-string text")

        updated = dict(state)
        updated["text"] = result.text
        updated["blocked"] = result.blocked
        if result.blocked:
            updated["blocked_by"] = defense.name
        if result.metadata:
            updated["defense_metadata"] = {
                **state["defense_metadata"],
                defense.name: result.metadata,
            }
        return updated

    return RunnableLambda(apply_defense)


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

    chain = RunnableLambda(
        lambda prompt: {
            "original_prompt": prompt,
            "text": attack.apply(prompt),
            "model_prompt": None,
            "blocked": False,
            "blocked_by": None,
            "defense_metadata": {},
        }
    )
    for defense in input_defenses:
        chain = chain | _defense_step(defense)

    model_step = RunnablePassthrough.assign(
        model_prompt=RunnableLambda(lambda state: state["text"]),
        text=(
            RunnableLambda(lambda state: {"text": state["text"]})
            | prompt_template
            | model
            | StrOutputParser()
        ),
    )
    chain = chain | RunnableBranch(
        (lambda state: state["blocked"], RunnableLambda(lambda state: state)),
        model_step,
    )

    for defense in output_defenses:
        chain = chain | _defense_step(defense)

    return chain | RunnableLambda(lambda state: state["text"])
