from typing import List, Optional
from dataclasses import dataclass
from abc import abstractmethod, ABC


@dataclass
class Template(ABC):
    @abstractmethod
    def __post_init__(self) -> None:
        """
        call the self._register_template to define the template like:
        self._register_template(
            prefix="the prefix if the dataset's prefix is not defined",
            prompt="the model input with {query}",
            sep="\n",
            use_history=False
        )
        :return: NoneType
        """
        pass

    def get_prompt(self, query: str, history: Optional[list] = None, prefix: Optional[str] = "") -> str:
        r"""
        Returns a string containing prompt without response.
        """
        return "".join(self._format_example(query, history, prefix))

    def get_dialog(self, query: str, resp: str, history: Optional[list] = None, prefix: Optional[str] = "") -> List[str]:
        r"""
        Returns a list containing 2 * n elements where the 2k-th is a query and the (2k+1)-th is a response.
        """
        return self._format_example(query, history, prefix) + [resp]

    def _register_template(self, prefix: str, prompt: str, sep: str, use_history: Optional[bool] = True) -> None:
        self.prefix = prefix
        self.prompt = prompt
        self.sep = sep
        self.use_history = use_history

    def _format_example(self, query: str, history: Optional[list] = None, prefix: Optional[str] = "") -> List[str]:
        prefix = prefix if prefix else self.prefix  # use prefix if provided
        prefix = prefix + self.sep if prefix else ""  # add separator for non-empty prefix
        history = history if (history and self.use_history) else []
        history = history + [(query, "<dummy>")]
        convs = []
        for turn_idx, (user_query, bot_resp) in enumerate(history):
            if turn_idx == 0:
                convs.append(prefix + self.prompt.format(query=user_query))
                convs.append(bot_resp)
            else:
                convs.append(self.sep + self.prompt.format(query=user_query))
                convs.append(bot_resp)
        return convs[:-1]  # drop last


class defaultTemplate(Template):
    def __post_init__(self):
        """
        Default template.
        """
        self._register_template(
            prefix="A chat between a curious user and an artificial intelligence assistant. "
                   "The assistant gives helpful, detailed, and polite answers to the user's questions.",
            prompt="Human: {query}\nAssistant: The answer to the question is ",
            sep="\n",
            use_history=False
        )


class compTemplate(Template):
    def __post_init__(self):
        """
        Default template.
        """
        self._register_template(
            prefix="A chat between a curious user and an artificial intelligence assistant. "
                   "The assistant gives helpful, detailed, and polite answers to the user's questions.",
            prompt="Human: {query}\nAssistant: ",
            sep="\n",
            use_history=False
        )


class TableQATemplate(Template):
    def __post_init__(self):
        self._register_template(
            prefix="A chat between a curious user and an artificial intelligence assistant. "
                   "The assistant must retrieves the table and gives a correct answer to the user's questions.",
            prompt="user: {query}\nassistant: According to the table, the answer to the question is ",
            sep="\n",
            use_history=False
        )


class TableTextQATemplate(Template):
    def __post_init__(self):
        self._register_template(
            prefix="A chat between a curious user and an artificial intelligence assistant. "
                   "The assistant must retrieves the table and text, then gives a correct answer to the user's questions.",
            prompt="user: {query}\nassistant: According to the table and text, the answer to the question is ",
            sep="\n",
            use_history=False
        )


class QATemplate(Template):
    def __post_init__(self):
        self._register_template(
            prefix="A chat between a curious user and an artificial intelligence assistant. "
                   "The assistant must retrieves and gives a correct answer to the user's questions.",
            prompt="user: {query}\nassistant: According to the given information, the answer to the question is ",
            sep="\n",
            use_history=False
        )


class CoLATemplate(Template):
    def __post_init__(self):
        self._register_template(
            prefix="A chat between a curious user and an artificial intelligence assistant. "
                   "The assistant must gives a correct answer to the user's instructions.",
            prompt="user: {query}\nassistant: Output 1 or 0 based on sentence grammar correctness, the answer to the question is ",
            sep="\n",
            use_history=False
        )


class RTETemplate(Template):
    def __post_init__(self):
        self._register_template(
            prefix="A chat between a curious user and an artificial intelligence assistant. "
                   "The assistant must gives a correct answer to the user's instructions.",
            prompt="user: {query}\nassistant: Output entailment or not_entailment based on the given sentences entailment, the answer to the question is ",
            sep="\n",
            use_history=False
        )


class SST2Template(Template):
    def __post_init__(self):
        self._register_template(
            prefix="A chat between a curious user and an artificial intelligence assistant. "
                   "The assistant must gives a correct answer to the user's instructions.",
            prompt="user: {query}\nassistant: Output 1 or 0 based on the emotions in the given movie review, the answer to the question is ",
            sep="\n",
            use_history=False
        )


class QNLITemplate(Template):
    def __post_init__(self):
        self._register_template(
            prefix="A chat between a curious user and an artificial intelligence assistant. "
                   "The assistant must gives a correct answer to the user's instructions.",
            prompt="user: {query}\nassistant: Output entailment or not_entailment based on the entailment between question and sentence, the answer to the question is ",
            sep="\n",
            use_history=False
        )


class MRPCTemplate(Template):
    def __post_init__(self):
        self._register_template(
            prefix="A chat between a curious user and an artificial intelligence assistant. "
                   "The assistant must gives a correct answer to the user's instructions.",
            prompt="user: {query}\nassistant: Output 1 or 0 based on the semantic similarity between sentence1 and sentence2, the answer to the question is ",
            sep="\n",
            use_history=False
        )


class STSBTemplate(Template):
    def __post_init__(self):
        self._register_template(
            prefix="A chat between a curious user and an artificial intelligence assistant. "
                   "The assistant must gives a correct answer to the user's instructions.",
            prompt="user: {query}\nassistant: Based on the semantic similarity between sentence1 and sentence2, the score is ",
            sep="\n",
            use_history=False
        )


STR_TEMPLATE_MAP = {
    "multi_hiertt": TableTextQATemplate,
    "skg": TableQATemplate,
    "compaqt": TableTextQATemplate,
    "squad": QATemplate,
    "hotpotqa": QATemplate,
    "adversarialqa": QATemplate,
    "sst2": SST2Template,
    "qnli": QNLITemplate,
    "mrpc": MRPCTemplate,
    "stsb": STSBTemplate,
    "cola": CoLATemplate,
    "rte": RTETemplate,
    "wikitq": TableQATemplate,
    "hybridqa": TableTextQATemplate,
    "addsub": defaultTemplate,
    "aqua": defaultTemplate,
    "arcc": defaultTemplate,
    "arce": defaultTemplate,
    "boolq": defaultTemplate,
    "gsm8k": defaultTemplate,
    "obqa": defaultTemplate,
    "piqa": defaultTemplate,
    "multiarith": defaultTemplate,
    "singleeq": defaultTemplate,
    "svamp": defaultTemplate,
    "comp": compTemplate,
    "math": compTemplate,
    "common": defaultTemplate,
}
