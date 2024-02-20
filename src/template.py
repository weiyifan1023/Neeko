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


STR_TEMPLATE_MAP = {
    "character-llm": defaultTemplate,
    "character-llm-single": defaultTemplate,
}
