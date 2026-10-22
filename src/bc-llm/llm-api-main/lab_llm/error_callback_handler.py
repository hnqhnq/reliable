import logging
from typing import Any, Union
from langchain.callbacks.base import BaseCallbackHandler

class ErrorCallbackHandler(BaseCallbackHandler):
    def __init__(self, logger: logging.Logger):
        super().__init__()
        self.logger = logger

    def on_llm_error(self, 
                     error: Union[Exception, KeyboardInterrupt], 
                     **kwargs: Any
                     ) -> None:
        self.logger.error(f"LLM Error: {str(error)}")
        self.logger.error(f"Error type: {type(error).__name__}")
        self.logger.error(f"Additional context: {kwargs}")

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Persist token metadata when the provider/LangChain exposes it."""
        llm_output = getattr(response, "llm_output", None)
        if llm_output:
            self.logger.info("LLM usage metadata: %s", llm_output)
