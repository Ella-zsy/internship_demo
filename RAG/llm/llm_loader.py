from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from config import LLM_MODEL_PATH


def load_llm():

    llm = LlamaCpp(
        model_path=LLM_MODEL_PATH,
        n_gpu_layers=100,
        n_batch=512,
        n_ctx=2048,
        f16_kv=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=True
    )

    return llm