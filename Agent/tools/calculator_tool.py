from langchain.tools import Tool


def calculator(expression):

    try:
        result = eval(expression)
        return str(result)

    except Exception:
        return "calculation error"


calculator_tool = Tool(
    name="Calculator",
    func=calculator,
    description="Useful for math calculations"
)