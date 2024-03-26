import pydantic
import re

class BaseTool(pydantic.BaseModel):

    name = "BaseTool"
    args = {"reason": "the reason you think you are calling this tool"}

    def __init__(self, *args, **kwargs):
        pass

    def validate_call(self, call: str) -> bool:
        """
        Validates the call string and extracts arguments.
        Arguments are expected to be in the format 'arg: value' and separated by newlines.
        """
        call = call.strip()
        if not 'BASETOOL' in call:
            return {'result': 'ToolHandler was looking for test BASETOOL but not found. Reformat your message and try again.'}
        
        extracted_args = self.extract_args(call)
        for arg in extracted_args.keys():
            if arg not in self.args:
                return {'result': f'Missing argument {arg}. Please check your arguments and try again.'}
        
        return extracted_args

    def extract_args(self, call: str) -> dict:
        """
        Extracts arguments from the call string.
        Arguments are expected to be in the format 'arg: value' and separated by newlines.
        """
        extracted_args = {}
        for arg in self.args.keys():
            search_pattern = f"{arg}: (.*)\n"
            match = re.search(search_pattern, call)
            if match:
                extracted_args[arg] = match.group(1)
        return extracted_args

    def execute(self, args):
        return {'result': 'You called execute on basetool. Basetool only acknowledges the call.'}