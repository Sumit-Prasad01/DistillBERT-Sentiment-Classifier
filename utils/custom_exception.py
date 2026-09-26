import sys


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Extracts detailed error message including file name and line number.
    """
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        return f"Error occurred in script: [{file_name}] at line number: [{line_number}] with message: [{str(error)}]"
    return str(error)


class CustomException(Exception):
    """
    Custom exception class that provides detailed traceback information.
    """
    def __init__(self, error_message: Exception, error_detail: sys = sys):
        super().__init__(str(error_message))
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        return self.error_message
