import sys
from networksecurity.logging import logger
import traceback

class NetworkSecurityException(Exception):
    def __init__(self, error_message, error_details: sys = None):
        """
        error_message: the original exception (or message)
        error_details: usually the `sys` module so we can call sys.exc_info()
        This constructor is defensive: it handles cases where sys.exc_info()
        returns (None, None, None) and when the caller passed a string.
        """
        self.error_message = error_message

        # Default values
        self.lineno = None
        self.file_name = None

        try:
            exc_tb = None

            # Preferred: get traceback from provided error_details (usually sys)
            if error_details is not None:
                try:
                    _, _, exc_tb = error_details.exc_info()
                except Exception:
                    exc_tb = None

            # If no traceback from sys.exc_info(), try to get it from the exception object
            if exc_tb is None and isinstance(error_message, BaseException):
                exc_tb = getattr(error_message, "__traceback__", None)

            # If we have a traceback object, extract filename and line number
            if exc_tb is not None:
                self.lineno = getattr(exc_tb, "tb_lineno", None)
                frame = getattr(exc_tb, "tb_frame", None)
                if frame is not None:
                    self.file_name = getattr(frame.f_code, "co_filename", None)

        except Exception:
            # Be defensive: never raise from the exception class itself
            self.lineno = None
            self.file_name = None

    def __str__(self):
        fname = self.file_name if self.file_name is not None else "Unknown"
        lineno = self.lineno if self.lineno is not None else "Unknown"
        return "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
            fname, lineno, str(self.error_message)
        )
