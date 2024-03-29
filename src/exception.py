import sys 
from src.log_config import logging 

# Define a function to extract error message 
def error_message_detail(error, error_details_object: sys):
    # Get the traceback information from error detail 
    _, _, exc_tb = error_details_object.exc.info()

    # Extract the filename from traceback 
    file_name = exc_tb.tb_frame.f_code.co_file_name 

    # Create a formatted message that contains filename, linenumber and error message 
    formatted_error_message = f"Error in Python File: [{file_name}] line number: [{exc_tb.tb_lineno}] error message: [{error}]"

    return formatted_error_message


# Define the custom exception class 
class FileOperationError(Exception):
    # Create a constructor to initiate the exception with error message detail 
    def __init__(self, formatted_error_message, error_details_object: sys):
        # Call the class constructor to set the error message 
        super().__init__(formatted_error_message)
        # Call the error message detail function to get the detailed message 
        self.formatted_error_message = error_message_detail(formatted_error_message, error_details_object = error_details_object)

    # Override the __str__ method to return the formatted error message
    def __str__(self):
        return self.formatted_error_message

# # Raising an exception
# if __name__=="__main__":
#     try:
#         a = 1/0
#     except Exception as e:
#         logging.info("Division by Zero error") 
#         raise FileOperationError(e, sys)