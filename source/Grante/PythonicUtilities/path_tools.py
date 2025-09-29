# Routine to store tools to work with path

import os

from pathlib import Path

import inspect

########################################################################
#                              Path tools                              #
########################################################################

# Defines a function to verify if a path exists or not. If not, create
# it

def verify_path(parent_path, file_name):

    if parent_path is None:

        return file_name

    # Checks if the parent path exists

    if not os.path.exists(parent_path):

        # Initializes a list of directories to create

        directories = [""]

        # Iterates through the path

        for i in range(len(parent_path)):

            # If the character is a bar, this means the last directory
            # name has been finished

            if parent_path[-(i+1)]=="/":

                # Verifies if the last directory is not empty

                if directories[-1]!="":

                    directories.append("")

            # Otherwise, saves the characters

            else:

                directories[-1] = parent_path[-(i+1)]+directories[-1]

        # Checks if the last saved directory is empty

        if directories[-1]=="":

            directories = directories[0:-1]

        # Initializes the path

        path = ""

        # Iterates through the directories

        for i in range(len(directories)):

            # Appends this bit of directory to the path

            path += "//"+directories[-(i+1)]

            # Checks if this path exists

            if not os.path.exists(path):

                # Creates the directory

                os.mkdir(path)

    # Joins everything together

    return parent_path+"//"+file_name

# Defines a function to take out the termination file name of a string

def take_outFileNameTermination(file_name, get_termination=False):

    # Initializes the new name

    clean_fileName = ""

    # Initializes the termination

    termination = ""

    termination_reading = False

    # Iterates through the file name

    for character in file_name:

        if character=="." or character=='.':

            if not get_termination:

                break 

            else:

                termination_reading = True

        else:

            if termination_reading:

                termination += character

            else:

                clean_fileName += character

    # Returns the file name without the termination

    if get_termination:

        return clean_fileName, termination
    
    else:

        return clean_fileName
    
# Defines a function to get the path to a file

def get_parent_path_of_file(file="__file__", path_bits_to_be_excluded=1,
function_calls_to_retrocede=1):

    if file=="__file__":

        # Tries to automatically retrieve where this function has been
        # called

        # Gets the previous frame, where the function has been called

        last_frame = inspect.stack()[function_calls_to_retrocede]

        # Gets the module where it's been called

        module = inspect.getmodule(last_frame[0])

        # Verifies if this module is valid and gets the __file__ attri-
        # bute

        if (module is not None) and hasattr(module, "__file__"):

            file = module.__file__ 

        else:

            raise NameError("To get the parent path of a file, you sho"+
            "uld type file=file, where file is an imported file; or yo"+
            "u should type file=__file__ to get the parent path to the"+
            " current file. Otherwise, it will try to get the path of "+
            "the file where this function, 'get_parent_path_of_file', "+
            "has been called last. Unfortunately, to no avail this tim"+
            "e")
    
    # Gets a list of the bits of the whole path
    
    current_path = Path(os.path.abspath(file)).parts

    # Joins them again without the last bit, which is the name of the 
    # file itself

    return str(Path(*current_path[0:-path_bits_to_be_excluded]))