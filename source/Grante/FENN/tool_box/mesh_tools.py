# Routine to store methods to read meshes generated in GMSH 

from ...PythonicUtilities.path_tools import verify_path, verify_file_existence, get_parent_path_of_file, take_outFileNameTermination

from ...PythonicUtilities.string_tools import string_toList

########################################################################
#                             Mesh reading                             #
########################################################################

# Defines a function to read a mesh .msh

def read_msh_mesh(file_name, parent_directory=None, verbose=False):

    # If the parent directory is None, get the parent path of the file 
    # where this function has been called

    if parent_directory is None:

        parent_directory = get_parent_path_of_file(
        function_calls_to_retrocede=2)

    # Takes out the file termination, and adds the correct .msh

    file_name = take_outFileNameTermination(file_name)+".msh"

    # Verifies the path

    file_name = verify_path(parent_directory, file_name)

    # Reads the msh file into a list of strings. Each element is a line

    lines_list = []

    try:

        with open(file_name, "r") as infile:

            for line in infile:

                lines_list.append(line.strip())

    except:

        raise FileNotFoundError("The file "+file_name+" was not found "+
        "while evaluating trying to read a msh mesh.")

    # Reads the physical groups. Gets the dictionaries of physical groups
    # of names to tags

    (domain_physicalGroupsNameToTag, 
    boundary_physicalGroupsNameToTag, start_reading_at_index
    ) = read_physical_groups(lines_list, 0)

    if verbose:

        print("#######################################################"+
        "#################\n#                        Domain physical g"+
        "roups                        #\n#############################"+
        "###########################################\n")

        for name, tag in domain_physicalGroupsNameToTag.items():

            print("Domain physical group name: "+str(name)+"; tag: "+str(
            tag))

        print("\n#####################################################"+
        "###################\n#                       Boundary physica"+
        "l groups                       #\n###########################"+
        "#############################################\n")

        for name, tag in boundary_physicalGroupsNameToTag.items():

            print("Boundary physical group name: "+str(name)+"; tag: "+
            str(tag))

    # Reads the node coordinates

    node_coordinates, start_reading_at_index = read_nodes(lines_list,
    start_reading_at_index)

    print(node_coordinates)
    
# Defines a function to read the bit about physical groups. The output is
# a dictionary of domain physical groups names to tags and another to 
# boundary information

def read_physical_groups(lines_list, start_reading_at_index, start_key=
"$PhysicalNames", end_key="$EndPhysicalNames"):

    # Initializes a dictionary where the keys corresponds to the topolo-
    # gycal dimension of the physical group and the values are dictiona-
    # ry themselves. The dictionaries as values store key-value pairs of
    # physical groups names to the respective physical groups tags

    physical_groups_dicts = {}

    # Initializes a flag to tell if reading is allowed already

    flag_reading = False

    # Iterates through the lines

    for i in range(start_reading_at_index, len(lines_list)):

        # Gets the line

        line = lines_list[i]

        # Verifies if reading is not allowed yet

        if not flag_reading:

            # Verifies if this line is equal to the start key

            if line==start_key:

                # Updates the flag reading to allow for reading

                flag_reading = True

            continue

        # Verifies if the end key has been reached

        elif line==end_key:

            # Modifies the start index for the next index

            start_reading_at_index = i+1

            break 

        # Iterates through the line looking for '"", that tell a physi-
        # cal group name

        line_numerical_info = ""

        physical_group_name = ""

        for j in range(len(line)):

            if line[j]=='"':

                physical_group_name = line[(j+1):(len(line)-1)]

                line_numerical_info = line[0:(j-1)]

                break 

        # Transforms this line to a list to get the different informa-
        # tion readily available, and independently

        line_info = string_toList("["+line_numerical_info+"]", 
        element_separator=" ")

        # If there are two elements in the list, the first is the physi-
        # cal group topological dimension, whereas the second one is the
        # physical group tag

        if len(line_info)==2:

            # Verifies if this topological dimension has already been
            # registered

            if line_info[0] in physical_groups_dicts:

                # Updates the dictionary of physical groups

                physical_groups_dicts[line_info[0]][physical_group_name
                ] = line_info[1]

            # Otherwise, creates the inner dictionary

            else:

                physical_groups_dicts[line_info[0]] = {
                physical_group_name: line_info[1]}

    # Verifies the keys of the greater dictionary

    keys = list(physical_groups_dicts.keys())

    keys.sort()

    # If there are not two keys throws an error

    if len(keys)!=2:

        raise ValueError("The number of categories of physical groups "+
        "using the topological dimension as criterion is "+str(len(keys)
        )+". There should be 2: one for the domain and another for the"+
        " boundary")
    
    # Separates the domain from the surface physical groups using the 
    # topological dimension

    boundary_physicalGroupsNameToTag = physical_groups_dicts[keys[0]]

    domain_physicalGroupsNameToTag = physical_groups_dicts[keys[1]]

    # Returns the dictionaries and the new index to start reading

    return (domain_physicalGroupsNameToTag, 
    boundary_physicalGroupsNameToTag, start_reading_at_index)
    
# Defines a function to read the bit about nodes. The output is a list of
# lists, where each list corresponds to a node with that index and the 
# components are the coordinates

def read_nodes(lines_list, start_reading_at_index, start_key=
"$Nodes", end_key="$EndNodes"):

    # Initializes a list of nodes

    nodes_coordinates = []

    # Initializes a flag to tell if reading is allowed already

    flag_reading = False

    # Iterates through the lines

    for i in range(start_reading_at_index, len(lines_list)):

        # Gets the line

        line = lines_list[i]

        # Verifies if reading is not allowed yet

        if not flag_reading:

            # Verifies if this line is equal to the start key

            if line==start_key:

                # Updates the flag reading to allow for reading

                flag_reading = True

            continue

        # Verifies if the end key has been reached

        elif line==end_key:

            # Modifies the start index for the next index

            start_reading_at_index = i+1

            break 

        # Transforms this line to a list to get the different informa-
        # tion readily available, and independently

        line_info = string_toList("["+line+"]", element_separator=" ")

        # If there are four elements in the list, the first is the node
        # index; the second, the third, and the fourth elements are the
        # coordinates

        if len(line_info)>1:

            # Adds the node coordinates, skipping the first element (
            # which is node index)

            nodes_coordinates.append(line_info[1:len(line_info)])

    return nodes_coordinates, start_reading_at_index