# Routine to store classes of tetrahedron finite elements. Each class is
# made for a type of finite element

# Defines a class to store the tetrahedron element with quadratic shape
# functions and 10 nodes

class Tetrahedron:

    def __init__(self):
        
        pass 

    # Defines a function to return the 10 quadratic shape functions

    def get_shape_functions(self, r, s, t):

        # First node: r = 1, s = 0, t = 0

        N_1 = r*((2*r)-1.0)

        # Second node: r = 0, s = 1, t = 0

        N_2 = s*((2*s)-1.0)

        # Third node: r = 0, s = 0, t = 1

        N_3 = t*((2*t)-1.0)

        # Evaluates the u quantity

        u = 1.0-r-s-t

        # Fourth node: r = 0, s = 0, t = 0

        N_4 = u*((2*u)-1.0)

        # Fifth node: r = 0.5, s = 0.5, t = 0

        N_5 = 4*r*s

        # Sixth node: r = 0, s = 0.5, t = 0.5

        N_6 = 4*s*t

        # Seventh node: r = 0, s = 0, t = 0.5

        N_7 = 4*t*u

        # Eigth node: r = 0.5, s = 0, t = 0

        N_8 = 4*r*u

        # Nineth node: r = 0.5, s = 0, t = 0.5

        N_9 = 4*r*t

        # Tenth node: r = 0, s = 0.5, t = 0

        N_10 = 4*s*u