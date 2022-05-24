#!/usr/bin/env python

from ase.io import read, write
import os
from ase.calculators.aims import Aims, AimsCube
import numpy as np
from ase.io.cube import read_cube_data
import math
import sys
from ase import Atom

# Set your environment variable to the species directory you want to use
os.environ['AIMS_SPECIES_DIR'] = '/home/e05/e05/mat92/Codes/FHIaims_Aug21/species_defaults/defaults_2020/intermediate/'

# These functions handle cube files and are borrowed from Reini Maurer's group https://github.com/maurergroup/cube_tools

def arr2D(list2D):
    '''arr2D converts a 2D list into a two-dimensional np.array. This is faster than converting a 2D list with the function np.array,
    because the latter involves a lot of checks on the dimensions of the elements of the list.'''
    h = len(list2D)
    w = len(list2D[0])
    array = np.empty((h,w))
    for i in range(h):
        for j in range(w):
            array[i,j] = list2D[i][j]
    return array

def arr1D(list1D):
    length = len(list1D)
    array = np.empty(length)
    for i in range(length):
        array[i] = list1D[i]
    return array

class cube:
    '''The cube class represents an object used for storing and manipulating data read from cube files. '''
    def __init__(self):
        self.bohr2ang = 0.52917721092
        self.n_atoms = None
        self.origin = None
        self.x_len = None
        self.y_len = None
        self.z_len = None
        self.n_points = None
        self.x_vec = None
        self.y_vec = None
        self.z_vec = None
        self.xyz_array = None
        self.jacobi = None
        self.filename = None
        self.atoms_array = None
        self.density = None
        self.cube_file_set = False

    def __call__(self,pos_x, pos_y,pos_z, silent=False):
        """
        This command takes three real numbers that correspond to a
        Cartesian x, y, and z position and returns the value of the
        Voxel that is closest to this position.
        """

        vec = [pos_x,pos_y,pos_z]
        if self.cube_file_set:
            cube_dimensions = self.xyz_array*[self.x_len,self.y_len,self.z_len]
            cube_dim_inv = np.linalg.inv(cube_dimensions)
            frac_coord = np.dot(cube_dim_inv.T,vec)
            #frac_coord = np.dot(vec-self.origin,cube_dim_inv)   # If you set an origin for your cubefile, you might need to calculate the fractional coordinates like this
            int_coord = np.array(frac_coord * [self.x_len,self.y_len,self.z_len],dtype=np.int)
            copy_density = self.density.reshape([self.x_len,self.y_len,self.z_len])
            return copy_density[int_coord[0],int_coord[1],int_coord[2]]
        else:
            print('First set a cube file with self.read')

    def read(self, filename, castep2cube_format=False):
        ''' The function reads a cube file starting from the third line (the first two lines are usually reserved for comments and contain no data).
        cube.read returns a one-dimensional np.array in which each element resembles a value of the density data in the cube file.
        The function also sets the variables: n_atoms, origin, x_len, y_len, z_len, n_points, x_vec, y_vec, z_vec, xyz_array, density and jacobi
        of the corresponding cube object.'''

        if filename:
            self.filename=str(filename)
        cubefile = open(self.filename, "r")
        print(("Reading cube file: ", cubefile.name))
        next(cubefile)
        next(cubefile)
        cube_content_list = [line for line in cubefile]
        cube_content_joined = "".join(cube_content_list)
        cubefile.close()
        cube_content_split = cube_content_joined.split()
        cube_content = list(map(float, cube_content_split))
        self.n_atoms = int(cube_content[0])
        self.origin = np.array(cube_content[1:4])*self.bohr2ang
        self.x_len = int(cube_content[4])
        self.y_len = int(cube_content[8])
        self.z_len = int(cube_content[12])
        self.n_points = self.x_len*self.y_len*self.z_len
        self.x_vec = np.array(cube_content[5:8])
        self.y_vec = np.array(cube_content[9:12])
        self.z_vec = np.array(cube_content[13:16])
        self.xyz_array = np.array([self.x_vec,self.y_vec,self.z_vec])*self.bohr2ang
        self.jacobi = np.linalg.det(self.xyz_array)
        self.atoms_array = arr1D(cube_content[16:16+5*self.n_atoms])
        self.atoms_array.shape = (self.n_atoms, 5)
        self.cube_file_set = True
        if castep2cube_format:
            self.density = arr1D(cube_content[16+5*self.n_atoms:len(cube_content)]) / self.jacobi
        else:
            self.density = arr1D(cube_content[16+5*self.n_atoms:len(cube_content)])
            
    def write(self, filename_out):
        '''writes out the header of a cube file'''
        fd=open('./'+str(filename_out), 'w+')
        fd.write('Density:\n')
        fd.write('\n')
        fd.write(str('%5i %12.6f %12.6f %12.6f\n') %(self.n_atoms, self.origin[0]/self.bohr2ang, self.origin[1]/self.bohr2ang, 
            self.origin[2]/self.bohr2ang))
        fd.write(str('%5i %12.6f %12.6f %12.6f\n') %(self.x_len, self.x_vec[0], self.x_vec[1], self.x_vec[2]))
        fd.write(str('%5i %12.6f %12.6f %12.6f\n') %(self.y_len, self.y_vec[0], self.y_vec[1], self.y_vec[2]))
        fd.write(str('%5i %12.6f %12.6f %12.6f\n') %(self.z_len, self.z_vec[0], self.z_vec[1], self.z_vec[2]))
        for i in range(len(self.atoms_array)):
                fd.write(str('%5i') %(self.atoms_array[i][0]))
                fd.write(str('%12.6f %12.6f %12.6f') %(self.atoms_array[i][1], self.atoms_array[i][2], \
                                                       self.atoms_array[i][3]))
                fd.write(str('%12.6f') %(self.atoms_array[i][4]))
                fd.write('\n')
        fd.close()



# Get the cubefile data, or read in a simple geometry here, and make sure it has periodic boundary conditions

data, atoms = read_cube_data('cube_001_hartree_potential.cube')
# atoms = read('geometry.in')
atoms.set_pbc(True)
atoms.wrap()




# This is the actual making of the grid

grid_density = 2 # Number of grid points per Angstrom in one direction

# Get norm of unit cell parameters to get number of grid points
    
norm = [np.sqrt(x[0]**2 + x[1]**2 + x[2]**2) for x in atoms.get_cell()]
print('The norm is ', norm)

# Get dimensions of grid

grid_dim = [int(round(x * grid_density)) for x in norm]

print('The grid dimensions are ', grid_dim)

# Get steps for grid point in real space

steps = [x / y for x,y in zip(atoms.get_cell(), grid_dim)]
print('The steps of the grid are ', steps)

# Make grid

grid = [x * steps[0] + y * steps[1] + z * steps[2] for x in range(grid_dim[0]) for y in range(grid_dim[1]) for z in range(grid_dim[2])]

print('There are ', len(grid), ' grid points')




# This filters that atoms aren't too close to the structure; this can be a simple sanity check with a low value, or you can look up typical bond distances of the atom you want to insert (e.g. Lithium)

atom_min_dist = 1.5 # Set minimal distance to an atom in Angstrom

# Find distances of grid points to atoms and filter the grid iteratively

grid_atom_min_filtered = grid

for atom in atoms:
    distances = [np.sqrt((atom.position[0] - x[0])**2 + (atom.position[1] - x[1])**2 + (atom.position[2] - x[2])**2) \
                for x in grid_atom_min_filtered]
    grid_atom_min_filtered = [x for x,y in zip(grid_atom_min_filtered,distances) if y > atom_min_dist]


print('There are now ', len(grid_atom_min_filtered), 'gridpoints after filtering for minimum distance')




# This filters out grid points that based on hartree potential would not lead to a voltage upon insertion, aka being at internal zero

# First find out background hartree potential value, i.e. in the pore center if it exists

pore_dist = 2 # Value from which onwards we consider things to be in a pore - choose this wisely based on the MOF at hand (might need to automate this based on diameter at some point)

pore_min_dist=0
pore_center=[]

for x in grid_atom_min_filtered:
    distances = [np.sqrt((atom.position[0] - x[0])**2 + (atom.position[1] - x[1])**2 + (atom.position[2] - x[2])**2) \
                for atom in atoms]
    if all(dist > pore_dist for dist in distances):
        if min(distances) > pore_min_dist:
            pore_min_dist = min(distances)
            pore_center = x

if pore_min_dist == 0:
    print('No pore center found at this distance; decrease the value to force finding it, but be aware that a plateau might not exist when the pore is small!')
    sys.exit()
else:    
    print('The minimum distance of the pore center from the framework is ', pore_min_dist, 'Angstrom, and its coordinates are ', pore_center)

cubefile = cube()

cubefile.read('cube_001_hartree_potential.cube')

hartree_zero = cubefile(pore_center[0], pore_center[1], pore_center[2]) # Set pore center value as zero

print('Internal zero has been set at ', hartree_zero, 'eV')

pot_min = 0.01 # Set value to filter grid by expected interaction strength (in Hartree)

grid_hartree_filtered = [x for x in grid_atom_min_filtered if abs(cubefile(x[0], x[1], x[2]) - hartree_zero) > pot_min]

print('There are now ', len(grid_hartree_filtered), 'gridpoints after filtering for Hartree potential interaction')




# Now we create new input files based on the grid we made, adding an extra atom (e.g. Lithium). Make sure to set magnetic moments appropriately in this case as well - e.g. by reading in the magnetic moments of a prior pristine calculation, or by setting them by hand. This also filters out geometric violations that stem from another periodic image of the cell

struct_in = read('geometry.in')

Li_atom = Atom('Li')
atoms_copy = atoms
atoms_copy.set_pbc(True)
atoms_copy.wrap()
atoms_copy.set_initial_magnetic_moments(struct_in.get_initial_magnetic_moments())
atoms_copy.append(Li_atom)

index = 0

#if not os.path.exists('Calculations'):
#    os.makedirs('Calculations')


grid_final = []

cubefile = AimsCube(plots=('hartree_potential', 'spin_density'))   # This creates the FHI-Aims input parameters

calc = Aims(
    xc=('hse06', 0.11),
    vdw_correction_hirshfeld='.true.',
#            hybrid_xc_coeff=0.5,
    hse_unit='B',
    sc_accuracy_etot=1e-4,
    sc_accuracy_eev=1e-2,
    sc_accuracy_rho=1e-5,
    sc_accuracy_forces=1e-2,
#        relax_geometry=('trm', 1e-2),
#        relax_unit_cell='full',
    collect_eigenvectors='.false.',
    spin='collinear',
#            charge=-1,
#            fixed_spin_moment=1.0,
        default_initial_moment=0.0,
    relativistic=('atomic_zora','scalar'),
    k_grid=(1,1,1),
    cubes=cubefile)

for x in grid_hartree_filtered:
    print(x)
    atoms_copy[-1].position = x
    distances = []
    for atom in atoms_copy:
        distance = atoms_copy.get_distances(-1, atom.index, mic=True)
        distances.append(distance[0])
    distances.pop()
    print(min(distances))
    if min(distances)>1.5:                                            # This is to reflect a typical minimum bond distance of Lithium to be 1.5 Angstrom
        if not os.path.exists('Calculations/{i}'.format(i=index)):
            os.makedirs('Calculations/{i}'.format(i=index))
        os.chdir('Calculations/{i}'.format(i=index))
        atoms_copy.calc = calc
        calc.write_input(atoms_copy)
        os.chdir('../../')


        index += 1
        grid_final.append(x)

print('The final grid has ', len(grid_final), 'gridpoints')
with open('grid_final.txt', 'w') as f:
    for gridpoint in grid_final:
        f.write("{Entry}\n".format(Entry=gridpoint))

