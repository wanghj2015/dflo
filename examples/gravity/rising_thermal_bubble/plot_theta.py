
import vtk
from vtk.util.numpy_support import vtk_to_numpy

import matplotlib.pyplot as plt
import numpy as np


def potential_temperature(rho, P):
    """
    Computes the potential temperature from density and pressure.

    Args:
    rho (float): density in kg/m^3
    P (float): pressure in Pa

    Returns:
    theta (float): potential temperature in Kelvin
    """

    R_gas = 287.058  # J/kg/K, specific gas constant for dry air
    #Cp = 1005.0  # J/kg/K, specific heat capacity at constant pressure for dry air
    #Cp = 1004.703  # J/kg/K, specific heat capacity at constant pressure for dry air

    gamma = 1.4
    Cp = gamma / (gamma-1.0) * R_gas


    # Calculate temperature from density and pressure
    T = P / (rho * R_gas)

    # Calculate potential temperature from temperature and pressure
    theta = T * (1000.0 / P) ** (R_gas / Cp)

    return theta


# Replace 'filename.vtu' with the name of your VTU file
filename = 'solution-034.vtu'

# Create a reader object and set the input filename
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(filename)

# Read the file
reader.Update()

# Get the output data object from the reader
data = reader.GetOutput()

# Access data in the file
num_points = data.GetNumberOfPoints()
num_cells = data.GetNumberOfCells()

# Print some information about the file
print(f'Number of points: {num_points}')
print(f'Number of cells: {num_cells}')


# Access point data in the file
points = data.GetPoints()

coordinates = vtk_to_numpy(points.GetData())

print (coordinates.shape)

x = coordinates[:,0]
y = coordinates[:,1]
#z = coordinates[:,2]


print ('x = ', np.min(x), np.max(x))
print ('y = ', np.min(y), np.max(y))


# Access scalar data in the file

#density = vtk_to_numpy(data.GetPointData().GetArray("density"))
density = vtk_to_numpy(data.GetPointData().GetScalars("Density"))

print ('density = ', np.min(density), np.max(density))

pressure = vtk_to_numpy(data.GetPointData().GetScalars("Pressure"))

print ('pressure = ', np.min(pressure), np.max(pressure))


potential = potential_temperature(density, pressure)

T0 = 300.0
potential  = potential - T0


# Create a plot of the data
nrows = 2
ncols = 2

fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 6))


im = axs[0,0].tricontourf(x, y, potential, levels=20)#, vmin=0.5, vmax=1.5)
fig.colorbar(im, ax=axs[0,0])

#im = axs[0,1].tricontourf(x, y, density, levels=20)#, vmin=0.5, vmax=1.5)
#fig.colorbar(im, ax=axs[0,1])

#im = axs[1,0].tricontourf(x, y, exb_drift_u, levels=20)#, vmin=0.5, vmax=1.5)
#fig.colorbar(im, ax=axs[1,0])

#im = axs[1,1].tricontourf(x, y, exb_drift_v, levels=20)#, vmin=0.5, vmax=1.5)
#fig.colorbar(im, ax=axs[1,1])


# Set plot parameters
#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_xlim(x.min(), x.max())
#ax.set_ylim(y.min(), y.max())

# Show the plot
plt.show()


