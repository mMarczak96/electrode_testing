import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import scipy as sp
import warnings
import math

def define_field(fields: list,width: int, height: int):

    field_dict = {}
    for field in fields:
        field_dict[field] = np.ones([height, width])

    return field_dict

def set_field(path: str, name: str, field_dict):
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        field_df = pd.read_excel(path, sheet_name=name, header=None, index_col=None, engine="openpyxl")
    
    array = field_dict[name]
    for i, row in enumerate(field_df.values):
        for j, value in enumerate(row):
            array[i, j] = value

def calculate_field(field: np.array, function: callable):
    for i, row in enumerate(field):
        for j, value in enumerate(row):
            field[i, j] = function
    return 0

def butler_volmer(field_dict):

    F = 96485.3329 # Faraday's constant
    E_A = 1 # Activation energy. Has to be calculated using experimental data! [J/mol]
    A = 1 # Pre-exponetial factor in Aarhenius equation. Obtained from experiment! [1/s]
    R = 8.314 # Universal gas constant [J/mol/K]
    alpha = 0.5 # Transfer coefficient. Obtained experimentally. Values [0,1]. A value of 0.5 means that overpotenital favors anodic and cathodic reaction equaly

    temp_field = field_dict['T']
    print(temp_field[1,1])

    i_0 = np.ones(field_dict['T'].shape)

    for i, row in enumerate(i_0):
        for j, value in enumerate(row):
            K_temp = value * A * np.exp(-E_A / (R * field_dict['T'][i, j]))
            K_0 = K_temp * pow(field_dict['V_2'][i,j],1) * pow(field_dict['H'][i,j],2)
            C_r = field_dict['V_2'][i,j]
            C_o = field_dict['H'][i,j]
            i_0[i,j] = F * K_0 * pow(C_r, alpha) * pow(C_o, 1- alpha)

    print(i_0)
    return i_0


if __name__ == '__main__':

    fields = ['U', 'p', 'E','I', 'T', 'V_2', 'V_3', 'H']
    field_dict = define_field(fields, 5, 5)

    for field in fields:
        set_field('electrode_fields.xlsx', field, field_dict)

    i_0 = butler_volmer(field_dict)

    plt.imshow(i_0, cmap = 'magma')
    plt.title( "Current density" )
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.colorbar()
    plt.show()


    # print(field_dict)