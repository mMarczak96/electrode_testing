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

def set_field_excell(path: str, name: str, field_dict):
    
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

def butler_volmer(field_dict: dict, E_range: list):

    F = 96485.3329 # Faraday's constant
    h = 6.62607015e-34 # Planck constant
    k_b = 1.380649e-23 # Boltzmann constant
    n = 1 # Number of electrons exchanged
    E_A = 1 # Activation energy. Has to be calculated using experimental data! [J/mol]. FIND PROPER VALUE
    kappa = 1 # Transmition coeff. Values from [0,1]. FIND PROPER VALUE
    R = 8.314 # Universal gas constant [J/mol/K]
    alpha = 0.5 # Transfer coefficient. Obtained experimentally. Values [0,1]. A value of 0.5 means that overpotenital favors anodic and cathodic reaction equaly
    i_dict = {}
    i_field = np.ones(field_dict['T'].shape)

    for count, value in enumerate(E_range):
        for i, row in enumerate(i_field):
            for j, value in enumerate(row):
                E = field_dict['E'][i,j]
                E = value
                print(E)
                C_V_2 = field_dict['V_2'][i,j]
                C_V_3 = field_dict['V_3'][i,j]
                C_VO_2 = field_dict['VO_2'][i,j]
                C_VO_0 = field_dict['VO_0'][i,j]
                C_H = field_dict['H'][i,j]
                C_H2O = field_dict['H2O'][i,j]
                T = field_dict['T'][i,j]
                A = (k_b * T) / h  # Pre-exponetial factor in Aarhenius equation. Obtained from experiment or ny assumtion. Transmition state heory is implemented here [1/s]
                K_temp = A * np.exp(-E_A / (R * T))
                electrode_type_list = ['positive', 'negative', 'merged']
                electrode_type = electrode_type_list[0]
                if  electrode_type == 'positive':
                    E_0 = 1
                    K_0 = K_temp * pow(C_VO_0,1) * pow(C_H2O,1) #without Donnan potentail
                    i_0 = F * K_0 * pow(C_VO_0, alpha) * pow(C_VO_2, alpha)
                    i_field[i,j] = i_0 * np.exp((alpha * n * F) / (R * T) * (E - E_0))

                elif electrode_type == 'negative':
                    E_0 = -0.26
                    K_0 = K_temp * pow(field_dict['V_3'][i,j],1)

                elif electrode_type == 'merged':
                    E_0 = 1 - (-0.26)

                else:
                    print("Wrong electrode type!!!")

        i_dict[f'i_{count}'] = i_field
    
    return i_dict


if __name__ == '__main__':

    fields = ['U', 'p', 'E','I', 'T', 'V_2', 'V_3', 'VO_2', 'VO_0', 'H2O', 'H']
    field_dict = define_field(fields, 5, 5)

    for field in fields:
        set_field_excell('electrode_fields.xlsx', field, field_dict)

    E_range = [-0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08]
    E_range = [1, 1.25, 1.5]
    i_dict = butler_volmer(field_dict, E_range)
    i_range = []
    for key, value in i_dict.items():
        i_range.append(value[0,0])

    # print(i_range)
    plt.scatter(E_range, i_range)
    # plt.show()


    i_field = i_dict['i_0']

    # plt.imshow(i_field, cmap = 'magma')
    # plt.title( "Current density" )
    # plt.xlabel('x-axis')
    # plt.ylabel('y-axis')
    # plt.colorbar()
    # plt.show()


    # print(field_dict)