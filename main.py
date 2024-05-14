import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import warnings
import math
import copy

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

def butler_volmer(field_dict: dict, E_range: list, exchange_current_method: str):

    exchange_current_method_list = ['calculated', 'fixed']
    if exchange_current_method not in exchange_current_method_list:
        print(f'Wrong exchange_current_method! Valid methods:{exchange_current_method_list}')
        return None 
    else:
        F = 96485.3329 # Faraday's constant
        h = 6.62607015e-34 # Planck constant
        k_b = 1.380649e-23 # Boltzmann constant
        n = 1 # Number of electrons exchanged
        E_A = 30000 # Activation energy. Has to be calculated using experimental data! [J/mol]. FIND PROPER VALUE. Estimated range [30, 100] [kJ/mol]
        kappa = 1 # Transmition coeff. Values from [0,1]. FIND PROPER VALUE
        R = 8.314 # Universal gas constant [J/mol/K]
        alpha = 0.5 # Transfer coefficient. Obtained experimentally. Values [0,1]. A value of 0.5 means that overpotenital favors anodic and cathodic reaction equaly
        i_dict = {}
        i_field = np.ones(field_dict['T'].shape)
        ni_range = []

        def exchange_current_density(K_0, C_r, C_o):
            return F * K_0 * pow(C_r, alpha) * pow(C_o, alpha)

        for count, potential in enumerate(E_range):
            for i, row in enumerate(i_field):
                for j, value in enumerate(row):
                    # E = field_dict['E'][i,j]
                    E = potential
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
                    electrode_type = electrode_type_list[2]
                    if  electrode_type == 'positive':
                        E_0 = 1
                        if exchange_current_method == 'calculated':
                            # K_0 from reaction: VO(2+) + H2O -> VO2(+) + e(-) + 2H(+)
                            K_0 = K_temp * pow(C_VO_0,1) * pow(C_H2O,1) #without Donnan potentail
                            i_0 = exchange_current_density(K_0, C_VO_0, C_VO_2)
                        elif exchange_current_method == 'fixed':
                            i_0 = 2.47 # literature data [A/cm^2]
                            # i_0 = 2.47e+4  # literature data [A/m^2]
                        ni = E - E_0
                        i_field[i,j] = round(i_0 * np.exp((alpha * n * F) / (R * T) * ni), 5)

                    elif electrode_type == 'negative':
                        E_0 = -0.26 # Does not work! but should xD
                        E_0 = 1 
                        if exchange_current_method == 'calculated':
                            # K_0 from reaction: V(3+) + e(-) -> V(2+) 
                            K_0 = K_temp * pow(C_V_3,1)  #without Donnan potentail
                            i_0 = exchange_current_density(K_0, C_VO_0, C_VO_2)
                        elif exchange_current_method == 'fixed':
                            i_0 = 2.47 # literature data [A/cm^2]
                            # i_0 = 2.47e+4  # literature data [A/m^2]
                        ni = E - E_0
                        i_field[i,j] = round(-i_0 * np.exp(-(alpha * n * F) / (R * T) * ni), 5)

                    elif electrode_type == 'merged':
                        E_0 = 1 - (-0.26)
                        if exchange_current_method == 'calculated':
                            # K_0 from reaction: V(3+) + e(-) -> V(2+) 
                            # K_0 = K_temp * pow(C_V_3,1)  #without Donnan potentail
                            # i_0 = exchange_current_density(K_0, C_VO_0, C_VO_2)
                            print('NOT IMPLEMENTED')
                        elif exchange_current_method == 'fixed':
                            i_0 = 2.47 # literature data [A/cm^2]
                            # i_0 = 2.47e+4  # literature data [A/m^2]
                        ni = E - E_0
                        i_field[i,j] = round(i_0 * (np.exp((1-alpha) * F * ni / (R * T)) - np.exp(-(alpha * F * ni / (R * T)))), 5)

                    else:
                        print("Wrong electrode type!!!")
            
            ni_range.append(ni)
            i_dict[f'i_{count}'] = copy.deepcopy(i_field)

        # Plotting an exemplary plot from the 1st cell of every field
        i_range = []
        for key, value in i_dict.items():
            i_range.append(value[0,0])

        plt.scatter(ni_range, i_range)
        plt.xlabel('E_electrode - E_Equilibrium, [V]')
        plt.ylabel('i_electrode, [A/cm2]')
        plt.show()


        return i_dict

def ButlerVolmer(E_range: list, electrode_type: str):
    F = 96485.3329 # Faraday's constant
    h = 6.62607015e-34 # Planck constant
    k_b = 1.380649e-23 # Boltzmann constant
    n = 1 # Number of electrons exchanged
    E_A = 0.3 # Activation energy. Has to be calculated using experimental data! [J/mol]. FIND PROPER VALUE. Estimated range [30, 100] [kJ/mol]
    kappa = 1 # Transmition coeff. Values from [0,1]. FIND PROPER VALUE
    R = 8.314 # Universal gas constant [J/mol/K]
    alpha = 0.5 # Transfer coefficient. Obtained experimentally. Values [0,1]. A value of 0.5 means that overpotenital favors anodic and cathodic reaction equaly
    C_VO_2 = 0.5
    C_VO_0 = 0.5
    C_H = 0.5
    C_H2O = 0.5
    T = 298
    A = (k_b * T) / h  # Pre-exponetial factor in Aarhenius equation. Obtained from experiment or ny assumtion. Transmition state heory is implemented here [1/s]
    print(f'Pre-exp factor: {A}')
    K_temp = A * np.exp(-E_A / (R * T))
    print(f'K_temp {K_temp}')

    i_list = []
    for E in E_range:
        if electrode_type == 'positive':
            E_0 = 1
            K_0 = K_temp * pow(C_VO_0,1) * pow(C_H2O,1) # without Donnan potentail
            # print(f'K_0 {K_0}')
            # i_0 = F * K_0 * pow(C_VO_0, alpha) * pow(C_VO_2, alpha)
            i_0 = 2.47 # literature data [A/cm^2]
            i = i_0 * np.exp((alpha * n * F) / (R * T) * (E - E_0))
            i_list.append(i)

    print(f'local electrode potential: {E_range}')
    print(f'local current: {i_list}')
    plt.scatter(E_range, i_list)
    plt.show()

if __name__ == '__main__':

    fields = ['U', 'p', 'E','I', 'T', 'V_2', 'V_3', 'VO_2', 'VO_0', 'H2O', 'H']
    field_dict = define_field(fields, 5, 5)

    for field in fields:
        set_field_excell('electrode_fields.xlsx', field, field_dict)

    E_range = [0.3, 0.6, 0.9]
    E_range = [0.9, 0.95, 1, 1.05, 1.1]
    E_range = np.arange(0.75, 1.80, 0.05)
    i_dict = butler_volmer(field_dict, E_range, 'fixed')
    i_range = []
    for key, value in i_dict.items():
        i_range.append(value[0,0])





    i_field = i_dict['i_0']

    plt.imshow(i_field, cmap = 'magma')
    plt.title( "Current density" )
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.colorbar()
    # plt.show()


    # print(field_dict)


    # E_range = np.arange(0.1, 1.5, 0.1)
    # ButlerVolmer(E_range, 'positive')
    # T = 298
    # A = 2 * (1.38e-23 * T) / 6.26e-34
    # # print(f'\n\nPre_exp: {A}')
    # print("Pre-exp factor: {:e}".format(A))
    # R = 8.314 # Universal gas constant [J/mol/K]
    # E_A = 1e+5
    # exp = np.exp(-E_A / (R * T))
    # print(f'exp: {exp}')
    # k_temp = A*exp 
    # print(f'K_temp: {k_temp}')
