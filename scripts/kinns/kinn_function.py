"""Module for the naive kinetics-informed neural network"""

import json
import re
import itertools
from collections import OrderedDict, defaultdict

import jax
import jax.numpy as np
from jax import grad, jit, vmap, jacfwd, pmap, random
from jax.example_libraries import optimizers
from jax.tree_util import tree_map
from jax.nn import tanh, swish, relu, sigmoid

from IPython.display import clear_output
import numpy as onp
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp, cumtrapz
from scipy.interpolate import interp1d

import tapsap

# Set up JAX configuration
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)

onp.random.seed(0)

def load_data(conc_file = None, flux_file = None, exp_file = None,
              conc_data = None, net_flux = None,
              pulse_train = None, pulse_test = None, cat_length = 0.17364,
              sample_sequence = [[14, 40, 1], [40, 100, 3], [100, 300, 5], [300, None, 30]]):
    '''

    Load data from json files and return the training and possible testing data.

    Args:
    conc_file: str
        The path to the concentration data file.
    flux_file: list
        Consists of the path to the flux in and flux out data file.
    exp_file: str
        The path to the outlet flux file, used for 0th moment calculation.
        Only need for multi-pulse calculation.
    pulse_train: list
        The pulse number for training data. If not specified, pulse 0 is used.
    pulse_test: list
        The pulse number for testing data.
    cat_length: float
        The catalyst zone length.
    sample_sequence: list
        The sequence for sampling the data.
        Default indices are set for non-uniform sampling to achieve higher resolution in 
        high derivative regions. This also result in sample sizes that are multiples 
        of 2, 5, 8, 10, and 20, which are typical number of processors we use in parallelization.

    Returns:
    A tuple of training data, which consists of:
        - Scaled concentration
        - Log-transformed time data
        - Net flux data
        - Scale of the concentration data
        - Total number of species

    '''
    if conc_file is not None:
        f = open(conc_file)
        thin_data = json.load(f)
        f.close()
        thin_data = json.loads(thin_data)['1']
    else:
        thin_data = conc_data

    if pulse_train is None:
        pulse_train = ['0']
    else:
        pulse_train = [str(i) for i in pulse_train]

    if pulse_test is None:
        pulse_test = []
    else:
        pulse_test = [str(i) for i in pulse_test]

    if exp_file is not None:
        n_moments = 1
        moment_dict = {}
        with open(exp_file, 'r') as f:
            experimental_data = json.load(f)
        experimental_data = json.loads(experimental_data, object_pairs_hook=OrderedDict)['1']

        for j, k in enumerate(experimental_data):
            if k == 'time':
                continue
            else:
                moment_dict[k] = {}
                for i in experimental_data[k].keys():
                    moment_dict[k][i] = {}
                    sub_times = np.array(experimental_data['time'][i])

                    for n in range(n_moments):
                        moment_dict[k][i]['M' + str(n)] = float(jax.scipy.integrate.trapezoid(np.array(experimental_data[k][i]) *
                                                                                              sub_times**n, sub_times))
        moment_train = []
        moment_test = []
        for j, k in enumerate(moment_dict):
            if k == 'Ar':
                continue
            else:
                for i in moment_dict[k].keys():
                    if i in pulse_train:
                        moment_train.append(np.array(moment_dict[k][i]['M0']))
                    elif i in pulse_test:
                        moment_test.append(np.array(moment_dict[k][i]['M0']))
                    else:
                        continue
        moment_train = np.array(moment_train).reshape(-1, len(pulse_train)).T
        try:
            moment_test = np.array(moment_test).reshape(-1, len(pulse_test)).T
        except ZeroDivisionError:
            print('No testing data available. Please specify pulse_test.')
            return

    n_species = 0
    y_train = []
    t_train = []

    y_test = []
    t_test = []

    for j, k in enumerate(thin_data):
        if k == 'time':
            for i in thin_data[k].keys():
                t_temp = []
                for start, end, step in sample_sequence:
                    if end is not None:
                        t_temp += thin_data[k][i][start:end:step]
                    else:
                        t_temp += thin_data[k][i][start::step]
                if i in pulse_train:
                    t_train.extend(np.array(t_temp))
                elif i in pulse_test:
                    t_test.extend(np.array(t_temp))
            t_train = np.log(np.array([t_train]).T)
            t_test = np.log(np.array([t_test]).T)

            if len(pulse_train) > 1:
                moment_t = []
                for n in range(len(pulse_train)):
                    moment_t.extend(np.broadcast_to(moment_train[n],
                                                    (len(t_temp), len(moment_train[n]))))
                t_train = np.concatenate((t_train, np.array(moment_t)), axis = 1)

                moment_t = []
                for n in range(len(pulse_test)):
                    moment_t.extend(np.broadcast_to(moment_test[n],
                                                    (len(t_temp), len(moment_test[n]))))
                t_test = np.concatenate((t_test, np.array(moment_t)), axis = 1)

        elif k == 'Ar':
            continue
        else:
            y_temp_train = []
            y_temp_test = []
            for i in thin_data[k].keys():
                y_temp_i = []
                if i in pulse_train:
                    for start, end, step in sample_sequence:
                        if end is not None:
                            y_temp_i += thin_data[k][i][start:end:step]
                        else:
                            y_temp_i += thin_data[k][i][start::step]
                    y_temp_train += list(y_temp_i)
                elif i in pulse_test:
                    for start, end, step in sample_sequence:
                        if end is not None:
                            y_temp_i += thin_data[k][i][start:end:step]
                        else:
                            y_temp_i += thin_data[k][i][start::step]
                    y_temp_test += list(y_temp_i)

            y_train.append(np.array(y_temp_train))
            y_test.append(np.array(y_temp_test))

    n_species = int(j) - 1

    # Scale to range 0 to 1
    y_train = onp.array(y_train)
    y_train_scale = np.max(y_train, axis = 1)
    conc_inv = (y_train.T/y_train_scale, t_train)

    n_latent = sum('*' in key for key in thin_data.keys())

    def read_flux(file_path = None):

        if flux_file is not None:
            f_path = file_path
            f = open(f_path)
            flux_data = json.load(f)
            f.close()
            flux_data = json.loads(flux_data)['1']
        else:
            flux_data = net_flux

        f_train = []
        f_test = []
        for _, k in enumerate(flux_data):
            if k == 'time':
                continue
            if k == 'Ar':
                continue
            else:
                f_temp_train = []
                f_temp_test = []
                for i in flux_data[k].keys():
                    if i in pulse_train:
                        for start, end, step in sample_sequence:
                            if end is not None:
                                f_temp_train += flux_data[k][i][start:end:step]
                            else:
                                f_temp_train += flux_data[k][i][start::step]
                    elif i in pulse_test:
                        for start, end, step in sample_sequence:
                            if end is not None:
                                f_temp_test += flux_data[k][i][start:end:step]
                            else:
                                f_temp_test += flux_data[k][i][start::step]
                f_train.append(np.array(f_temp_train))
                f_test.append(np.array(f_temp_test))

        for i in range(n_latent):
            f_train.append(np.zeros(t_train.shape[0]))
            f_test.append(np.zeros(t_test.shape[0]))
        f_train = np.array(f_train).T
        f_test = np.array(f_test).T
        return f_train, f_test

    if net_flux is None:
        flux_in_train, flux_in_test = read_flux(flux_file[0])
        flux_out_train, flux_out_test = read_flux(flux_file[1])
        net_flux_train = (flux_in_train - flux_out_train)/cat_length
    else:
        net_flux_train = read_flux()[0]/cat_length

    data_train = (conc_inv[0], conc_inv[-1], net_flux_train)

    if len(pulse_test) > 0:
        try:
            net_flux_test = (flux_in_test - flux_out_test)/cat_length
        except:
            net_flux_test = read_flux()[1]/cat_length
        y_test = onp.array(y_test)
        y_test_scale = np.max(y_test, axis = 1)
        conc_inv_test = (y_test.T/y_test_scale, t_test)
        data_test = (conc_inv_test[0], conc_inv_test[-1], net_flux_test)
        return (data_train, y_train_scale, n_species), (data_test, y_test_scale, n_species)

    return (data_train, y_train_scale, n_species)


def load_data_y(exp_file, zone_lengths, zone_porosity, mass, reactants, products,
                pulse_num, pulse_train, pulse_test, n_species,
                radius = 1., inert_mass = 40.0, add_noise = 0.,
                sample_sequence = [[4, 40, 1], [40, 100, 3], [100, 300, 5], [300, -100, 60]]):
    '''
    Load TAP outlet flux data from json files, estimating the catalyst zone concentrations and
    flux through Y-Procedure, and returning the training and testing data

    Args:
    exp_file: str
        The path to the outlet flux file.
    zone_lengths: dict
        The length (cm) of each zone in the TAP reactor.
    zone_porosity: dict
        The porosity of each zone in the TAP reactor.
    mass: list
        Atomic mass of gas species.
    reactants: list
        Chemical formula of reactants.
    products: list
        Chemical formula of products.
    pulse_num: int
        Total number of pulses.
    pulse_train: list
        The pulse number for training data.
    pulse_test: list
        The pulse number for testing data.
    n_species: int
        Total number of species (gas and surface).
    radius: float
        The radius of TAP reactor.
    inert_mass: float
        Atomic mass of the inert gass, default as Ar.
    add_noise: float
        The proportional scaling factor used to add noise to data, default to noisefree.
    sample_sequence: list
        The sequence for sampling the data.
        Default indices are set for non-uniform sampling to achieve higher resolution in 
        high derivative regions. This also result in sample sizes that are multiples 
        of 2, 5, 8, 10, and 20, which are typical number of processors we use in parallelization.

    Returns:
    Two tuples for training data and testing data, which consist of:
        - Scaled concentration
        - Log-transformed time data
        - Net flux data
        - Scaled atomic uptake
        - Scale of the concentration data
        - Total number of species
    '''
    f = open(exp_file)
    experimental_data = json.load(f)
    f.close()
    experimental_data = json.loads(experimental_data)['1']

    y_total = []
    t_total = []
    n_moment = 1
    moment_dict = {}
    for j, k in enumerate(experimental_data):
        if k == 'time':
            for i in experimental_data[k].keys():
                t_temp = [experimental_data[k][i][0]]+experimental_data[k][i][10:]
                t_total.extend(np.array(t_temp)+t_temp[-1]*(int(i)))
            t_total = onp.array([t_total]).T

        else:
            y_temp = []
            moment_dict[k] = {}
            for i in experimental_data[k].keys():
                moment_dict[k][i] = {}
                sub_times = np.array(experimental_data['time'][i])
                for n in range(n_moment):
                    moment_dict[k][i]['M' + str(n)] = float(jax.scipy.integrate.trapezoid(np.array(experimental_data[k][i]) *
                                                                                              sub_times**n, sub_times))

                experimental_data_temp = [experimental_data[k][i][0]] + experimental_data[k][i][10:]
                experimental_data_temp = list(np.array(experimental_data_temp))
                y_temp += experimental_data_temp
            y_total.append(onp.array(y_temp))

    out_flux_total = onp.array(y_total).reshape((-1, pulse_num, len(t_temp)))

    cross_sec_area = radius**2 * onp.pi
    volume_cat = zone_lengths['zone1'] * cross_sec_area

    estimate_y_rate = y_rate(out_flux_total, t_total[:len(t_temp)], zone_lengths,
                             zone_porosity, pulse_num, mass, len(reactants), inert_mass)
    net_flux_gas = np.array(estimate_y_rate)/cross_sec_area/zone_lengths['zone1']

    gas_uptake = cumtrapz(estimate_y_rate.reshape(-1, len(t_total)), t_total.flatten(), initial = 0)
    atom_uptake = calculate_atomic_uptake(reactants, products, gas_uptake)
    atom_uptake = np.array(atom_uptake)/volume_cat
    atom_uptake = atom_uptake.reshape((atom_uptake.shape[0], pulse_num, -1))

    estimate_y_conc = np.array(y_conc(out_flux_total, t_total[:len(t_temp)], zone_lengths,
                                      zone_porosity, pulse_num, mass, radius))

    pulse_train = [str(p) for p in pulse_train]
    pulse_test = [str(p) for p in pulse_test]

    moment_train = []
    moment_test = []
    for j, k in enumerate(moment_dict):
        if k == 'Ar':
            continue
        else:
            for i in moment_dict[k].keys():
                if i in pulse_train:
                    moment_train.append(np.array(moment_dict[k][i]['M0']))
                elif i in pulse_test:
                    moment_test.append(np.array(moment_dict[k][i]['M0']))
                else:
                    continue
    moment_train = np.array(moment_train).reshape(-1, len(pulse_train)).T
    try:
        moment_test = np.array(moment_test).reshape(-1, len(pulse_test)).T
    except ZeroDivisionError:
        print('No testing data available. Please specify pulse_test.')
        return

    t_train = []
    t_test = []
    for i in range(pulse_num):
        t_1 = []
        for start, end, step in sample_sequence:
            if end is not None:
                t_1 += t_temp[start:end:step]
            else:
                t_1 += t_temp[start::step]
        if str(i) in pulse_train:
            t_train.extend(np.array(t_1))
        elif str(i) in pulse_test:
            t_test.extend(np.array(t_1))

    t_train = np.log(np.array([t_train]).T)
    t_test = np.log(np.array([t_test]).T)

    moment_t = []
    for n in range(len(pulse_train)):
        moment_t.extend(np.broadcast_to(moment_train[n], (len(t_1), len(moment_train[n]))))
    t_train = np.concatenate((t_train, np.array(moment_t)), axis = 1)
    moment_t = []
    for n in range(len(pulse_test)):
        moment_t.extend(np.broadcast_to(moment_test[n], (len(t_1), len(moment_test[n]))))
    t_test = np.concatenate((t_test, np.array(moment_t)), axis = 1)

    n_gas = len(reactants + products)
    y_train = []
    y_test = []
    for j in range(n_gas):
        y_temp_train = []
        y_temp_test = []
        for i in range(pulse_num):
            y_temp = list(estimate_y_conc[j,i,:])
            y_temp_i = []
            if str(i) in pulse_train:
                for start, end, step in sample_sequence:
                    if end is not None:
                        y_temp_i += y_temp[start:end:step]
                    else:
                        y_temp_i += y_temp[start::step]
                noise = np.std(np.array(y_temp_i)) * add_noise
                y_temp_i_noise = list(np.array(y_temp_i) +
                                      onp.random.normal(0.0, noise, np.array(y_temp_i).shape))
                y_temp_train += y_temp_i_noise
            elif str(i) in pulse_test:
                for start, end, step in sample_sequence:
                    if end is not None:
                        y_temp_i += y_temp[start:end:step]
                    else:
                        y_temp_i += y_temp[start::step]
                noise = np.std(np.array(y_temp_i)) * add_noise
                y_temp_i_noise = list(np.array(y_temp_i) +
                                      onp.random.normal(0.0, noise, np.array(y_temp_i).shape))
                y_temp_test += y_temp_i_noise
        y_train.append(np.array(y_temp_train))
        y_test.append(np.array(y_temp_test))

    y_train = onp.array(y_train)
    y_train_scale = np.max(y_train, axis = 1)

    y_test = onp.array(y_test)
    y_test_scale = np.max(y_test, axis = 1)

    conc_inv = (y_train.T/y_train_scale, t_train)
    conc_inv_test = (y_test.T/y_test_scale, t_test)

    net_flux_train = []
    net_flux_test = []
    for j in range(n_species):
        f_temp_train = []
        f_temp_test = []
        if j < n_gas:
            for i in range(pulse_num):
                f_temp = list(net_flux_gas[j,i,:])
                f_temp_i = []
                if str(i) in pulse_train:
                    for start, end, step in sample_sequence:
                        if end is not None:
                            f_temp_i += f_temp[start:end:step]
                        else:
                            f_temp_i += f_temp[start::step]
                    f_temp_train += f_temp_i
                elif str(i) in pulse_test:
                    for start, end, step in sample_sequence:
                        if end is not None:
                            f_temp_i += f_temp[start:end:step]
                        else:
                            f_temp_i += f_temp[start::step]
                    f_temp_test += f_temp_i
        else:
            f_temp_train.extend(np.zeros(net_flux_train[0].shape))
            f_temp_test.extend(np.zeros(net_flux_test[0].shape))
        net_flux_train.append(np.array(f_temp_train))
        net_flux_test.append(np.array(f_temp_test))
    net_flux_train = np.array(net_flux_train).T
    net_flux_test = np.array(net_flux_test).T

    uptake_train = []
    uptake_test = []
    for j in range(len(atom_uptake)):
        u_temp_train = []
        u_temp_test = []
        for i in range(pulse_num):
            u_temp = list(atom_uptake[j,i,:])
            u_temp_i = []
            if str(i) in pulse_train:
                for start, end, step in sample_sequence:
                    if end is not None:
                        u_temp_i += u_temp[start:end:step]
                    else:
                        u_temp_i += u_temp[start::step]
                noise = np.std(np.array(u_temp_i)) * add_noise
                u_temp_i_noise = list(np.array(u_temp_i) +
                                      onp.random.normal(0.0, noise, np.array(u_temp_i).shape))
                u_temp_train += u_temp_i_noise
            elif str(i) in pulse_test:
                for start, end, step in sample_sequence:
                    if end is not None:
                        u_temp_i += u_temp[start:end:step]
                    else:
                        u_temp_i += u_temp[start::step]
                noise = np.std(np.array(u_temp_i)) * add_noise
                u_temp_i_noise = list(np.array(u_temp_i) +
                                      onp.random.normal(0.0, noise, np.array(u_temp_i).shape))
                u_temp_test += u_temp_i_noise
        uptake_train.append(np.array(u_temp_train))
        uptake_test.append(np.array(u_temp_test))
    uptake_train = np.array(uptake_train)
    uptake_train_scale = np.max(uptake_train)
    uptake_train = uptake_train.T/uptake_train_scale
    uptake_test = np.array(uptake_test)
    uptake_test_scale = np.max(uptake_test)
    uptake_test = uptake_test.T/uptake_test_scale

    data_scale = np.concatenate([y_train_scale,
                                 np.array([uptake_train_scale]*(n_species-n_gas))])
    data_test_scale = np.concatenate([y_test_scale,
                                      np.array([uptake_test_scale]*(n_species-n_gas))])

    data_train = (conc_inv[0], conc_inv[-1], net_flux_train, uptake_train)
    data_test = (conc_inv_test[0], conc_inv_test[-1], net_flux_test, uptake_test)

    return (data_train, data_scale, n_species), (data_test, data_test_scale, n_species)


def y_rate(out_flux_all, t, zone_lengths, zone_porosity, pulse_num, mass, n_reactant,
           inert_mass = 40.0, baseline_amount = 0.5):
    '''
    Estimate reaction rate using Y-Procedure

    Args:
    out_flux_all: array
        The TAP outlet flux.
    t: array
        The data collection timestamps.
    zone_lengths: dict
        The length (cm) of each zone in the TAP reactor.
    zone_porosity: dict
        The porosity of each zone in the TAP reactor.
    pulse_num: int
        Total number of pulses.
    mass: list
        Atomic mass of gas species.
    n_reactant: int
        The number of reactant.
    inert_mass: float
        Atomic mass of the inert gass, default as Ar.
    baseline_amount: float
        The points in time to take the baseline.

    Return:
    The Y-Procedure estimated rate
    '''
    t = t.flatten()
    estimate_y_rate = []

    for j in range(3):
        estimate_y_rate_temp = []
        for i in range(pulse_num):
            out_flux = out_flux_all[:, i, :]

            out_flux = tapsap.baseline_correction(out_flux.T, t, 
                                                  [t[-1]-baseline_amount, t[-1]])['flux'].T

            inert_flux = out_flux[-1]
            inert_moments = tapsap.moments(inert_flux, t)

            out_flux = out_flux/inert_moments['M0']

            diffusion = tapsap.diffusion_moments(inert_moments, zone_lengths, zone_porosity, 
                                                 inert_mass, mass[j])['diffusion']

            if j < n_reactant:
                inert_graham = tapsap.grahams_law(inert_flux, t, float(40), mass[j])
                estimate_y_rate_temp.append(tapsap.rate_y(onp.array(out_flux[j, :]),
                                                          onp.array(t),diffusion,zone_lengths,zone_porosity,inert_flux = inert_graham, smoothing_amt= 3)*inert_moments['M0'])
            else:
                estimate_y_rate_temp.append(-tapsap.rate_y(onp.array(out_flux[j, :]),
                                                           onp.array(t),diffusion,zone_lengths,zone_porosity,inert_flux = None, smoothing_amt= 3)*inert_moments['M0'])

        estimate_y_rate.append(estimate_y_rate_temp)
    return onp.array(estimate_y_rate)


def calculate_atomic_uptake(reactants, products, gas_uptake):
    '''
    Calculate atomic uptake from the gas uptake

    Args:
    reactants: list
        Chemical formula of reactants.
    products: list
        Chemical formula of products.
    gas_uptake: array
        The gas uptake as the cumulative integral of the rate.

    Return:
    The atomic uptake

    '''
    def parse_molecule(molecule):
        elements = defaultdict(int)
        i = 0
        while i < len(molecule):
            if i + 1 < len(molecule) and molecule[i+1].isdigit():
                elements[molecule[i]] += int(molecule[i+1])
                i += 2
            else:
                elements[molecule[i]] += 1
                i += 1
        return elements

    def get_all_elements(molecules):
        all_elements = set()
        for molecule in molecules:
            all_elements.update(parse_molecule(molecule).keys())
        return sorted(list(all_elements))

    all_molecules = reactants + products
    all_elements = get_all_elements(all_molecules)
    atomic_uptake = onp.zeros((len(all_elements), gas_uptake.shape[-1]))

    for i, molecule in enumerate(all_molecules):
        parsed_molecule = parse_molecule(molecule)
        for j, element in enumerate(all_elements):
            atomic_uptake[j] +=  parsed_molecule[element] * gas_uptake[i]

    return atomic_uptake


def y_conc(out_flux_all, t, zone_lengths, zone_porosity, pulse_num, mass, radius,
           inert_mass = 40.0, baseline_amount = 0.3):
    '''
    Estimate thin zone concentrations for gas species using Y-Procedure

    Args:
    out_flux_all: array
        The TAP outlet flux.
    t: array
        The data collection timestamps.
    zone_lengths: dict
        The length (cm) of each zone in the TAP reactor.
    zone_porosity: dict
        The porosity of each zone in the TAP reactor.
    pulse_num: int
        Total number of pulses.
    mass: list
        Atomic mass of gas species.
    radius: float
        The TAP reactor radius.
    inert_mass: float
        Atomic mass of the inert gass, default as Ar.
    baseline_amount: float
        The points in time to take the baseline.

    Return:
    The Y-Procedure estimated concentration
    '''

    t = t.flatten()
    estimate_y_conc = []

    for j in range(3):
        estimate_y_conc_temp = []

        for i in range(pulse_num):
            out_flux = out_flux_all[:, i, :]
            out_flux = tapsap.baseline_correction(out_flux.T, t, [t[-1]-baseline_amount, t[-1]])['flux'].T

            inert_flux = out_flux[-1]
            inert_moments = tapsap.moments(inert_flux, t)

            out_flux = out_flux/inert_moments['M0']

            diffusion = tapsap.diffusion_moments(inert_moments, zone_lengths,
                                                 zone_porosity, inert_mass, mass[j])
            concentration_units = tapsap.concentration_units(diffusion['diffusion'],
                                                             zone_lengths, radius, inert_moments['M0'])

            estimate_y_conc_temp.append(tapsap.concentration_y(onp.array(out_flux[j, :]),
                                                               onp.array(t), diffusion['diffusion'],
                                                               zone_lengths, zone_porosity, smoothing_amt = 6)*concentration_units)

        estimate_y_conc.append(estimate_y_conc_temp)
    return onp.array(estimate_y_conc)


def encode_reactions(reactions, species):
    '''
    Encode the reactions into a format that can be used for the neural network
    
    Args:
    reactions: list
        The list of reactions.
    species: list
        The list of species.
        
    Returns:
    encoded_rxns: list
        The encoded reactions contains the reactants and products 
        and corresponding stoichiometry coefficient 
        and a flag indicating whether the reaction is reversible.
    '''
    def parse_term(term):
        match = re.match(r'(\d*)\s*(.+)', term.strip())
        if match:
            coef = int(match.group(1)) if match.group(1) else 1
            specie = match.group(2)
        else:
            coef, specie = 1, term.strip()
        return coef, species.index(specie)

    encoded_rxns = []
    for reaction in reactions:
        reactants, products = reaction.split('<->' if '<->' in reaction else '->')
        encoded_rxn = []
        for side in [reactants, products]:
            encoded_side = [parse_term(term) for term in side.split('+')]
            encoded_rxn.append(encoded_side)
        encoded_rxn.append(np.array(1.0) if '<->' in reaction else np.array(0.0))
        encoded_rxns.append(encoded_rxn)
    return encoded_rxns

@jit
def calculate_reaction_rates(encoded_rxns, k, concentrations):
    '''
    Calculate the reaction rates for a set of reactions
    
    Args:
    encoded_rxns: list
        The encoded reactions.
    k: np.array
        The rate constants for the reactions.
    concentrations: np.array
        The concentrations of the species.
        
    Returns:
    rates: np.array
        The reaction rates.
        
    '''
    def single_reaction_rate(rxn, k_f, k_b):
        reactants, products, is_reversible = rxn
        forward_rate = k_f
        for coef, idx in reactants:
            forward_rate *= concentrations[idx]**coef
        backward_rate = k_b * is_reversible
        for coef, idx in products:
            backward_rate *= concentrations[idx]**coef
        return forward_rate - backward_rate

    rates = np.zeros_like(concentrations)
    rev_count = 0
    for i, rxn in enumerate(encoded_rxns):
        _, _, rev_flag = rxn
        rate = single_reaction_rate(rxn, k[i+rev_count], k[i+rev_count+1])
        for coef, idx in rxn[0]:  # reactants
            rates = rates.at[idx].add(-coef * rate)
        for coef, idx in rxn[1]:  # products
            rates = rates.at[idx].add(coef * rate)
        rev_count += rev_flag.astype(int)

    return rates


def random_layer_params(m, n, key, scale=1e-2):
    '''
    Initialize the weights and biases for a fully-connected neural network layer
    '''
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


def random_model_params(m, key, scale=1e-2):
    '''
    Initialize the parameters for a kinetic model
    '''
    return (scale * random.normal(key, (m,)))


def init_network_params(sizes, key, scale):
    '''
    Initialize the parameters for a fully-connected neural network
    
    Args:
    sizes: list
        The sizes of the layers in the network.
    key: PRNGKey
        The key for random number generation.
    scale: float
        The scale for the random initialization.
    '''
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k, scale) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


def init_model_params(size, key, scale):
    '''
    Initialize the parameters for a kinetic model

    Args:
    size: int
        The size of the model.
    key: PRNGKey
        The key for random number generation.
    scale: float
        The scale for the random initialization.
    '''
    key = random.split(key,2)[-1]
    return [random_model_params(s, key, scale) for s in size]

@jit
def act_fun(x):
    '''
    The activation function for the neural network
    '''
    return swish(x)

@jit
def state(params, t):
    '''
    Compute the state of the neural network at time t
    '''
    activations = t
    for w, b in params[:-1]:
        outputs = np.dot(w, activations) + b
        activations = act_fun(outputs)

    final_w, final_b = params[-1]
    y = np.abs(np.dot(final_w, activations) + final_b)
    return y
# Make a batched version of the `state` function
batched_state = vmap(state, in_axes=(None,0))

@jit
def diff_state(params,t):
    '''
    Compute the derivative of the state function with respect to time
    '''
    i = np.arange(len(t))
    return np.nan_to_num(jacfwd(lambda t : batched_state(params,t))(t)[i,:,i,0])


def loss(f_errors, params, batch, flux, scale, encoded_rxns, weighted_param):
    '''
    Compute the loss function for the neural network
    '''
    return np.array([_.mean() for _ in f_errors(params, batch, flux, scale,
                                                encoded_rxns, weighted_param)]).sum()


def loss_uptake(f_errors, params, batch, flux, uptake, scale, encoded_rxns, weighted_param):
    '''
    Compute the loss function for the neural network when uptake constraints are applied
    '''
    return np.array([_.mean() for _ in f_errors(params, batch, flux, uptake, scale,
                                                encoded_rxns, weighted_param)]).sum()


@jit
def errors(params, batch, flux, scale, encoded_rxns, weighted_param):
    '''
    Compute the errors for the neural network
    '''
    x, t = batch
    nn_params, model_params = params
    pred_x = batched_state(nn_params, t)
    err_data = ((pred_x-x)**2).mean(axis=1)
    err_model = ((diff_state(nn_params,t) / np.exp(t[:,0:1]) -
                  batched_model([pred_x,t], model_params, flux, scale,
                                encoded_rxns) / scale)**2).sum(axis=1)
    return [err_data, weighted_param*err_model]

@jit
def errors_uptake(params, batch, flux, uptake, scale, encoded_rxns, weighted_param, nt = 30):
    '''
    Compute the errors for the neural network when uptake contraints are applied
    The index need to be changed based on adspecies chemical formula

    nt: total active site concentration
    '''
    x, t = batch
    nn_params, model_params = params
    pred_x = batched_state(nn_params, t)
    err_data = ((pred_x[:,:3]-x)**2).mean(axis=1)
    err_u = (((pred_x[:,-3] - uptake[:,0])**2).sum() +
             ((pred_x[:,-3] + pred_x[:,-2] - uptake[:,1])**2)).sum()
    err_nt = ((pred_x[:,-1] + pred_x[:,-2] + pred_x[:,-3] - nt/scale[-1])**2).sum()
    err_model = ((diff_state(nn_params,t) / np.exp(t[:,0:1]) -
                  batched_model([pred_x,t], model_params, flux, scale,
                                encoded_rxns) / scale)**2).sum(axis=1)
    err = [err_data, err_u, err_nt, err_model]
    return [err[i]*weighted_param[i] for i in range(len(weighted_param))]

@jit
def model(batch, model_params, flux, scale, encoded_rxns):
    '''
    Compute the model for the kinetic model
    '''
    x, _ = batch
    k, = model_params
    k, x = [np.abs(_) for _ in [k, x]]
    x = x * scale
    rates = calculate_reaction_rates(encoded_rxns, k, x)
    return np.reshape(rates + flux, (len(x), 1))
batched_model = lambda batch, model_params, flux, scale, encoded_rxns : vmap(model, in_axes=(0, None, 0, None, None))(batch, model_params, flux, scale, encoded_rxns)[:,:,0]


def initialize_training(layer_sizes, model_size, rand_key = 0, nn_scale = 1e-2, model_scale = 1e-2,
                        param_file = None):
    '''
    Initialize the parameters for the training process
    '''
    key = random.PRNGKey(rand_key)
    if param_file is None:
        nn_params = init_network_params(layer_sizes, key, nn_scale)
        model_params = init_model_params(model_size, key, model_scale)
    else:
        saved_params = np.load(param_file, allow_pickle = True)
        nn_params = saved_params['data'].item()['nn_params']
        model_params = saved_params['data'].item()['model_params']

    params = [nn_params, model_params]
    opt_inv = optimizers.adam(1e-3, b1=0.9, b2=0.9999, eps=1e-100)
    opt_inv = (opt_inv[0](params), opt_inv[1], opt_inv[2], [])
    return params, opt_inv


def train(params, data_train, num_epochs, num_iter, opt_objs, err_tags, encoded_rxns,
          device_count=1, initial_weighted_param=1e-10, ferrors=errors, loss = loss):
    '''
    Train the neural network model
    
    Args:
    params: list
        The parameters for the neural network model.
    data_train: tuple
        The training data.
    num_epochs: int
        The number of epochs for training.
    num_iter: int
        The number of iterations for training.
    opt_objs: tuple
        The optimizer objects.
    err_tags: list
        The tags for the errors.
    encoded_rxns: list
        The encoded reactions.
    device_count: int
        The number of devices for parallelization.
    initial_weighted_param: float
        The initial value for the weighted parameter.
    ferrors: function
        The error function for the neural network.
    
    Returns:
    params: list
        The trained parameters for the neural network model.
    opt_state: list
        The optimizer state.
    opt_update: function
        The optimizer update function.
    get_params: function
        The function to get the parameters.
    iter_data: list
        The data for each iteration.
    '''
    opt_state, opt_update, get_params, iter_data = opt_objs
    data, data_scale, _ = data_train

    @jit
    def step(data, params, weighted_param):
        batch = [data[0], data[1]]
        extra_args = data[2:]

        def loss_wrapper(*args):
            ferrors, params, batch, *rest = args
            return np.nan_to_num(loss(ferrors, params, batch, *rest, encoded_rxns, weighted_param))

        grads = grad(loss_wrapper, argnums=1)(
            ferrors, params, batch, *extra_args, data_scale)

        return grads

    step_parallel = pmap(step, in_axes = (0, None, None))

    def mean_map(grads):
        return tree_map(lambda x: x.mean(axis = 0), grads)
    mean_fn = jit(mean_map)

    @jit
    def update(itercount, grads, opt_state):
        return opt_update(itercount, grads, opt_state)

    itercount = itertools.count()

    data_parallel = tuple(d.reshape(device_count,-1,d.shape[-1]) for d in data)
    iter_data += [get_params(opt_state).copy()]

    weighted_param = initial_weighted_param

    for j in range(num_epochs):
        clear_output(wait=True)

        for _ in range(int(num_iter)):
            params = get_params(opt_state)

            grads_parallel = step_parallel(data_parallel, params, weighted_param)
            grads = mean_fn(grads_parallel)

            opt_state = update(next(itercount), grads, opt_state)

        batch = [data[0], data[1]]

        params = get_params(opt_state)
        loss_it_batch = loss(ferrors, params, batch, *data[2:],
                             data_scale, encoded_rxns, weighted_param)

        errs = [_.sum() for _ in ferrors(params,batch,*data[2:],
                                         data_scale,encoded_rxns,weighted_param)]
        print(weighted_param)
        print('Epoch: {:4d}, Loss Batch: {:.5e}'.format(j, loss_it_batch)+\
                  ''.join([', Fit {}: {:.5e}'.format(_,__) for _,__ in zip(err_tags,errs)]))
        print(params[-1])
        iter_data += [params.copy()]

    return params, [opt_state, opt_update, get_params, iter_data]


def uncertainty_analysis(params, data_train, weight_parameter, encoded_rxns):
    '''
    Evaluate the uncertainty for the kinetic parameters
    
    Args:
    params: list
        The parameters for the neural network model.
    data_train: tuple
        The training data.
    weight_parameter: float
        The ending weighted parameter for the training.
    encoded_rxns: list
        The encoded reactions.
        
    Returns:
    standard_deviation: np.array
        The uncertainties for the kinetic parameters.
    '''
    nn_params, model_params = params

    def error_wrt_params(p):
        if isinstance(weight_parameter, np.ndarray):
            (y_train, t_train, net_flux_train,uptake_train), y_train_scale, _ = data_train
            alpha = weight_parameter
            errs_list = errors_uptake([nn_params, p], [y_train, t_train],net_flux_train,
                                uptake_train,y_train_scale,encoded_rxns,weight_parameter)
            return np.array([(errs_list[i]/alpha[i]).mean() for i in range(len(alpha))]).mean()
        else:
            (y_train, t_train, net_flux_train), y_train_scale, _ = data_train
            alpha = np.array([[1],[weight_parameter]])
            return (np.array(errors([nn_params, p], [y_train, t_train],net_flux_train,
                                y_train_scale,encoded_rxns,weight_parameter))/alpha).mean()

    hessian_func = jit(jax.hessian(error_wrt_params))
    hessian_matrix = hessian_func(model_params)
    covariance_matrix = np.linalg.inv(np.array(hessian_matrix))[0][0]
    standard_deviation = np.sqrt(np.diag(covariance_matrix))

    return standard_deviation


def rebuild_ode(model_params, data_train, encoded_rxn, t_span = (0.0, 2.5),
                x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
                k_true = np.array([15, 0.7, 0.33, 0.4, 0.02, 15.2]),
                species = None,
                xscale = 'log', save_path = None):
    '''
    Rebuild the ODE model using the trained parameters

    Args:
    model_params: list
        The trained parameters for the kinetic model.
    data_train: tuple
        The training data.
    encoded_rxn: list
        The encoded reactions.
    t_span: tuple
        The time span for the ODE model.
    x0: np.array
        The initial condition for the ODE model.
    k_true: np.array
        The true rate constants for the ODE model.
    species: list
        The list of species.
    xscale: str
        The scale for plotting the x-axis.
    save_path: str
        The path for saving the plot.

    Returns:
    solution: np.array
        The solution for the ODE model using the trained kinetic parameters.
    solution_true: np.array
        The solution for the ODE model using the true kinetic parameters.
    '''
    if species is None:
        species = ['CO', 'O$_2$', 'CO$_2$', 'CO*', 'O*', '*']

    (y_train, t_train, net_flux_train), y_train_scale, _ = data_train

    def model_ode(t, x, k, flux, time_points):

        # Interpolate the flux values at the current time t
        flux_interp = interp1d(time_points, flux, axis=1, fill_value = 'extrapolate')
        current_flux = flux_interp(t).reshape(6, 1)
        x = x*y_train_scale
        dCdt = calculate_reaction_rates(encoded_rxn, k, x).reshape(-1,1) + current_flux
        dCdt = dCdt.T/y_train_scale
        return dCdt.flatten()

    k = np.abs(np.array(model_params[0]))
    flux = net_flux_train.T

    time_points_flux = np.exp(t_train).flatten()

    ode_func = lambda t, x: model_ode(t, x, k, flux, time_points_flux)
    ode_true = lambda t, x: model_ode(t, x, k_true, flux, time_points_flux)

    solution = solve_ivp(ode_func, t_span, x0, t_eval=time_points_flux, method = 'LSODA')
    solution_true = solve_ivp(ode_true, t_span, x0, t_eval=time_points_flux)

    x_solution = np.abs(solution.y)
    x_true = np.abs(solution_true.y)

    _, axs = plt.subplots(tight_layout = True)
    axs.plot(np.exp(t_train),y_train[:,:], '-', linewidth = 5.0,
             alpha = 0.25, markeredgecolor='none')
    axs.set_prop_cycle(None)
    axs.plot(solution.t, x_solution.T/np.max(x_solution, axis = 1),
             '-.', linewidth = 2.0)
    axs.set_prop_cycle(None)
    axs.plot(solution_true.t, x_true.T/np.max(x_true, axis = 1),
             linestyle=':', linewidth = 6.0, alpha = 0.65)

    axs.set_xscale(xscale)
    axs.set_xlabel('time $(s)$')
    axs.set_ylabel('scaled concentration $(unitless)$')

    leg = axs.legend(species, ncol = 1, frameon = False, loc = [0, 0.4])
    for lh in leg.legendHandles:
        lh.set_alpha(1)

    if save_path is not None:
        plt.savefig(save_path)

    return solution, solution_true


def save_param(params, path):
    '''
    Save the parameters to a npz file
    '''
    nn_params_inv, model_params_inv = params
    np.savez(path, data = {'nn_params' :nn_params_inv, 'model_params': model_params_inv})
