import os, sys 
from ase.io import read, write
import numpy as np
from deepmd.calculator import DP
from deepmd.infer import calc_model_devi
from deepmd.infer import DeepPot
from deepmd.infer.model_devi import calc_model_devi_v, calc_model_devi_f
import glob
import json
import requests


def MakeBoxRaw (path = None):
    """
    path = path to the folder containing the training/validation dp.traj
    """
    file = open(path + "/training/box.raw","w")
    allatoms = read(path + "training/dp.traj", index= ":")
    for atoms in allatoms:
        cell = atoms.get_cell()
        cell = cell.flatten()
        
        file.write(str(cell) + "\n")
    file.close()
    
    file = open(path + "/validation/box.raw","w")
    allatoms = read(path + "validation/dp.traj", index= ":")
    for atoms in allatoms:
        cell = atoms.get_cell()
        cell = cell.flatten()
        
        file.write(str(cell) + "\n")
    file.close()
    

def MakeCoordRaw (path = None):
    """
    path = path to the folder containing the training/validation dp.traj
    """
    file = open(path + "/training/coord.raw","w")
    allatoms = read(path + "/training/dp.traj", index = ":")
    for atoms in allatoms:
        positions = atoms.get_positions()
        positions = positions.flatten()
        
        file.write(str(positions) + "\n")
    file.close()
    
    file = open(path + "/validation/coord.raw","w")
    allatoms = read(path + "/validation/dp.traj", index = ":")
    for atoms in allatoms:
        positions = atoms.get_positions()
        positions = positions.flatten()
        
        file.write(str(positions) + "\n")
    file.close()

def MakeEnergyRaw (path = None, apply_constraint = False):
    """
    path = path to the folder containing the training/validation dp.traj
    apply_constraint = True if the constraint is to be applied
    """
    file = open(path + "/training/energy.raw","w")
    allatoms = read(path + "/training/dp.traj", index = ":")
    for atoms in allatoms:
        file.write(str(atoms.get_potential_energy(apply_constraint = apply_constraint)) + "\n")   
    file.close()

    file = open(path + "/validation/energy.raw","w")
    allatoms = read(path + "/validation/dp.traj", index = ":")
    for atoms in allatoms:
        file.write(str(atoms.get_potential_energy(apply_constraint = apply_constraint)) + "\n")      
    file.close()

def MakeForceRaw (path = None,apply_constraint = False):
    """
    path = path to the folder containing the training/validation dp.traj
    apply_constraint = True if the constraint is to be applied
    """
    file = open(path + "/training/force.raw","w")
    allatoms = read(path + "/training/dp.traj", index = ":")
    for atoms in allatoms:
        forces = atoms.get_forces(apply_constraint = apply_constraint)
        forces = forces.flatten()
        file.write(str(forces) + "\n")
    file.close()
    
    file = open(path + "/validation/force.raw","w")
    allatoms = read(path + "/validation/dp.traj", index = ":")
    for atoms in allatoms:
        forces = atoms.get_forces(apply_constraint = apply_constraint)
        forces = forces.flatten()
        file.write(str(forces) + "\n")
    file.close()


def MakeTypeRaw(path = None,type_map = None):
    """
    path = path to the folder containing the training/validation dp.traj
    type_map = chemical symbol type dictionary (ex:{'O': 0, 'Si': 1, 'Au': 2})"""
    atoms = read(path+ "/training/dp.traj", index=0)
    file = open(path+"/training/type.raw","w")
    num_list = []
    for atom in atoms:
        num_list.append(type_map[atom.symbol])

    for n in num_list:
        file.write(str(n) + " ") # need to make the numbers spaced
    file.close()
    
    atoms = read(path+ "/validation/dp.traj", index=0)
    file = open(path + "/validation/type.raw","w")
    num_list = []
    for atom in atoms:
        num_list.append(type_map[atom.symbol])

    for n in num_list:
        file.write(str(n) + " ") # need to make the numbers spaced
    file.close()

def MakeVirialRaw (path = None):
    """
    path = path to the folder containing the training/validation dp.traj
    """
    file = open(path + "/training/virial.raw","w")
    allatoms = read(path + "/training/dp.traj", index = ":")
    for atoms in allatoms:
        virial = atoms.get_stress(voigt = False)
        virial = virial.flatten()
        file.write(str(virial) + "\n")  
    file.close()
    
    file = open(path + "/validation/virial.raw","w")
    allatoms = read(path + "/validation/dp.traj", index = ":")
    for atoms in allatoms:
        virial = atoms.get_stress(voigt = False)
        virial = virial.flatten()
        file.write(str(virial) + "\n")  
    file.close()
    

def write_dptraj(paths = None, train_split = 0.8, valid_split = 0.1):
    """
    paths = the list of paths to the data that will be split into train/valid/test
    train_split = the decimal value amount of the data that will be used for training
    valid_split = the decimal value amount of the data that will be used for validation
    """
    for path in paths:
        allatoms = read(path, index = ":")
        len_allatoms = len(allatoms)
        train_len = int(np.ceil(train_split*len_allatoms))
        valid_len = int(np.ceil(valid_split*len_allatoms))
        test_len = int(len_allatoms - train - valid)
        
        train = read(path, index = ":"+str(train_len))
        valid = read(path, index = str(train_len) + ":" + str(train_len+valid_len))
        test = read(path, index = str(train_len+valid_len)+":")

        write(path+"/training/dp.traj", train)
        write(path+"/validation/dp.traj", valid)
        write(path+"/testing/dp.traj", test)

def make_sets(paths = None, train_set_size = None, valid_set_size = None):
    """
     paths = the list of paths to the folder that has the training and validation folders
     train_set_size = the number of images to be placed inside of of the set.00* for training
     valid_set_size = the number of images to be placed inside of of the set.00* for validation
    """
    for path in paths:
        os.chdir(working_dir + "/" + path + "/training")
        os.system("source raw_to_set.sh " + str(train_set_size)
        os.chdir(working_dir)

        os.chdir(working_dir + "/" + path + "/validation")
        os.system("source raw_to_set.sh " + str(valid_set_size))
        os.chdir(working_dir)

def make_raw(paths = None, type_map = None):
    """
    paths = the paths to the data the will be used to generate the .raw files
    type_map = chemical symbol type dictionary (ex:{'O': 0, 'Si': 1, 'Au': 2})
    """
    for p in paths:
        MakeBoxRaw (path = p)
        MakeCoordRaw (path = p)
        MakeEnergyRaw (path = p, apply_constraint = False)
        MakeForceRaw (path = p, apply_constraint = False)
        MakeTypeRaw(path = p, type_map = type_map)
        MakeVirialRaw (path = p)


# create the .json  input file
def make_inJSON(type_map = None, model_folder = None,
                descriptor_type = 'se_e2_a',rcut_smth = 4.80, rcut = 5.00, sel = [10,10,10], neuron  = [32,32,32,32],
                fitting_net_neuron = [32,32,32], 
                decay_steps = 5000, start_lr = 0.005, stop_lr = 3.51e-8, 
                start_pref_e = 0.02, limit_pref_e  = 8, start_pref_f = 1000, limit_pref_f  = 1, 
                start_pref_v = 0, limit_pref_v  = 0,
                training_systems = None, validation_systems = None,
                numb_steps = 500000):
    """
    type_map = list that contains the atoms contained in the structure
    model_folder = where to save in.json
    
    the rest of the input are the inputs to the parameters required inside the in.json
    """
    train_systems_to_use = []
    valid_systems_to_use = []
    for system_t, system_v in zip(training_systems, validation_systems):
        train_systems_to_use.append(system_t)
        valid_systems_to_use.append(system_v)

        
    url = 'https://raw.githubusercontent.com/deepmodeling/deepmd-kit/master/examples/water/se_e3/input.json'
    r = requests.get(url, allow_redirects=True)
    input_json = json.loads(r.content) # i think this is causing the problem (most likely the example doesnt have learning_rate in its dictionary)
    
    input_json['model']['type_map'] = type_map
    
    input_json['model']['descriptor']['type'] = descriptor_type
    input_json['model']['descriptor']['sel'] = sel
    input_json['model']['descriptor']['rcut_smth'] = rcut_smth
    input_json['model']['descriptor']['rcut'] = rcut
    input_json['model']['descriptor']['neuron'] = neuron
    input_json['model']['descriptor']['resnet_dt'] = False
    input_json['model']['descriptor']['seed'] = np.random.randint(99999)
    
    input_json['model']['fitting_net']['neuron'] = fitting_net_neuron
    input_json['model']['fitting_net']['resnet_dt'] = True
    input_json['model']['fitting_net']['seed'] = np.random.randint(99999)
    
    input_json['learning_rate']['type'] = 'exp'
    input_json['learning_rate']['decay_steps'] = decay_steps
    input_json['learning_rate']['start_lr'] = start_lr
    input_json['learning_rate']['stop_lr'] = stop_lr
    
    input_json['loss']['start_pref_e'] = start_pref_e
    input_json['loss']['limit_pref_e'] = limit_pref_e
    input_json['loss']['start_pref_f'] = start_pref_f
    input_json['loss']['limit_pref_f'] = limit_pref_f
    input_json['loss']['start_pref_v'] = start_pref_v
    input_json['loss']['limit_pref_v'] = limit_pref_v
    
    input_json['training']['training_data']['systems'] = train_systems_to_use
    input_json['training']['training_data']['batch_size'] = 'auto'
    
    input_json['training']['validation_data']['systems'] = valid_systems_to_use
    input_json['training']['validation_data']['batch_size'] = 1
    input_json['training']['validation_data']['numb_btch'] = 30
    
    input_json['training']['numb_steps'] = numb_steps
    input_json['training']['seed'] = np.random.randint(99999)
    input_json['training']['disp_file'] = 'lcurve.out'
    input_json['training']['disp_freq'] = 1000
    input_json['training']['numb_test'] = 4
    input_json['training']['save_freq'] = 100000
    input_json['training']['save_ckpt'] = 'model.ckpt'
    input_json['training']['disp_training'] = True
    input_json['training']['time_training'] = True
    input_json['training']['profiling'] = False
    input_json['training']['profiling_file'] = 'timeline.json'
    
    with open('in.json', 'w') as file:
        file.write(json.dumps(input_json, indent=2))
        
    os.system("mv in.json ./" + model_folder)
