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


def MakeBoxRaw (path = None, num_config_train = None, num_config_valid = None):
    """needs the path to the database and the number of trajectories"""
    file = open(path + "/training/box.raw","w")
    for config in range(num_config_train):
    
        atoms = read(path+"/training/dp.traj", index = config)

        cell = atoms.get_cell()
        cell = np.reshape(cell,-1)

        str_cell = []
        str_cell.append(str(cell[0]))
        for i in range(1,len(cell)):
            str_cell.append(" " +  str(cell[i]))
        str_cell.append(" \n")

        for i in range(len(str_cell)): # this actually writes the file
            file.write(str_cell[i])
        
    file.close()
    
    file = open(path + "/validation/box.raw","w")
    for config in range(num_config_valid):
    
        atoms = read(path + "/validation/dp.traj", index = config)

        cell = atoms.get_cell()
        cell = np.reshape(cell,-1)

        str_cell = []
        str_cell.append(str(cell[0]))
        for i in range(1,len(cell)):
            str_cell.append(" " +  str(cell[i]))
        str_cell.append(" \n")

        for i in range(len(str_cell)): # this actually writes the file
            file.write(str_cell[i])
        
    file.close()

def MakeCoordRaw (path = None,num_config_train = None, num_config_valid = None):
    """requires a path to the atoms database, and the number of trajectories inside the database"""
    file = open(path + "/training/coord.raw","w")
    for config in range(num_config_train): # makes a line for each trajectory
        atoms = read(path + "/training/dp.traj", index = config)

        positions = atoms.get_positions()
        positions = np.reshape(positions, -1)

        str_positions = []
        str_positions.append(str(positions[0]))
        for i in range(1,len(positions)):
            str_positions.append(" " +  str(positions[i]))
        str_positions.append(" \n")

        for i in range(len(str_positions)): # this actually writes the file
            file.write(str_positions[i])

    file.close()
    
    
    file = open(path + "/validation/coord.raw","w")
    for config in range(num_config_valid): # makes a line for each trajectory
        atoms = read(path + "/validation/dp.traj", index = config)

        positions = atoms.get_positions()
        positions = np.reshape(positions, -1)

        str_positions = []
        str_positions.append(str(positions[0]))
        for i in range(1,len(positions)):
            str_positions.append(" " +  str(positions[i]))
        str_positions.append(" \n")

        for i in range(len(str_positions)): # this actually writes the file
            file.write(str_positions[i])

    file.close()

def MakeEnergyRaw (path = None,num_config_train = None, num_config_valid = None, apply_constraint = False):
    file = open(path + "/training/energy.raw","w")
    
    for config in range(num_config_train):
        atoms = read(path +"/training/dp.traj", index = config)
        
        energy = atoms.get_potential_energy(apply_constraint = apply_constraint)
        str_energy = str(energy)
        
        file.write(str_energy + "\n")
        
    file.close()

    file = open(path + "/validation/energy.raw","w")
    
    for config in range(num_config_valid):
        atoms = read(path +"/validation/dp.traj", index = config)
        
        energy = atoms.get_potential_energy(apply_constraint = apply_constraint)
        str_energy = str(energy)
        
        file.write(str_energy + "\n")
        
    file.close()

def MakeForceRaw (path = None,num_config_train = None, num_config_valid = None, apply_constraint = False):
    """requires a path to the atoms database, and the number of trajectories inside the database"""
    file = open(path + "/training/force.raw","w")
    for config in range(num_config_train): # makes a line for each trajectory
        atoms = read(path + "/training/dp.traj", index = config)

        forces = atoms.get_forces(apply_constraint = apply_constraint)
        forces = np.reshape(forces, -1)

        str_forces = []
        str_forces.append(str(forces[0]))
        for i in range(1,len(forces)):
            str_forces.append(" " +  str(forces[i]))
        str_forces.append(" \n")

        for i in range(len(str_forces)): # this actually writes the file
            file.write(str_forces[i])

    file.close()
    
    file = open(path + "/validation/force.raw","w")
    for config in range(num_config_valid): # makes a line for each trajectory
        atoms = read(path + "/validation/dp.traj", index = config)

        forces = atoms.get_forces(apply_constraint = apply_constraint)
        forces = np.reshape(forces, -1)

        str_forces = []
        str_forces.append(str(forces[0]))
        for i in range(1,len(forces)):
            str_forces.append(" " +  str(forces[i]))
        str_forces.append(" \n")

        for i in range(len(str_forces)): # this actually writes the file
            file.write(str_forces[i])

    file.close()


def MakeTypeRaw(path = None,type_map = None):
    """#right no this code only works when all atoms of the same chemical formula are of the same type
    type_raw = {'O': 0, 'Si': 1, 'Au': 2} for example"""
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

def MakeVirialRaw (path = None, num_config_train = None, num_config_valid = None):
    """needs the path to the database and the number of trajectories"""
    file = open(path + "/training/virial.raw","w")
    for config in range(num_config_train):
    
        atoms = read(path+"/training/dp.traj", index = config)

        virial = atoms.get_stress(voigt = False)
        virial = np.reshape(virial,-1)

        str_virial = []
        str_virial.append(str(virial[0]))
        for i in range(1,len(virial)):
            str_virial.append(" " +  str(virial[i]))
        str_virial.append(" \n")

        for i in range(len(str_virial)): # this actually writes the file
            file.write(str_virial[i])
        
    file.close()
    
    file = open(path + "/validation/virial.raw","w")
    for config in range(num_config_valid):
    
        atoms = read(path+"/validation/dp.traj", index = config)

        virial = atoms.get_stress(voigt = False)
        virial = np.reshape(virial,-1)

        str_virial = []
        str_virial.append(str(virial[0]))
        for i in range(1,len(virial)):
            str_virial.append(" " +  str(virial[i]))
        str_virial.append(" \n")

        for i in range(len(str_virial)): # this actually writes the file
            file.write(str_virial[i])
        
    file.close()
    

def write_dptraj():
    for path, nt,nv in zip(paths, num_config_train, num_config_valid):
        train = read(path+"/combined.traj", index = ":"+str(nt))
        valid = read(path+"/combined.traj", index = str(nt) + ":" + str(nt+nv))
        test = read(path+"/combined.traj", index = str(nv+nt)+":")

        write(path+"/training/dp.traj", train)
        write(path+"/validation/dp.traj", valid)
        write(path+"/testing/dp.traj", test)

def make_sets():
    for path in paths:
        os.chdir(working_dir + "/" + path + "/training")
        os.system("source raw_to_set.sh 75")
        os.chdir(working_dir)

        os.chdir(working_dir + "/" + path + "/validation")
        os.system("source raw_to_set.sh 75")
        os.chdir(working_dir)

def make_raw():
    for p,nt,nv in zip(paths,num_config_train,num_config_valid):
        MakeBoxRaw (path = p, num_config_train = nt, num_config_valid = nv)
        MakeCoordRaw (path = p,num_config_train = nt, num_config_valid = nv)
        MakeEnergyRaw (path = p,num_config_train = nt, num_config_valid = nv, apply_constraint = False)
        MakeForceRaw (path = p,num_config_train = nt, num_config_valid = nv, apply_constraint = False)
        MakeTypeRaw(path = p)
        MakeVirialRaw (path = p, num_config_train = nt, num_config_valid = nv)


# create the .json  input file
def make_inJSON(type_map = None,
                rcut_smth = 4.80, rcut = 5.00, sel = [10,10,10], neuron  = [32,32,32,32],
                fitting_net_neuron = [32,32,32], \
                decay_steps = 5000, start_lr = 0.005, stop_lr = 3.51e-8, \
                start_pref_e = 0.02, limit_pref_e  = 8, start_pref_f = 1000, limit_pref_f  = 1, \
                start_pref_v = 0, limit_pref_v  = 0,\
                training_systems = None, validation_systems = None,\
                numb_steps = 500000, model_folder = None):
    """
    paths = paths to the trajecotries to train (must put in form ["path_A", "path_B", ...])
    save_paths = path to save the in.json to
    type_map = list that contains the atoms contained in the structure
    
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
    
    input_json['model']['descriptor']['type'] = 'se_e2_a'
    input_json['model']['descriptor']['sel'] = sel
    input_json['model']['descriptor']['rcut_smth'] = rcut_smth
    input_json['model']['descriptor']['rcut'] = rcut
    input_json['model']['descriptor']['neuron'] = neuron
    input_json['model']['descriptor']['resnet_dt'] = False
    input_json['model']['descriptor']['seed'] =np.random.randint(99999)
    
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
