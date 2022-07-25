import os, sys 
sys.path.append("./../")
from helper.dp_validation import get_dev
from ase.io import read, write, Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution as MBD
from ase.optimize import BFGS
from ase import units
from ase.db import connect
import datetime
import numpy as np
from deepmd.calculator import DP
from deepmd.infer import DeepPot
from deepmd.infer import calc_model_devi



def DPMD(use_db = None, database = None ,idx = None, initial_traj = None, initial_traj_idx = None,\
         metal = None, n_metal = None, zeo = None, method = "DFT_SPE", iteration = None, DFT_status = "Complete",\
         model = None, graphs = None, temperature_K = 300, time_step = 0.1, temperature_dyn = 300, friction_coeff = 0.002,
         interval = 10, MD_steps = 30000, type_map = None):
    """
    use_db = tells whether to use the database for grabbing atoms or not. 
    (use_db = True means use the database, use_db = False means use an initial traj)
    (either define a file for an initial traj, or call the Atoms object from the database) 
    (if both are supplied, the inital_traj is used)
    database = the database the houses all of the trajectories
    idx = the index that we are going to draw from for an inital trajectory (from the database)
    inital_traj = the initial trajectory to draw from 
    initial_traj_idx = the index of the initial trajectory file to use as the initial for the MD
    
    n_metal = the number of metal atoms inside the zeolite to be used in our MD
    zeo = the zeolite to be used in our MD
    method = the method at which the initial trajectory was generated from
    iteration = the iteration number that the input trajectory was made in
    model = the path to the model file that you want to use for the MD
    temperature_K = the temepratue in Kelvins that you want to base your initial Maxwell Boltzmann velocity Distribution on
    time_step = the time step in fs that you want the MD to take each step
    temperature_dyn = the temeprature in kB you want the dynamics to be at
    friction_coeff = the friction coefficent in atomic units that you want in the dynamics
    interval = the number of MD steps between each saved trajectory to the growing traj_path file
    MD_steps = the total number of steps in the MD run
    """
    file = open("./time_spent.txt", "a")
    
    begin_time = datetime.datetime.now()
    file.write("begin_time = " + str(begin_time) + "\n")
    
    file.write("metal = " + str(metal) + "\n")
    file.write("n_metal = " + str(n_metal) + "\n")
    file.write("zeo = " + str(zeo) + "\n")
    file.write("model = " + str(model) + "\n")
    file.write("temperature_K = " + str(temperature_K) + "\n")
    file.write("time_step = " + str(time_step) + "\n")
    file.write("temperature_dyn = " + str(temperature_dyn) + "\n")
    file.write("fricition_coeff = " + str(friction_coeff) + "\n")
    file.write("interval = " + str(interval) + "\n")
    file.write("MD_steps = " + str(MD_steps) + "\n")
    
    if use_db == True:
        db = connect(database)
        atoms = db.get_atoms(metal = metal, idx = idx, n_metal = n_metal, zeo = zeo, method = method, iteration = iteration, \
                         DFT_status = DFT_status)
    
    if use_db == False:
        atoms = read(initial_traj, index = initial_traj_idx)
     
    atoms.calc = DP(model = model)

    MBD(atoms, temperature_K = temperature_K)

    # now run using constant energy using the Langevin algorithm
    dyn = Langevin(atoms, time_step*units.fs, temperature_dyn*units.kB , friction_coeff)

    traj = Trajectory("./DPMD.traj", 'w', atoms, properties = ("energy","forces"))
    dyn.attach(traj.write, interval=interval) 
    dyn.run(MD_steps)
    
    end_time = datetime.datetime.now() - begin_time
    file.write("end_time = " + str(end_time))
    file.close()
    
    type_map = type_map
    atoms = read("./DPMD.traj", index = ":")
    dev = get_dev(atoms = atoms,type_map = type_map, graphs = graphs)
    np.save("dev.npy", dev)