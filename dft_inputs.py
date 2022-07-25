import os, sys 
sys.path.append("/global/u2/r/rk7837kl/lib/")
from kul_tools import KulTools as KT
from ase.io import read
from ase.db import connect


def set_up_encut_calcs(input_file = None, atom_type = None, calc_type = 'opt', encuts = None, vasp_parameters = None, cell_opt = False):
    """
    input_file = The file that contains the system that will be used to test the encut
    atom_type = The structure type that kT uses to add certain defauls to the vasp parameters (for kT)
    calc_type = The type of DFT calculation (for kT)
    encuts = The list of potential encuts
    vasp_parameters = the set of default vasp parameters (see below for an example)
    cell_opt = tells if the system will be doing a unit cell optimization (if True then the encuts will be multiplied by 1.5 for the
    calculation)
    
    good defaults for vasp_parameters = {'xc':'pbe', 'nsw':50000, 'isif':3,'kpts':(2,2,2),'ibrion':2,'ediff':0.0002}"""
    len_encuts = len(encuts)
    str_len_encuts = str(len_encuts)
    characters = len(str_len_encuts)
    
    vasp_parameters_copy = vasp_parameters.copy()
    #set up the folder
    for i,e in enumerate(encuts):
        os.mkdir("./" + str(i).zfill(int(characters+1))+"_{}".format(e))
        os.system("cp vasprun.xml ./" + str(i).zfill(int(characters+1))+"_{}".format(e))
        os.system("cp submit_geoopt.sh ./" + str(i).zfill(int(characters+1))+"_{}".format(e))
        
        if cell_opt == True:
            vasp_parameters_copy['encut'] = 1.5*e
            
        if cell_opt == False:
            vasp_parameters_copy['encut'] = e

    # write the file
        file = open("encut_check.py","w")
        file.write("from ase import io \n")
        file.write("import os, sys \n")
        file.write("sys.path.append('/global/u2/r/rk7837kl/lib/') \n")
        file.write("from kul_tools import KulTools as KT \n")
        file.write("kt = KT(gamma_only=False,structure_type='"+str(atom_type)+"') \n")
        file.write("kt.set_calculation_type("+str(calc_type)+") \n")
        file.write("atoms = read('" + input_file + "') \n")
        file.write("atoms.pbc=True \n")
        file.write("kt.set_structure(atoms) \n")
        file.write("kt.set_overall_vasp_params("+str(vasp_parameters_copy)+") \n")
        file.write("kt.run() \n")
        file.close()
        os.system("mv encut_check.py ./" + str(i).zfill(int(characters+1))+"_{}".format(e))
    
def set_up_kpts_calcs(input_file = None, atom_type = None, calc_type = 'opt', kpts = None, vasp_parameters = None):
    """
    input_file = The file that contains the system that will be used to test the encut
    atom_type = The structure type that kT uses to add certain defauls to the vasp parameters (for kT)
    calc_type = The type of DFT calculation (for kT)
    kpts = The list of potential k-points
    vasp_parameters = the set of default vasp parameters (see below for an example)
    
    good defaults for vasp_parameters = {'xc':'pbe','encut':400, 'nsw':0, 'ismear':1,'sigma':0.2} """
    len_kpts = len(kpts)
    str_len_kpts = str(len_kpts)
    characters = len(str_len_kpts)
    
    vasp_parameters_copy = vasp_parameters.copy()
    #set up the folder
    for i,k in enumerate(kpts):
        os.mkdir("./" + str(i).zfill(int(characters+1))+"_{}".format(k))
        os.system("cp" + input_file + "./" + str(i).zfill(int(characters+1))+"_{}".format(k))
        os.system("cp submit_DFTSPE.sh ./" + str(i).zfill(int(characters+1))+"_{}".format(k))
        
        vasp_parameters_copy['kpts'] = (k,k,k)
        
        # write the file
        file = open("kpts_check.py","w")
        file.write("from ase import io \n")
        file.write("import os, sys \n")
        file.write("sys.path.append('/global/u2/r/rk7837kl/lib/') \n")
        file.write("from kul_tools import KulTools as KT \n")
        file.write("kt = KT(gamma_only=False,structure_type='"+str(atom_type)+"') \n")
        file.write("kt.set_calculation_type("+str(calc_type)+") \n")
        file.write("atoms = read('" + input_file + "') \n")
        file.write("atoms.pbc=True \n")
        file.write("kt.set_structure(atoms) \n")
        file.write("kt.set_overall_vasp_params("+str(vasp_parameters_copy)+") \n")
        file.write("kt.run() \n")
        file.close()
        os.system("mv kpts_check.py ./" + str(i).zfill(int(characters+1))+"_{}".format(k))


def run_SinglePointEnergy(path =  None,vasp_parameters = {'gga':'RP',"encut":400, "nsw":0},atom_type = None):    
    """
    (all paths are with respect to the job folder that the code will be run in)
    path = the path to the atoms to have DFT run on it 
    vasp_parameters = the vasp parameters to be used in the calculation
    atom_type = the type of atoms used in the calcualtion for kT to add other defaults into vasp parameters (for kT)
    """
    kt = KT(gamma_only=False,structure_type=str(atom_type))
    kt.set_calculation_type('opt')
    atoms = read(path)
    
    atoms.pbc=True
    kt.set_structure(atoms)
    kt.set_overall_vasp_params(vasp_parameters)
    kt.run()


    

def run_VASPMD(path = None, atom_type = None, vasp_parameters = {"nsw":100000, "encut":400, "ediff":1e-5, "xc":'pbe', "ibrion":0, "potim":0.5, "tebeg":298, "isif":2,"smass":0}): 
    """
    path = the path to the Atoms object that will be the starting point for the VASP MD
    atom_type = the type of atoms used in the calcualtion for kT to add other defaults into vasp parameters (for kT)
    vasp_parameters = the vasp parameters to be used in the calculation
    
    """
    kt = KT(gamma_only=False,structure_type=str(atoms_type))
    kt.set_calculation_type('md')
    atoms = read(path)
    atoms.pbc=True
    kt.set_structure(atoms)
    kt.set_overall_vasp_params(vasp_parameters)
    kt.run()