import os
import numpy as np
from typing import Tuple
from .light_lib import Brdf


def run_engine(brdf: Brdf, 
               model_file: str,
               svb: np.array,
               ovb: np.array,
               save_imgs: bool = False,
               output_dir: str = None,
               instance_count: int = 9) -> np.array:
    assert instance_count <= 25 and instance_count > 0, \
        'Engine runs with 0 < instance_count <= 25'
    assert svb.shape[1] == 3 and ovb.shape[1] == 3, \
        'Engine requires n x 3 numpy arrays as input for sun and observer vectors'
    assert model_file[4:] == '.obj', 'Model file must be *.obj'
    lce_dir = os.environ['LCEDIR']

    cwd = os.getcwd()
    if output_dir is None:
        output_dir = f'{cwd}/out'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    save_str = ""
    if save_imgs:
        save_str = "-s"

    brdf_ind = query_brdf_registry(brdf)
    results_file = "light_curve0.lcr"
    write_light_curve_command_file(svb, ovb)

    if brdf.cd and brdf.cs and brdf.n:
        brdf_opt_str = f'-b {brdf_ind} -D {brdf.cd} -S {brdf.cs} -N {brdf.n}'
    else:
        brdf_opt_str = f'-M -b {brdf_ind}'
    
    opts_str = (f'-m {model_file} -i {instance_count} -c {svb.shape[0]}'
                f' -r {results_file} -x {output_dir} {save_str} {brdf_opt_str}')
    run_str = f'{lce_dir}/LightCurveEngine {opts_str}'
    print(f'Running Light Curve Engine: \n{run_str}\n')
    os.chdir(lce_dir)
    os.system(run_str)
    os.chdir(cwd)
    with open(f'{lce_dir}/{results_file}', 'r') as f:
        lc_data = f.read().splitlines()
    return np.array([float(x) for x in lc_data])


def run_brdf_registry(arg: str) -> int:
    lce_dir = os.environ['LCEDIR']
    brdf_registry_path = f'{lce_dir}/BRDFRegistry'
    # Not sure why it's misinterpreting the output (hence // 256)
    return os.system(f'{brdf_registry_path} {arg}') // 256 


def print_all_registered_brdfs():
    run_brdf_registry('all')
    

def query_brdf_registry(brdf: Brdf) -> int:
    brdf_ind = run_brdf_registry(brdf.name)
    assert brdf_ind < 200, 'BRDF name not valid!'
    return brdf_ind

def write_light_curve_command_file(svb: np.array, 
                                   ovb: np.array, 
                                   command_file='light_curve0.lcc'):
    lce_dir = os.environ['LCEDIR']

    with open(f'{lce_dir}/{command_file}', 'w') as f:
        header = (f'Light Curve Command File\n'
            f'\nBegin header\n'
            f'{"Format".ljust(20)} {"SunXYZViewerXYZ".ljust(20)}\n'
            f'{"Reference Frame".ljust(20)} {"ObjectBody".ljust(20)}\n'
            f'{"Data Points".ljust(20)} {str(svb.shape[0]).ljust(20)}\n'
            f'End header\n\n')
        

        vf = lambda v: (f'{str(np.round(v[0],6)).ljust(10)} ' 
                        f'{str(np.round(v[1],6)).ljust(10)} ' 
                        f'{str(np.round(v[2],6)).ljust(10)}')

        data = f'Begin data\n' + ''.join(
            [f'{vf(s)}{vf(o)}\n' for s,o in zip(svb, ovb)]) + 'End data'

        f.write(header)
        f.write(data)
