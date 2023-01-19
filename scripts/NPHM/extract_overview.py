from pathlib import Path

import pyvista as pv
from elias.util import ensure_directory_exists
from tqdm import tqdm

if __name__ == '__main__':

    scans_folder = "//wsl.localhost/Ubuntu/mnt/rohan/cluster/daidalos/sgiebenhain/figure_dataset/"
    file_name_format_1 = "*/*/*/target.ply"
    file_name_format_2 = "*/*/target.ply"
    output_folder = "D:/Projects/NPHM/data/overview"
    max_expression_id = 23

    ensure_directory_exists(output_folder)

    scan_paths_1 = Path(scans_folder).glob(file_name_format_1)
    scan_paths_2 = Path(scans_folder).glob(file_name_format_2)

    scans_metadata = []

    for scan_path in tqdm(scan_paths_1):
        # format: id85_3/bullshit/target.ply
        expression_id = scan_path.parent.parent.name.split('_')[-1]
        identity = scan_path.parent.parent.parent.name

        scans_metadata.append({"scan_path": str(scan_path), "expression_id": expression_id, "identity": identity})

    for scan_path in tqdm(scan_paths_2):
        # format: expression_3/target.ply
        expression_id = scan_path.parent.name.split('_')[-1]
        identity = scan_path.parent.parent.name

        scans_metadata.append({"scan_path": str(scan_path), "expression_id": expression_id, "identity": identity})

    for scan_metadata in tqdm(scans_metadata):
        scan_path = scan_metadata["scan_path"]
        expression_id = scan_metadata["expression_id"]
        identity = scan_metadata["identity"]

        mesh = pv.read(scan_path)
        p = pv.Plotter(off_screen=True)
        p.add_mesh(mesh)
        p.camera_position = 'xy'

        rendered_img = p.screenshot(filename=f"{output_folder}/{identity}_{expression_id}")

    identities = set()
    for rendered_file in Path(output_folder).iterdir():
        identity = '_'.join(rendered_file.name.split('_')[:-1])
        identities.add(identity)

    for identity in identities:
        for expression_id in range(max_expression_id):
            if not Path(f"{output_folder}/{identity}_{expression_id}.png").exists() and not Path(f"{output_folder}/{identity}_{expression_id}.txt").exists():
                open(f"{output_folder}/{identity}_{expression_id}.txt", mode='w')

