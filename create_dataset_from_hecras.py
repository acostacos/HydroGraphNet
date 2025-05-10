import os
import numpy as np

from transform_helper_files.hecras_data_retrieval import get_cell_area, get_water_level, get_roughness, get_rainfall
from transform_helper_files.shp_data_retrieval import get_cell_position, get_cell_elevation

def get_cell_slope():
    cell_slope_np_path = ''
    return np.load(cell_slope_np_path)

def get_cell_aspect():
    cell_aspect_np_path = ''
    return np.load(cell_aspect_np_path)

def get_cell_curvature():
    cell_curvature_np_path = ''
    return np.load(cell_curvature_np_path)

def get_cell_infiltration(num_nodes: int):
    return np.zeros(num_nodes)

def get_cell_flow_accumulation():
    cell_flow_accumulation_np_path = ''
    return np.load(cell_flow_accumulation_np_path)

def create_hydrograph_id_files(event_keys: list[str], dataset_folder: str):
    for event_key in event_keys:
        train_event_keys = [k for k in event_keys if k != event_key]
        train_filename = f'train_for_{event_key}.txt'
        train_path = os.path.join(dataset_folder, train_filename)
        with open(train_path, 'w') as f:
            for train_key in train_event_keys:
                f.write(f"{train_key}")

        test_filename = f'test_for_{event_key}.txt'
        test_path = os.path.join(dataset_folder, test_filename)
        with open(test_path, 'w') as f:
            f.write(f"{event_key}")

def create_constant_text_files(hec_ras_filepath: str, node_shp_filepath: str, dem_path: str, dataset_folder: str, prefix: str):
    pos = get_cell_position(node_shp_filepath)
    pos_path = os.path.join(dataset_folder, f"{prefix}_XY.txt")
    np.savetxt(pos_path, pos)

    area = get_cell_area(hec_ras_filepath)
    area_path = os.path.join(dataset_folder, f"{prefix}_CA.txt")
    np.savetxt(area_path, area)

    elevation = get_cell_elevation(node_shp_filepath)
    elevation_path = os.path.join(dataset_folder, f"{prefix}_CE.txt")
    np.savetxt(elevation_path, elevation)

    manning = get_roughness(hec_ras_filepath)
    manning_path = os.path.join(dataset_folder, f"{prefix}_N.txt")
    np.savetxt(manning_path, manning)

    slope = get_cell_slope()
    slope_path = os.path.join(dataset_folder, f"{prefix}_CS.txt")
    np.savetxt(slope_path, slope)

    aspect = get_cell_aspect()
    aspect_path = os.path.join(dataset_folder, f"{prefix}_A.txt")
    np.savetxt(aspect_path, aspect)

    curvature = get_cell_curvature()
    curvature_path = os.path.join(dataset_folder, f"{prefix}_CU.txt")
    np.savetxt(curvature_path, curvature)

    infiltration = get_cell_infiltration(len(pos))
    infiltration_path = os.path.join(dataset_folder, f"{prefix}_IP.txt")
    np.savetxt(infiltration_path, infiltration)

    flow_accum = get_cell_flow_accumulation()
    flow_accum_path = os.path.join(dataset_folder, f"{prefix}_FA.txt")
    np.savetxt(flow_accum_path, flow_accum)

def create_dynamic_text_files(hec_ras_filepath: str, node_shp_filepath: str, dataset_folder: str, prefix: str, hydrograph_id: str):
    water_level = get_water_level(hec_ras_filepath)
    elevation = get_cell_elevation(node_shp_filepath)
    water_depth = np.clip(water_level - elevation, a_min=0)
    assert np.all(water_depth > 0)
    water_depth_path = os.path.join(dataset_folder, f"{prefix}_WD_{hydrograph_id}.txt")
    np.savetxt(water_depth_path, water_depth)

    precipitation = get_rainfall(hec_ras_filepath)
    precipitation_path = os.path.join(dataset_folder, f"{prefix}_Pr_{hydrograph_id}.txt")
    np.savetxt(precipitation_path, precipitation)

    # TODO: How to get inflow and volume?
    # inflow_path = os.path.join(folder, f"{prefix}_US_InF_{hydrograph_id}.txt")
    # volume_path = os.path.join(folder, f"{prefix}_V_{hydrograph_id}.txt")

def main():
    prefix = "init"
    base_dataset_floder = f"outputs_phy/hecras_data/{prefix}"
    node_shp_path = "C:\\Users\\Carlo\\Documents\\School\\Masters\\NUS\\gnn_flood_modeling\\data\\datasets\\init\\initp01\\raw\\cell_centers.shp"
    dem_path = "C:\\Users\\Carlo\\Documents\\School\\Masters\\NUS\\Dissertation\\Data\\Wollombi_Model_LR\\DEM_5m_EPSG28356_ND999_Wollombi.tif"
    hec_ras_file_paths = {
        'p01': "C:\\Users\\Carlo\\Documents\\School\\Masters\\NUS\\gnn_flood_modeling\\data\\datasets\\init\\initp01\\raw\\Model_01.p01.hdf",
        'p02': "C:\\Users\\Carlo\\Documents\\School\\Masters\\NUS\\gnn_flood_modeling\\data\\datasets\\init\\initp02\\raw\\Model_01.p02.hdf",
        'p03': "C:\\Users\\Carlo\\Documents\\School\\Masters\\NUS\\gnn_flood_modeling\\data\\datasets\\init\\initp03\\raw\\Model_01.p03.hdf",
        'p04': "C:\\Users\\Carlo\\Documents\\School\\Masters\\NUS\\gnn_flood_modeling\\data\\datasets\\init\\initp04\\raw\\Model_01.p04.hdf",
    }

    # Create dataset folder
    if not os.path.exists(base_dataset_floder):
        os.makedirs(base_dataset_floder)
    print(f"Saving files in folder: {base_dataset_floder}", flush=True)

    print(f"Creating hydrograph id files...", flush=True)
    event_keys = list(hec_ras_file_paths.keys())
    create_hydrograph_id_files(event_keys, base_dataset_floder)

    # Create static text files; Can use arbitrary hec_ras file as they are the same
    print(f"Saving constant features...", flush=True)
    sample_hec_ras_path = next(iter(hec_ras_file_paths.values()))
    create_constant_text_files(sample_hec_ras_path, node_shp_path, dem_path, base_dataset_floder, prefix)

    # Create dynamic text files
    for event_key, hec_ras_file_path in hec_ras_file_paths.items():
        print(f"Saving dynamic features for event {event_key}...", flush=True)
        create_dynamic_text_files(hec_ras_file_path, node_shp_path, base_dataset_floder, prefix, hydrograph_id=event_key)

if __name__ == "__main__":
    main()
