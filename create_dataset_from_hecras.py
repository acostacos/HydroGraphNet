import os
import numpy as np
import yaml
import pandas as pd

from transform_helper_files.hecras_data_retrieval import get_cell_area, get_water_level, get_roughness, get_cumulative_rainfall, get_water_volume, get_face_flow
from transform_helper_files.shp_data_retrieval import get_cell_position, get_cell_elevation, get_edge_index
from transform_helper_files.dem_feature_extraction import get_filled_dem, get_slope, get_aspect, get_curvature, get_flow_accumulation

def get_info_from_config(config_file_path: str, root_dir: str) -> dict:
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)

    dataset_config = config['dataset_parameters']
    nodes_shp_path = os.path.join(root_dir, 'raw', dataset_config['nodes_shp_file'])
    edges_shp_path = os.path.join(root_dir, 'raw', dataset_config['edges_shp_file'])
    train_summary_path = os.path.join(root_dir, 'raw', dataset_config['training']['dataset_summary_file'])
    test_summary_path = os.path.join(root_dir, 'raw', dataset_config['testing']['dataset_summary_file'])
    inflow_boundary_nodes = dataset_config['inflow_boundary_nodes']

    return {
        'nodes_shp_path': nodes_shp_path,
        'edges_shp_path': edges_shp_path,
        'train_summary_path': train_summary_path,
        'test_summary_path': test_summary_path,
        'inflow_boundary_nodes': inflow_boundary_nodes,
    }

def get_hec_ras_paths_from_summary(summary_path: str, root_dir: str) -> dict:
    summary_df = pd.read_csv(summary_path)

    datasets = {}
    for _, row in summary_df.iterrows():
        run_id = row['Run_ID']
        hec_ras_path = row['HECRAS_Filepath']
        datasets[run_id] = os.path.join(root_dir, 'raw', hec_ras_path)
    return datasets

def create_hydrograph_id_file(run_ids: list[str], dataset_folder: str, filename: str):
    with open(os.path.join(dataset_folder, filename), 'w') as f:
        for i, run_id in enumerate(run_ids):
            postfix = "\n" if i < len(run_ids) - 1 else ""
            f.write(f"{run_id}{postfix}")

def get_cell_infiltration(num_nodes: int):
    return np.zeros(num_nodes)

def downsample_dynamic_data(dynamic_data: np.ndarray, step: int, aggr: str = 'first') -> np.ndarray:
    if step == 1:
        return dynamic_data

    # Trim array to be divisible by step
    trimmed_length = (dynamic_data.shape[0] // step) * step
    trimmed_array = dynamic_data[:trimmed_length]

    if aggr == 'first':
        return trimmed_array[::step]

    elif aggr in ['mean', 'sum']:
        # Reshape to group consecutive elements
        if dynamic_data.ndim == 1:
            reshaped = trimmed_array.reshape(-1, step) # (timesteps, step)
        else:
            reshaped = trimmed_array.reshape(-1, step, dynamic_data.shape[1]) # (timesteps, step, feature)

        if aggr == 'mean':
            return np.mean(reshaped, axis=1)
        elif aggr == 'sum':
            return np.sum(reshaped, axis=1)

    raise ValueError(f"Aggregation method '{aggr}' is not supported")

def get_water_depth(hec_ras_path: str, nodes_shp_path: str):
    """Get water depth from water level and elevation"""
    water_level = get_water_level(hec_ras_path)
    elevation = get_cell_elevation(nodes_shp_path)[None, :]
    water_depth = np.clip(water_level - elevation, a_min=0, a_max=None)
    return water_depth

def get_clipped_water_volume(hec_ras_path: str):
    """Remove exterme values in water volume"""
    CLIP_VOLUME = 100000  # in cubic meters
    water_volume = get_water_volume(hec_ras_path)
    water_volume = np.clip(water_volume, a_min=0, a_max=CLIP_VOLUME)
    return water_volume

def get_inflow(hec_ras_path: str, edges_shp_path: str, inflow_boundary_nodes: list[int]):
    """Get inflow at boundary nodes"""
    face_flow = get_face_flow(hec_ras_path)
    edge_index = get_edge_index(edges_shp_path)
    inflow_to_boundary_mask = np.isin(edge_index[1], inflow_boundary_nodes)
    if np.any(inflow_to_boundary_mask):
        # Flip the dynamic edge features accordingly
        face_flow[:, inflow_to_boundary_mask] *= -1

    inflow_edges_mask = np.any(np.isin(edge_index, inflow_boundary_nodes), axis=0)
    inflow = face_flow[:, inflow_edges_mask].sum(axis=1)

    out = np.stack([np.zeros(inflow.shape[0]), inflow]).T  # Shape (num_timesteps, 2)
    return out

def get_interval_rainfall(hec_ras_path: str):
    """Get interval rainfall from cumulative rainfall"""
    cumulative_rainfall = get_cumulative_rainfall(hec_ras_path)
    last_ts_rainfall = np.zeros((1, cumulative_rainfall.shape[1]), dtype=cumulative_rainfall.dtype)
    intervals = np.diff(cumulative_rainfall, axis=0)
    interval_rainfall = np.concatenate((intervals, last_ts_rainfall), axis=0)
    return interval_rainfall.sum(axis=1)

def create_constant_text_files(hec_ras_filepath: str, node_shp_filepath: str, dem_path: str, dataset_folder: str, prefix: str):
    pos = get_cell_position(node_shp_filepath)
    pos_path = os.path.join(dataset_folder, f"{prefix}_XY.txt")
    np.savetxt(pos_path, pos, delimiter='\t')

    area = get_cell_area(hec_ras_filepath)
    area_path = os.path.join(dataset_folder, f"{prefix}_CA.txt")
    np.savetxt(area_path, area, delimiter='\t')

    elevation = get_cell_elevation(node_shp_filepath)
    elevation_path = os.path.join(dataset_folder, f"{prefix}_CE.txt")
    np.savetxt(elevation_path, elevation, delimiter='\t')

    filled_dem = get_filled_dem(dem_path, os.path.join(dataset_folder, 'filled_dem.tif'))

    slope = get_slope(filled_dem, os.path.join(dataset_folder, 'slope_dem.tif'), pos)
    slope_path = os.path.join(dataset_folder, f"{prefix}_CS.txt")
    np.savetxt(slope_path, slope, delimiter='\t')

    aspect = get_aspect(filled_dem, os.path.join(dataset_folder, 'aspect_dem.tif'), pos)
    aspect_path = os.path.join(dataset_folder, f"{prefix}_A.txt")
    np.savetxt(aspect_path, aspect, delimiter='\t')

    curvature = get_curvature(filled_dem, os.path.join(dataset_folder, 'curvature_dem.tif'), pos)
    curvature_path = os.path.join(dataset_folder, f"{prefix}_CU.txt")
    np.savetxt(curvature_path, curvature, delimiter='\t')

    manning = get_roughness(hec_ras_filepath)
    manning_path = os.path.join(dataset_folder, f"{prefix}_N.txt")
    np.savetxt(manning_path, manning, delimiter='\t')

    flow_accum = get_flow_accumulation(filled_dem, os.path.join(dataset_folder, 'flow_dir_dem.tif'), os.path.join(dataset_folder, 'flow_acc_dem.tif'), pos)
    flow_accum_path = os.path.join(dataset_folder, f"{prefix}_FA.txt")
    np.savetxt(flow_accum_path, flow_accum, delimiter='\t')

    infiltration = get_cell_infiltration(len(pos))
    infiltration_path = os.path.join(dataset_folder, f"{prefix}_IP.txt")
    np.savetxt(infiltration_path, infiltration, delimiter='\t')

def create_dynamic_text_files(hec_ras_filepath: str,
                              node_shp_filepath: str,
                              edge_shp_filepath: str,
                              inflow_boundary_nodes: list[int],
                              dataset_folder: str,
                              prefix: str,
                              hydrograph_id: str,
                              spin_up_timesteps: int = 0,
                              ts_from_peak_water_volume: int = None,
                              downsample_interval: int = 1):
    volume = get_water_volume(hec_ras_filepath)
    total_water_volume = volume.sum(axis=1)
    peak_idx = np.argmax(total_water_volume).item()
    end_idx = peak_idx + ts_from_peak_water_volume

    water_depth = get_water_depth(hec_ras_filepath, node_shp_filepath)
    water_depth = water_depth[spin_up_timesteps:end_idx]
    water_depth = downsample_dynamic_data(water_depth, downsample_interval, aggr='mean')
    water_depth_path = os.path.join(dataset_folder, f"{prefix}_WD_{hydrograph_id}.txt")
    np.savetxt(water_depth_path, water_depth, delimiter='\t')

    volume = get_clipped_water_volume(hec_ras_filepath)
    volume = volume[spin_up_timesteps:end_idx]
    volume = downsample_dynamic_data(volume, downsample_interval, aggr='mean')
    volume_path = os.path.join(dataset_folder, f"{prefix}_V_{hydrograph_id}.txt")
    np.savetxt(volume_path, volume, delimiter='\t')

    inflow = get_inflow(hec_ras_filepath, edge_shp_filepath, inflow_boundary_nodes)
    inflow = inflow[spin_up_timesteps:end_idx]
    inflow = downsample_dynamic_data(inflow, downsample_interval, aggr='mean')
    inflow_path = os.path.join(dataset_folder, f"{prefix}_US_InF_{hydrograph_id}.txt")
    np.savetxt(inflow_path, inflow, delimiter='\t')

    precipitation = get_interval_rainfall(hec_ras_filepath)
    precipitation = precipitation[spin_up_timesteps:end_idx]
    precipitation = downsample_dynamic_data(precipitation, downsample_interval, aggr='sum')
    precipitation_path = os.path.join(dataset_folder, f"{prefix}_Pr_{hydrograph_id}.txt")
    np.savetxt(precipitation_path, precipitation, delimiter='\t')

def main():
    root_dir = ""
    config_file_path = ""
    dem_path = ""
    base_dataset_folder = f"outputs_phy/hecras_data"
    prefix = "M80"
    spin_up_timesteps = 864
    ts_from_peak_water_volume = 24 # Set to None to disable
    downsample_interval = 3

    # Get important paths
    info = get_info_from_config(config_file_path, root_dir)
    train_hec_ras_paths = get_hec_ras_paths_from_summary(info['train_summary_path'], root_dir)
    test_hec_ras_paths = get_hec_ras_paths_from_summary(info['test_summary_path'], root_dir)

    # Create dataset folder
    if not os.path.exists(base_dataset_folder):
        os.makedirs(base_dataset_folder)
    print(f"Saving files in folder: {base_dataset_folder}", flush=True)

    print(f"Creating hydrograph id files...", flush=True)
    create_hydrograph_id_file(train_hec_ras_paths.keys(), base_dataset_folder, 'train.txt')
    create_hydrograph_id_file(test_hec_ras_paths.keys(), base_dataset_folder, 'test.txt')

    # Create static text files; Can use arbitrary hec_ras file as they are the same
    print(f"Saving constant features...", flush=True)
    sample_hec_ras_path = next(iter(train_hec_ras_paths.values()))
    create_constant_text_files(sample_hec_ras_path,
                               info['nodes_shp_path'],
                               dem_path,
                               base_dataset_folder,
                               prefix)

    # Create dynamic text files
    all_hec_ras_paths = {**train_hec_ras_paths, **test_hec_ras_paths}
    for event_key, hec_ras_file_path in all_hec_ras_paths.items():
        print(f"Saving dynamic features for event {event_key}...", flush=True)
        create_dynamic_text_files(hec_ras_file_path,
                                  info['nodes_shp_path'],
                                  info['edges_shp_path'],
                                  info['inflow_boundary_nodes'],
                                  base_dataset_folder,
                                  prefix,
                                  hydrograph_id=event_key,
                                  spin_up_timesteps=spin_up_timesteps,
                                  ts_from_peak_water_volume=ts_from_peak_water_volume,
                                  downsample_interval=downsample_interval)

if __name__ == "__main__":
    main()
