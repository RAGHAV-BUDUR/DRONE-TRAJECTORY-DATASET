# DRONE-TRAJECTORY-DATASET

Synthetic UAV Flight Trajectories dataset sourced from RIOTU-Lab on Hugging Face:
[riotu-lab/Synthetic-UAV-Flight-Trajectories](https://huggingface.co/datasets/riotu-lab/Synthetic-UAV-Flight-Trajectories).

## Summary

This dataset comprises over 5,000 random UAV (Unmanned Aerial Vehicle) trajectories collected over 20 hours of flight time. It is intended for training AI models such as trajectory prediction applications. The dataset is generated through an automated pipeline for the creation and preprocessing of UAV synthetic trajectories, making it ready for direct AI model training.

## Data Description

The dataset features parameterized trajectories following predefined patterns, specifically circular and infinity-like paths:
- 2D trajectories (`gazebo_trajectory2D-2_*.csv`)
- 3D trajectories (`gazebo_trajectory3D_*.csv`, `gazebo_trajectory3D-2_*.csv`)
- Standard trajectories (`gazebo_trajectory_*.csv`)

## Dataset Structure

Each CSV file contains sequential time-series trajectory points with the following fields:
- `timestamp`: Recording time of the data point (UNIX epoch timestamp in seconds).
- `tx`: Position X coordinate in meters.
- `ty`: Position Y coordinate in meters.
- `tz`: Position Z coordinate in meters.

## Total Files and Size

- **Total CSV files:** 5,093 trajectories
- **Total rows:** ~766,140 data points
- **Total size:** ~55.7 MB

## FRED The Florence RGB-Event Drone Dataset

https://drive.google.com/drive/folders/1pISIErXOx76xmCqkwhS3-azWOMlTKZMp

## MMUAD Dataset

https://github.com/ntu-aris/MMAUD

https://www.kaggle.com/datasets/sakethramkodukula/mmaud-data

## Citation

If you use this work in your research, please cite:

```bibtex
@article{nacar2025vector,
  title={VECTOR: Velocity-Enhanced GRU Neural Network for Real-Time 3D UAV Trajectory Prediction},
  author={Nacar, O. and Abdelkader, M. and Ghouti, L. and Gabr, K. and Al-Batati, A. and Koubaa, A.},
  journal={Drones},
  volume={9},
  number={1},
  pages={8},
  year={2025},
  publisher={MDPI},
  doi={10.3390/drones9010008}
}
```

## License

This dataset is released under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
