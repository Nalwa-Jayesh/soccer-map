# Soccer Player Cross-Mapping System

A GPU-accelerated computer vision system for mapping players between different camera views (broadcast and tactical) in soccer videos.

## ✨ Features

- **GPU Accelerated**: Uses CUDA for fast, real-time player detection.
- **High-Quality Detection**: Powered by a custom-trained YOLO model for accurate player detection.
- **Advanced Mapping**: Combines color, spatial, and texture features to robustly map players between views.
- **Side-by-Side Visualization**: Generates an output video with a side-by-side view of both cameras, with mapped players highlighted.
- **Centralized Configuration**: All key parameters (paths, thresholds, batch sizes) are now easily adjustable in one place.
- **Well-Documented**: All modules and functions include clear docstrings for maintainability.

## 🔑 Key Files

| File/Folder | Description |
|---|---|
| `mapping.py` | **Main executable script.** Run this to start the mapping process. All configuration is at the top of this file. |
| `src/` | The source code directory containing all the core logic and visualization utilities. |
| `requirements.txt` | Contains all the necessary Python packages to run the project. |
| `setup.py` | Project metadata and installation configuration. |
| `output/` | The directory where all output files are saved. |
| `output/player_mapping_results.json` | The JSON file containing the detailed per-frame mapping results. |
| `output/broadcast_with_mapping.mp4`| The final output video showing the side-by-side visualization. |

## 🚀 Getting Started

### 1. Installation

Clone the repository and install the required packages.
```bash
git clone https://github.com/Nalwa-Jayesh/soccer-map
cd soccer-map
pip install -r requirements.txt
```

### 2. Configuration

All key parameters (input/output paths, model path, thresholds, etc.) are centralized in the `CONFIG` dictionary at the top of `mapping.py`. Adjust these values as needed for your environment and use case.

### 3. Run the Mapping

Execute the main mapping script from your terminal. This will process the videos and generate the `player_mapping_results.json` and the final visualization video.
```bash
python mapping.py
```
*You can customize the input videos and model by editing the `CONFIG` dictionary or passing arguments. Use `python mapping.py --help` for more options.*

### 4. View the Results

- The detailed mapping data can be found in `output/player_mapping_results.json`.
- The final side-by-side video with mapped players highlighted can be found in `output/broadcast_with_mapping.mp4`.

## 🧹 Code Quality & Refactoring

- All redundant scripts and utilities have been removed for clarity.
- Visualization logic is now modular and centralized in `src/visualization.py`.
- All code is well-documented with consistent docstrings.
- Configuration is centralized for easy tuning.

## 📄 License

MIT License