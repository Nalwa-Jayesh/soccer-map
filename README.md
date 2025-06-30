# Soccer Player Cross-Mapping System

A GPU-accelerated computer vision system for mapping players between different camera views (broadcast and tactical) in soccer videos.

## ✨ Features

- **GPU Accelerated**: Uses CUDA for fast, real-time player detection.
- **High-Quality Detection**: Powered by a custom-trained YOLO model for accurate player detection.
- **Advanced Mapping**: Combines color, spatial, and texture features to robustly map players between views.
- **Side-by-Side Visualization**: Generates an output video with a side-by-side view of both cameras, with mapped players highlighted.

## 🔑 Key Files

| File/Folder | Description |
|---|---|
| `mapping.py` | **Main executable script.** Run this to start the mapping process. |
| `broadcast.mp4` | The main broadcast video feed. |
| `tacticam.mp4` | The tactical camera video feed. |
| `best.pt` | The pre-trained YOLO model for player detection. |
| `requirements.txt` | Contains all the necessary Python packages to run the project. |
| `src/` | The source code directory containing all the core logic. |
| `output/` | The directory where all output files are saved. |
| `output/player_mapping_results.json` | The JSON file containing the detailed per-frame mapping results. |
| `output/broadcast_with_mapping.mp4`| The final output video showing the side-by-side visualization. |

## 🚀 Getting Started

### 1. Installation

Clone the repository and install the required packages.
```bash
git clone <repository_url>
cd soccer_cross-map
pip install -r requirements.txt
```

### 2. Run the Mapping

Execute the main mapping script from your terminal. This will process the videos and generate the `player_mapping_results.json` and the final visualization video.
```bash
python mapping.py
```
*You can customize the input videos and model by passing arguments. Use `python mapping.py --help` for more options.*

### 3. View the Results

- The detailed mapping data can be found in `output/player_mapping_results.json`.
- The final side-by-side video with mapped players highlighted can be found in `output/broadcast_with_mapping.mp4`.