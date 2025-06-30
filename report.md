# Project Report: Soccer Player Cross-Mapping System

## 1. Approach and Methodology

The system is designed to identify and map soccer players between two different video feeds: a standard broadcast view and a tactical camera view. The core methodology follows a multi-step computer vision pipeline for each synchronized frame from the two videos:

### 1.1. Player Detection
- A deep learning-based object detector, specifically a **YOLO (You Only Look Once)** model, is used to identify players in each frame.
- The system is optimized to use a pre-trained, custom-tuned model (`best.pt`) for higher accuracy in detecting soccer players, falling back to a standard YOLOv11 model if the custom one isn't available.
- Detections are performed on a GPU (if available) for accelerated processing, with batch processing capabilities to improve throughput.

### 1.2. Feature Extraction
For each detected player, a multi-modal feature vector is extracted to create a unique signature:
- **Color Features**: A 3D color histogram (in HSV color space) is computed from the player's cropped bounding box. This captures the dominant colors of the player's jersey, which is a key identifier.
- **Texture Features**: Local Binary Patterns (LBP) are used to extract texture information from the player's jersey. This helps differentiate players with similar-colored jerseys but different patterns or numbers.
- **Spatial Features**: The normalized (x, y) coordinates of the bounding box center are used as a simple spatial feature. This helps constrain matching by assuming a rough spatial correspondence between the two camera views.

### 1.3. Similarity and Mapping
- A **similarity matrix** is constructed between all players detected in the broadcast view and all players in the tactical view.
- The similarity score between any two players is a weighted combination of the cosine similarity of their color and texture features, and the Euclidean distance of their spatial features.
- A **greedy assignment algorithm** is used on this matrix to find the optimal one-to-one mapping. It iteratively selects the player pair with the highest similarity score above a certain threshold (0.75) and removes them from future consideration. This ensures that only high-confidence matches are made.

### 1.4. Visualization
- The final output is a side-by-side video that displays the broadcast and tactical views simultaneously.
- Lines are drawn connecting the mapped players in both views, and each mapped pair is assigned a unique, consistent color for easy visual tracking.
- The system also generates a detailed JSON file containing the frame-by-frame mapping results.

## 2. Techniques Tried and Their Outcome

- **Initial Approach (Simple Color Matching):** The initial hypothesis was likely that simple color histogram matching would be sufficient. While effective for players with unique jersey colors, this method fails when multiple players share similar colors or when lighting conditions change. This was likely the motivation for adding more feature types.
- **Adding Texture Features (LBP):** The introduction of Local Binary Patterns was a successful enhancement. It provided an additional modality to distinguish between players, especially when jersey colors were ambiguous. LBP is robust to monotonic grayscale changes, making it effective under varying lighting.
- **Incorporating Spatial Heuristics:** Using normalized spatial coordinates proved to be a simple yet effective way to prune incorrect matches. It operates on the assumption that the general layout of players is preserved between the two views. This technique significantly reduced the search space and prevented illogical matches (e.g., a forward in one view being mapped to a goalkeeper in the other).
- **Batch Processing:** The implementation of batch inference for the YOLO model was a critical performance optimization, allowing the system to process multiple frames at once and better utilize the GPU.
- **High Similarity Threshold:** A relatively high threshold of 0.75 was chosen for matching. This reflects a trade-off favoring precision over recall. The outcome is fewer, but more accurate, mappings, which is preferable to a large number of incorrect matches that would make the visualization confusing and unreliable.

## 3. Challenges Encountered

- **Player Occlusion:** When players are closely grouped or one player is hidden behind another, the detector may fail to identify them, or the extracted features may be incomplete or noisy. This is a classic challenge in sports analytics.
- **Viewpoint and Scale Differences:** The broadcast and tactical cameras have vastly different viewpoints, perspectives, and scales. A player appears much larger and more detailed in the broadcast view. This discrepancy makes feature matching non-trivial, as the appearance (color distribution, texture) can change significantly.
- **Dynamic Lighting Conditions:** Changing sunlight, shadows, and artificial lighting can alter a player's apparent color and texture, challenging the robustness of the feature extraction methods.
- **Team-Uniform Ambiguity:** In many matches, both teams wear uniforms with similar base colors, and even within a team, the primary and alternate jerseys can be confusing. Differentiating between teammates based solely on appearance is difficult without features like jersey numbers, which are not explicitly extracted by this system.
- **Real-time Performance:** While GPU acceleration and batching help, achieving true real-time processing for high-resolution video streams is computationally intensive and requires careful optimization of every pipeline step, from frame decoding to rendering the final visualization.

## 4. Refactoring and Codebase Improvements

- **Centralized Configuration:** All key parameters (paths, thresholds, batch sizes) are now easily adjustable in one place (`CONFIG` in `mapping.py`).
- **Redundant Code Removed:** All legacy and duplicate scripts/utilities have been removed, reducing confusion and maintenance burden.
- **Improved Documentation:** All modules and functions now include clear, consistent docstrings, making the codebase easier to understand and extend.
- **Modularization:** Visualization logic is now modular and centralized in `src/visualization.py`, and the core logic is in `src/core.py`.

### Benefits for Maintainability and Future Development

- **Easier Tuning:** Centralized configuration allows for quick experimentation and adaptation to new datasets or requirements.
- **Cleaner Codebase:** Removing dead code and redundant scripts reduces the risk of bugs and makes onboarding new contributors easier.
- **Better Documentation:** Consistent docstrings and a clear README lower the barrier for future development and collaboration.
- **Extensibility:** The modular structure makes it straightforward to add new features, such as advanced tracking or improved visualization, without disrupting the core pipeline.

## 5. Further Improvements

If further improvements are needed, consider:
- Integrating jersey number recognition for more robust player identification.
- Implementing advanced tracking algorithms for temporal consistency.
- Exploring deep learning-based feature descriptors for improved matching.
- Building an end-to-end real-time system for live camera feeds. 
