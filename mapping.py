import json
import logging
import os
import sys
import time
import traceback
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import argparse

from src.gpu_manager import GPUManager
from src.detection import get_detector
from src.features import extract_color_features, extract_spatial_features, extract_texture_features
from src.similarity import calculate_similarity
from src.tracking import update_tracks, clean_old_tracks
from src.core import PlayerMappingSystem
from src.visualization import visualize_mapping_from_file

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Centralized configuration
CONFIG = {
    'broadcast_video': 'broadcast.mp4',
    'tacticam_video': 'tacticam.mp4',
    'model_path': 'best.pt',
    'output_dir': 'output',
    'output_file': 'output/player_mapping_results.json',
    'visualization_file': 'output/broadcast_with_mapping.mp4',
    'max_frames': 1000,
    'batch_size': 4,
    'similarity_threshold': 0.75,
}

"""
Main entry point for the Soccer Player Cross-Mapping System.
This script runs the player mapping pipeline and generates visualizations.
"""

def main():
    """
    Main execution function for the multi-camera player mapping system.
    Handles initialization, processing, and visualization, with error handling and logging.
    """
    logger.info("=== Multi-Camera Player Mapping System ===")

    # Create output directory if it doesn't exist
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    try:
        # Initialize mapping system
        logger.info("Initializing player mapping system...")
        mapper = PlayerMappingSystem(CONFIG['model_path'] if os.path.exists(CONFIG['model_path']) else None)

        # Process videos
        results = mapper.process_videos(
            broadcast_path=CONFIG['broadcast_video'],
            tacticam_path=CONFIG['tacticam_video'],
            output_path=CONFIG['output_file'],
            max_frames=CONFIG['max_frames'],  # Limit frames for demo
        )

        # Display results
        if results and "global_mapping" in results:
            mappings_list = results["global_mapping"]["mappings"]

            logger.info("\n=== FINAL PLAYER MAPPING RESULTS ===")
            for frame_idx, mapping in enumerate(mappings_list):
                logger.info(f"Frame {frame_idx}:")
                for broadcast_player, tacticam_player in mapping.items():
                    logger.info(f"  {broadcast_player} -> {tacticam_player}")

            # Create visualization video
            logger.info("\n=== CREATING VISUALIZATION VIDEO ===")
            viz_success = visualize_mapping_from_file(
                broadcast_path=CONFIG['broadcast_video'],
                tacticam_path=CONFIG['tacticam_video'],
                mapping_results_path=CONFIG['output_file'],
                output_video_path=CONFIG['visualization_file'],
                detect_players_fn=mapper.detect_players_in_frame
            )

            if viz_success:
                logger.info("\n✅ COMPLETE! Check these files:")
                logger.info(f"📊 Mapping Data: {CONFIG['output_file']}")
                logger.info(f"🎥 Visualization: {CONFIG['visualization_file']}")
            else:
                logger.warning(
                    "Visualization creation failed, but mapping data is available"
                )

        else:
            logger.warning(
                "No mapping results generated. Check input videos and try again."
            )

        logger.info(f"\nDetailed results saved to: {CONFIG['output_file']}")

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Processing failed with error: {e}")
        logger.error(traceback.format_exc())

        # Provide helpful troubleshooting information
        logger.info("\n=== TROUBLESHOOTING ===")
        logger.info(
            "1. Ensure video files 'broadcast.mp4' and 'tacticam.mp4' exist in current directory"
        )
        logger.info(
            "2. Install required packages: pip install opencv-python numpy scipy scikit-learn ultralytics"
        )
        logger.info("3. For custom YOLO model, place 'best.pt' in current directory")
        logger.info(
            "4. Check video file formats and ensure they are readable by OpenCV"
        )
        logger.info("5. Ensure sufficient disk space for output video file")


def create_visualization_only():
    """
    Standalone function to create visualization from existing mapping results.
    Useful when you already have mapping results and just want to create the video.
    Returns:
        bool: True if visualization was created successfully, False otherwise.
    """
    logger.info("=== Creating Visualization from Existing Results ===")

    broadcast_video = "broadcast.mp4"
    mapping_results = "output/player_mapping_results.json"
    output_video = "output/broadcast_with_mapping.mp4"

    # Initialize mapper (model not needed for visualization)
    mapper = PlayerMappingSystem()

    # Create visualization
    success = visualize_mapping_from_file(
        broadcast_path=broadcast_video,
        tacticam_path=tacticam_video,
        mapping_results_path=mapping_results,
        output_video_path=output_video,
        detect_players_fn=mapper.detect_players_in_frame
    )

    if success:
        logger.info(f"✅ Visualization created: {output_video}")
    else:
        logger.error("❌ Visualization creation failed")

    return success


if __name__ == "__main__":
    """
    Command-line interface for the Soccer Player Mapping System.
    Parses arguments and runs the appropriate pipeline step.
    """
    parser = argparse.ArgumentParser(description="Soccer Player Mapping System")
    parser.add_argument('--visualize', action='store_true', help='Run real-time visualization utility')
    parser.add_argument('--broadcast', type=str, default='broadcast.mp4', help='Broadcast video path')
    parser.add_argument('--tacticam', type=str, default='tacticam.mp4', help='Tacticam video path')
    parser.add_argument('--model', type=str, default='best.pt', help='YOLO model path')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    parser.add_argument('--no-save', action='store_true', help='Do not save output video')
    args = parser.parse_args()

    if args.visualize:
        stream_and_visualize_mapping(
            args.broadcast,
            args.tacticam,
            args.model,
            not args.cpu,
            not args.no_save
        )
    elif len(sys.argv) > 1 and sys.argv[1] == "--visualize-only":
        create_visualization_only()
    else:
        main()
