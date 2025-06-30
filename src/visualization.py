import cv2
import os
import logging
from src.core import PlayerMappingSystem
from typing import Optional

def stream_and_visualize_mapping(broadcast_path: str, tacticam_path: str, model_path: Optional[str] = None, use_gpu: bool = True, save_output: bool = True):
    """
    Stream both videos, run detection and mapping in real time, overlay results, and display/save visualization.
    """
    os.makedirs('output', exist_ok=True)
    broadcast_cap = cv2.VideoCapture(broadcast_path)
    tacticam_cap = cv2.VideoCapture(tacticam_path)
    fps = broadcast_cap.get(cv2.CAP_PROP_FPS)
    width = int(broadcast_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(broadcast_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = None
    if save_output:
        out = cv2.VideoWriter('output/visualization.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width*2, height))
    mapper = PlayerMappingSystem(model_path, use_gpu)
    frame_idx = 0
    while True:
        ret_b, frame_b = broadcast_cap.read()
        ret_t, frame_t = tacticam_cap.read()
        if not ret_b or not ret_t:
            break
        players_b = mapper.detect_players_in_frame(frame_b, frame_idx)
        players_t = mapper.detect_players_in_frame(frame_t, frame_idx)
        # Simple mapping: by index (for demo); replace with real mapping logic
        for i, player in enumerate(players_b):
            bbox = player['bbox']
            cv2.rectangle(frame_b, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,255,0), 2)
            cv2.putText(frame_b, f"B{i}", (int(bbox[0]), int(bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        for i, player in enumerate(players_t):
            bbox = player['bbox']
            cv2.rectangle(frame_t, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,0,0), 2)
            cv2.putText(frame_t, f"T{i}", (int(bbox[0]), int(bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        vis = cv2.hconcat([frame_b, frame_t])
        cv2.imshow('Player Mapping Visualization', vis)
        if save_output and out is not None:
            out.write(vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_idx += 1
    broadcast_cap.release()
    tacticam_cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

def create_visualization_from_existing(broadcast_path: str, mapping_results_path: str, output_path: str, max_frames: int = 1000) -> bool:
    """
    Create visualization video from existing mapping results.
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("CREATING VISUALIZATION FROM EXISTING RESULTS")
    logger.info("=" * 60)
    if not os.path.exists(broadcast_path):
        logger.error(f"❌ Broadcast video not found: {broadcast_path}")
        return False
    if not os.path.exists(mapping_results_path):
        logger.error(f"❌ Mapping results not found: {mapping_results_path}")
        return False
    try:
        logger.info("Initializing visualization system...")
        mapper = PlayerMappingSystem(use_gpu=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        success = mapper.create_visualization_video(
            broadcast_path=broadcast_path,
            mapping_results_path=mapping_results_path,
            output_video_path=output_path,
            max_frames=max_frames
        )
        if success:
            logger.info("=" * 60)
            logger.info("✅ VISUALIZATION COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info(f"📹 Output video: {output_path}")
            logger.info(f"📊 Source mapping data: {mapping_results_path}")
            logger.info(f"🎬 Source broadcast video: {broadcast_path}")
            try:
                output_size = os.path.getsize(output_path) / (1024*1024)
                source_size = os.path.getsize(broadcast_path) / (1024*1024)
                logger.info(f"📏 Output file size: {output_size:.1f} MB")
                logger.info(f"📏 Source file size: {source_size:.1f} MB")
            except:
                pass
            logger.info("\n🎯 Visualization Features:")
            logger.info("• Colored bounding boxes for each detected player")
            logger.info("• Persistent player IDs (P1, P2, P3, etc.)")
            logger.info("• Detection confidence scores")
            logger.info("• Cross-camera mapping information")
            logger.info("• Frame counter and processing statistics")
            logger.info("• Semi-transparent overlay with system info")
        else:
            logger.error("❌ Visualization creation failed!")
        return success
    except Exception as e:
        logger.error(f"❌ Visualization failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def create_full_pipeline(broadcast_path: str, tacticam_path: str, output_video_path: str, mapping_results_path: str = None, model_path: str = None, max_frames: int = 500) -> bool:
    """
    Run full pipeline: mapping + visualization.
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("RUNNING FULL PIPELINE: MAPPING + VISUALIZATION")
    logger.info("=" * 60)
    if not os.path.exists(broadcast_path):
        logger.error(f"❌ Broadcast video not found: {broadcast_path}")
        return False
    if not os.path.exists(tacticam_path):
        logger.error(f"❌ Tacticam video not found: {tacticam_path}")
        return False
    if mapping_results_path is None:
        mapping_results_path = "output/mapping_results.json"
    try:
        logger.info("Initializing player mapping system...")
        mapper = PlayerMappingSystem(
            model_path=model_path if model_path and os.path.exists(model_path) else None,
            use_gpu=True
        )
        gpu_info = mapper.gpu_manager.get_device_info()
        logger.info(f"Using device: {gpu_info['device_name']} ({gpu_info['device']})")
        os.makedirs(os.path.dirname(mapping_results_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        logger.info("\n" + "=" * 40)
        logger.info("STEP 1: CREATING PLAYER MAPPINGS")
        logger.info("=" * 40)
        results = mapper.process_videos(
            broadcast_path=broadcast_path,
            tacticam_path=tacticam_path,
            output_path=mapping_results_path,
            max_frames=max_frames,
            batch_size=8 if gpu_info['gpu_available'] else 1
        )
        if not results:
            logger.error("❌ Mapping process failed!")
            return False
        mappings = results.get('global_mapping', {}).get('mappings', {})
        if mappings:
            logger.info(f"✅ Successfully mapped {len(mappings)} players:")
            confidences = results.get('global_mapping', {}).get('confidence', {})
            for broadcast_id, tacticam_id in mappings.items():
                confidence = confidences.get(broadcast_id, 0)
                logger.info(f"  {broadcast_id} ↔ {tacticam_id} (confidence: {confidence:.3f})")
        else:
            logger.warning("⚠️  No player mappings found")
        logger.info("\n" + "=" * 40)
        logger.info("STEP 2: CREATING VISUALIZATION")
        logger.info("=" * 40)
        viz_success = mapper.create_visualization_video(
            broadcast_path=broadcast_path,
            mapping_results_path=mapping_results_path,
            output_video_path=output_video_path,
            max_frames=max_frames
        )
        if viz_success:
            logger.info("\n" + "=" * 60)
            logger.info("🎉 FULL PIPELINE COMPLETED SUCCESSFULLY!")
        else:
            logger.error("❌ Visualization step failed!")
        return viz_success
    except Exception as e:
        logger.error(f"❌ Full pipeline failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False 