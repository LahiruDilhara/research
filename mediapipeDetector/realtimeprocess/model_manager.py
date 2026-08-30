"""
realtimeprocess/model_manager.py

Model Discovery & Real-Time Inference Manager.

Automatically discovers trained .pth weight files in deepLearningModels/weights/,
maps filenames to model architectures and feature variants, loads weights on CPU/CUDA,
applies StandardScaler dataset normalization matching training (model_arch.py: normalize),
and runs 5-finger batch inference on live 5-frame sequence windows.
"""

import importlib
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deepLearningModels.run_all import ALL_SCRIPTS
from deepLearningModels.model_arch import parse_variant_csv, get_data_paths
from realtimeprocess.realtime_pipeline import (
    HandScaleNormalizer,
    compute_window_velocities,
    unroll_per_finger_window,
    extract_variant_tensor,
    ALL_21_LANDMARK_NAMES,
    FINGERS,
)


class ModelManager:
    """Discovers, loads, and executes PyTorch models for real-time streaming touch inference."""

    def __init__(self, weights_dir: Path = None, device: str = None):
        self.project_root = PROJECT_ROOT
        self.weights_dir = weights_dir or (self.project_root / "deepLearningModels" / "weights")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.normalizer = HandScaleNormalizer()
        self.available_models = []
        self.active_model_idx = 0
        self.active_model = None
        self.active_info = None
        self.scaler_cache = {}

        self._discover_models()
        if self.available_models:
            self.load_model_by_index(0)

    def get_scaler_for_variant(self, variant_name: str) -> StandardScaler:
        """Loads/fits and caches StandardScaler for the specified feature variant using training_dataset.csv."""
        variant_name = variant_name.lower().strip()
        if variant_name in self.scaler_cache:
            return self.scaler_cache[variant_name]

        train_csv, _ = get_data_paths(self.project_root)
        if not train_csv.exists():
            print(f"[ModelManager Warning] Training CSV not found at '{train_csv}'. Using un-scaled features.")
            return None

        try:
            X_tr_raw, _, _, _ = parse_variant_csv(train_csv, variant_name)
            N_tr, T, C = X_tr_raw.shape

            scaler = StandardScaler()
            scaler.fit(X_tr_raw.reshape(N_tr, -1))

            self.scaler_cache[variant_name] = scaler
            return scaler
        except Exception as e:
            print(f"[ModelManager Warning] Failed to compute StandardScaler for variant '{variant_name}': {e}")
            return None

    def _discover_models(self):
        """Scans deepLearningModels/weights/*.pth and maps to ALL_SCRIPTS architecture catalog."""
        self.available_models = []

        if not self.weights_dir.exists():
            print(f"[Warning] Weights directory not found: {self.weights_dir}")
            return

        for script_file, display_title, variant_name in ALL_SCRIPTS:
            module_name = script_file.replace(".py", "")
            try:
                mod = importlib.import_module(module_name)
            except ImportError:
                try:
                    mod = importlib.import_module(f"deepLearningModels.{module_name}")
                except ImportError:
                    continue

            arch_name = getattr(mod, "ARCH_NAME", module_name)
            configs   = getattr(mod, "CONFIGS", [{'id': 1}])
            cfg       = configs[0]

            weights_path = self.weights_dir / f"{arch_name}_cfg{cfg['id']:02d}.pth"
            if weights_path.exists():
                self.available_models.append({
                    "arch_name": arch_name,
                    "display_title": display_title,
                    "variant_name": variant_name,
                    "script_file": script_file,
                    "weights_path": weights_path,
                    "module": mod,
                    "cfg": cfg,
                })

        print(f"[ModelManager] Discovered {len(self.available_models)} trained model checkpoint(s) in {self.weights_dir}")
        for idx, m in enumerate(self.available_models):
            print(f"  [{idx + 1}] {m['arch_name']} ({m['variant_name']}) -> {m['weights_path'].name}")

    def set_device(self, device_str: str):
        """Switches execution device between 'cuda' and 'cpu'."""
        req_device = torch.device(device_str)
        if req_device.type == "cuda" and not torch.cuda.is_available():
            print("[ModelManager] CUDA requested but not available. Staying on CPU.")
            self.device = torch.device("cpu")
        else:
            self.device = req_device

        if self.active_model is not None:
            self.active_model.to(self.device)
            print(f"[ModelManager] Active Model transferred to {self.device}")

    def load_model_by_index(self, idx: int) -> bool:
        """Loads a model by its catalog index in available_models."""
        if idx < 0 or idx >= len(self.available_models):
            print(f"[ModelManager Error] Index {idx} out of range (0..{len(self.available_models)-1})")
            return False

        info = self.available_models[idx]
        mod = info["module"]
        cfg = info["cfg"]
        weights_path = info["weights_path"]
        variant_name = info["variant_name"]

        # Pre-cache scaler for active variant
        self.get_scaler_for_variant(variant_name)

        create_model_fn = getattr(mod, "create_model", None)
        if create_model_fn is None:
            print(f"[ModelManager Error] Module {info['arch_name']} missing create_model()")
            return False

        dummy_X = extract_variant_tensor(
            {f: {} for f in FINGERS},
            variant_name
        )
        feature_dim = dummy_X.shape[2]

        model = create_model_fn(feature_dim, cfg)
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        self.active_model = model
        self.active_model_idx = idx
        self.active_info = info
        self.active_info["seq_len"] = dummy_X.shape[1]
        self.active_info["feature_dim"] = feature_dim

        print(f"[ModelManager] Loaded Active Model: '{info['arch_name']}' ({variant_name}) on {self.device}")
        return True

    def switch_next_model(self):
        """Switches to the next available model in catalog."""
        if not self.available_models:
            return
        next_idx = (self.active_model_idx + 1) % len(self.available_models)
        self.load_model_by_index(next_idx)

    def switch_prev_model(self):
        """Switches to the previous available model in catalog."""
        if not self.available_models:
            return
        prev_idx = (self.active_model_idx - 1) % len(self.available_models)
        self.load_model_by_index(prev_idx)

    def predict_window(
        self,
        norm_frames_5: list[dict[str, float]],
        hand_scores_5: list[float] = None,
        frame_w: float = 640.0,
        frame_h: float = 480.0
    ) -> dict[str, dict]:
        """
        Given a 5-frame sequence window of scale-normalized wrist-centered landmarks:
        1. Computes 4 frame-to-frame velocity steps.
        2. Validates quality & cleaning filters (process.sh steps 8 & 9: min_avg_score 0.65, zero vel check).
        3. Unrolls 5 per-finger feature rows.
        4. Builds input tensor (5, seq_len, feature_dim) matching active model.
        5. Applies StandardScaler matching training normalization (model_arch.py: normalize).
        6. Runs model inference and returns per-finger Touch status & probability dict.
        """
        if self.active_model is None:
            return {f: {"touch": False, "prob": 0.0, "reason": "No Model"} for f in FINGERS}

        # Step 1: Compute 4 velocity steps from normalized 5 frames
        v_steps_4 = compute_window_velocities(norm_frames_5)

        # Step 2: Validate window quality & dataset cleaning rules (process.sh Steps 8 & 9)
        from realtimeprocess.realtime_pipeline import validate_realtime_window_quality
        is_valid, reason = validate_realtime_window_quality(v_steps_4, hand_scores_5, min_avg_score=0.65)
        if not is_valid:
            return {f: {"touch": False, "prob": 0.0, "reason": reason} for f in FINGERS}

        # Step 3: Unroll 5 fingers
        finger_rows = unroll_per_finger_window(norm_frames_5, v_steps_4)

        # Step 4: Extract feature tensor
        variant_name = self.active_info["variant_name"]
        X_np = extract_variant_tensor(finger_rows, variant_name)

        # Step 5: Apply StandardScaler matching training normalization (model_arch.py: normalize)
        scaler = self.get_scaler_for_variant(variant_name)
        if scaler is not None:
            N_b, T, C = X_np.shape
            X_scaled = scaler.transform(X_np.reshape(N_b, -1)).reshape(N_b, T, C)
        else:
            X_scaled = X_np

        # Step 6: PyTorch Inference
        X_tensor = torch.from_numpy(X_scaled.astype(np.float32)).to(self.device)

        with torch.inference_mode():
            logits = self.active_model(X_tensor)
            probs  = torch.sigmoid(logits).cpu().numpy().ravel()

        results = {}
        for idx, f_name in enumerate(FINGERS):
            p = float(probs[idx])
            results[f_name] = {
                "touch": bool(p >= 0.5),
                "prob": p,
            }

        return results
