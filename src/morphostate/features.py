"""Optional timm adapter; users supply registered crops and licensed model access."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def extract_features(manifest, output, model_name, pretrained, checkpoint, batch_size, device):
    try:
        import timm
        from PIL import Image
        from timm.data import create_transform, resolve_model_data_config
    except ImportError as exc:
        raise ImportError("Install image extras: pip install -e '.[images]'") from exc
    path = Path(manifest).resolve()
    target = Path(output)
    if target.suffix != ".npz" or target.exists() or target.with_suffix(".json").exists():
        raise ValueError("Output must be a new .npz file")
    frame = pd.read_csv(path, dtype=str)
    if not {"cell_id", "patch_path"} <= set(frame) or frame.cell_id.isna().any() or frame.cell_id.duplicated().any():
        raise ValueError("Manifest requires unique nonempty cell_id and patch_path")
    if not len(frame) or batch_size < 1:
        raise ValueError("Empty manifest or invalid batch size")
    device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    # An explicit --pretrained downloads weights under the model provider's terms.
    # Local checkpoint loading is delegated to timm; load only trusted weights.
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=0,
                              checkpoint_path=checkpoint or "").to(device).eval()
    transform_config = resolve_model_data_config(model)
    transform = create_transform(**transform_config, is_training=False)
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(key.encode("utf-8"))
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str((array.dtype, array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    encoder_identity = {"model": model_name, "weights_sha256": digest.hexdigest(),
                        "transform": transform_config, "timm_version": timm.__version__}
    results = []
    with torch.inference_mode():
        for start in range(0, len(frame), batch_size):
            patches = []
            for image_path in frame.patch_path.iloc[start:start + batch_size]:
                with Image.open(path.parent / image_path) as image:
                    patches.append(transform(image.convert("RGB")))
            embedding = model(torch.stack(patches).to(device))
            if not torch.is_tensor(embedding) or embedding.ndim != 2:
                raise ValueError("Encoder must output [batch, embedding_dim]; provide a dedicated adapter otherwise")
            results.append(embedding.cpu().numpy().astype(np.float32))
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, cell_ids=frame.cell_id.to_numpy(dtype=str), features=np.concatenate(results))
    target.with_suffix(".json").write_text(json.dumps({"feature_source": "image", "feature_encoder": encoder_identity,
        "pretrained": pretrained, "checkpoint": checkpoint, "timm_version": timm.__version__,
        "note": "Registered crops supplied externally. Registration, biological scale and training overlap not verified."}, indent=2), encoding="utf-8")

