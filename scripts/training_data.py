"""Read-only preflight for the detection datasets supported by the training wrapper."""

import math
from pathlib import Path


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def class_names(config):
    names = config.get('names')
    if isinstance(names, dict):
        indexed = {}
        for key, value in names.items():
            if type(key) is int:
                index = key
            elif isinstance(key, str) and key.isascii() and key.isdigit():
                index = int(key)
            else:
                raise ValueError('YOLO names keys must be integer class IDs')
            if index in indexed:
                raise ValueError('Duplicate YOLO class ID after numeric conversion')
            indexed[index] = value
        if sorted(indexed) != list(range(len(indexed))):
            raise ValueError('YOLO class IDs must be contiguous from zero')
        names = [indexed[index] for index in range(len(indexed))]
    if not isinstance(names, list) or not names or any(
            not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError('data.yaml requires nonempty class names')
    if 'nc' in config and (type(config['nc']) is not int or config['nc'] != len(names)):
        raise ValueError('YOLO nc differs from the number of class names')
    return names


def inside(root, path):
    path = path.resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'YOLO dataset path escapes dataset_dir: {path}')
    return path


def labels_directory(images):
    parts = list(images.parts)
    positions = [index for index, part in enumerate(parts) if part == 'images']
    if not positions:
        raise ValueError('YOLO image directories must contain an images path component')
    parts[positions[-1]] = 'labels'
    return Path(*parts)


def inspect_split(root, directory, number_of_classes):
    labels = inside(root, labels_directory(directory))
    if not labels.is_dir():
        raise FileNotFoundError(f'YOLO labels directory missing: {labels}')
    images = sorted(path for path in directory.rglob('*')
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f'No supported images in YOLO split: {directory}')
    counts = [0] * number_of_classes
    annotations, resolved_images = [], set()
    negative_images = 0
    used_labels = set()
    for image in images:
        resolved = inside(root, image)
        if resolved in resolved_images:
            raise ValueError(f'Duplicate image alias in YOLO split: {image}')
        resolved_images.add(resolved)
        label = inside(root, labels / image.relative_to(directory).with_suffix('.txt'))
        if label in used_labels:
            raise ValueError(f'Multiple images share one YOLO label: {label}')
        used_labels.add(label)
        if not label.is_file():
            raise FileNotFoundError(f'Missing YOLO label: {label}; use an empty .txt for an intentional negative image')
        annotations.append(str(label))
        objects = 0
        for line_number, line in enumerate(label.read_text(encoding='utf-8-sig').splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            location = f'{label}:{line_number}'
            if len(fields) != 5:
                raise ValueError(f'{location}: detection requires class_id cx cy width height (5 fields)')
            if not fields[0].isascii() or not fields[0].isdigit():
                raise ValueError(f'{location}: class ID must be a nonnegative integer')
            slot = int(fields[0])
            if slot >= number_of_classes:
                raise ValueError(f'{location}: class ID absent from data.yaml names')
            try:
                coordinates = [float(value) for value in fields[1:]]
            except ValueError as error:
                raise ValueError(f'{location}: invalid box coordinates') from error
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in coordinates):
                raise ValueError(f'{location}: box coordinates must be finite and normalized to [0,1]')
            if coordinates[2] <= 0 or coordinates[3] <= 0:
                raise ValueError(f'{location}: box width/height must be positive')
            counts[slot] += 1
            objects += 1
        negative_images += objects == 0
    return ({'images_dir': str(directory), 'labels_dir': str(labels), 'images': len(images),
             'objects': sum(counts), 'objects_by_class': counts, 'negative_images': negative_images},
            annotations, resolved_images)


def inspect_yolo(root, train):
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError('YOLO preflight requires PyYAML in the training environment') from error
    path = next((root / name for name in ('data.yaml', 'data.yml') if (root / name).is_file()), None)
    if path is None:
        raise FileNotFoundError(f'YOLO dataset requires data.yaml or data.yml in {root}')
    path = inside(root, path)
    config = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    if not isinstance(config, dict):
        raise ValueError('YOLO data.yaml must be a mapping')
    if 'kpt_shape' in config:
        raise ValueError('This training wrapper supports detection boxes, not YOLO pose')
    names = class_names(config)
    base = root
    if config.get('path') is not None:
        if not isinstance(config['path'], str):
            raise ValueError('YOLO path must be a directory string')
        supplied = Path(config['path'])
        candidate = inside(root, supplied if supplied.is_absolute() else root / supplied)
        if candidate.is_dir():
            base = candidate
        elif supplied.is_absolute():
            raise FileNotFoundError(f'YOLO base directory missing: {candidate}')
        # RF-DETR ignores a nonexisting relative base (stock exports may repeat the root name).
    required = ['train', 'val'] + (['test'] if train.get('run_test', True) else [])
    summaries, annotations, seen = {}, [str(path)], set()
    for split in required:
        value = config.get(split)
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'YOLO {split} must point to one image directory; image lists are not supported by this wrapper')
            relative = Path(value)
            directory = inside(root, relative if relative.is_absolute() else base / relative)
        else:
            alternatives = ['valid/images', 'val/images'] if split == 'val' else [split + '/images']
            directory = next((root / item for item in alternatives if (root / item).is_dir()), root / alternatives[0])
        if not directory.is_dir():
            raise FileNotFoundError(f'YOLO {split} images missing: {directory}')
        summary, files, images = inspect_split(root, directory, len(names))
        if seen.intersection(images):
            raise ValueError('YOLO splits share the same image files')
        seen.update(images)
        annotations.extend(files)
        summaries[split] = summary
    if summaries['train']['objects'] == 0:
        raise ValueError('YOLO training split has no labeled objects')
    return {'format': 'yolo', 'root': str(root), 'class_names': names,
            'splits': summaries, 'annotation_files': annotations,
            'validation_scope': 'Detection labels and file paths; image decoding and duplicate image content require separate checks'}


def inspect_dataset(train):
    root = Path(train['dataset_dir']).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'dataset_dir must be an existing directory: {root}')
    requested = train.get('dataset_file', 'auto')
    if requested not in {'auto', 'roboflow', 'yolo'}:
        raise ValueError('Wrapper supports dataset_file=auto, roboflow (COCO split JSON), or yolo')
    coco = (root / 'train/_annotations.coco.json').is_file()
    if requested == 'yolo' or (requested == 'auto' and not coco):
        return inspect_yolo(root, train)
    annotations = [root / split / '_annotations.coco.json' for split in ('train', 'valid', 'test')
                   if split != 'test' or train.get('run_test', True) or (root / split / '_annotations.coco.json').is_file()]
    for path in annotations:
        if not path.is_file():
            raise FileNotFoundError(f'Expected COCO split annotation: {path}')
    return {'format': 'roboflow', 'root': str(root), 'annotation_files': [str(path) for path in annotations],
            'validation_scope': 'COCO annotation file existence; use the dataset evaluator to validate content'}
