import argparse, time, numpy as np, tensorflow as tf, os, glob, json
from PIL import Image
from datetime import datetime

def load_image(path, size, dtype, quantization=None):
    img = Image.open(path).convert("RGB").resize(size, Image.BILINEAR)
    x = np.asarray(img)

    # Normalize to [0, 1] for float inputs
    if dtype == np.float32:
        x = x.astype(np.float32) / 255.0
    elif dtype == np.uint8:
        x = x.astype(np.uint8)
    elif dtype == np.int8:
        # Handle quantized INT8 input
        x = x.astype(np.float32) / 255.0  # Normalize to [0, 1]
        if quantization:
            scale, zero_point = quantization
            x = (x / scale) + zero_point
        x = np.clip(x, -128, 127).astype(np.int8)
    else:
        x = x.astype(dtype)

    return x[np.newaxis, ...]

def get_best_output_detail(out_details):
    # Pick the largest output tensor (most common for classifiers)
    return max(out_details, key=lambda d: int(np.prod(d["shape"])) if "shape" in d else 0)

def get_output_vector(interp, out_detail):
    y = interp.get_tensor(out_detail["index"])
    y = np.asarray(y)

    # Dequantize if needed
    q = out_detail.get("quantization", None)
    if q and isinstance(q, tuple) and len(q) == 2:
        scale, zero = q
        if scale and scale > 0:
            y = scale * (y.astype(np.float32) - float(zero))
        else:
            y = y.astype(np.float32)

    if y.size == 1:
        return y.reshape(1).astype(np.float32), True
    return y.reshape(-1).astype(np.float32), False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to .tflite file")
    ap.add_argument("--images", required=True, help="Path to test image or directory of images")
    ap.add_argument("--threads", type=int, default=1, help="CPU threads")
    ap.add_argument("--runs", type=int, default=1, help="Number of timed runs per image")
    ap.add_argument("--max-images", type=int, default=None, help="Maximum number of images to process")
    ap.add_argument("--batch-size", type=int, default=100, help="Process images in batches to manage memory")
    ap.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to save progress")
    ap.add_argument("--resume", type=str, default=None, help="Resume from checkpoint file")
    ap.add_argument("--progress-interval", type=int, default=50, help="Print progress every N images")
    args = ap.parse_args()

    # Get list of image files
    if os.path.isdir(args.images):
        # Find all image files in directory
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(args.images, ext)))
        image_files.sort()  # Sort for consistent ordering
    else:
        # Single image file
        image_files = [args.images]

    if not image_files:
        print(f"No image files found in {args.images}")
        return

    # Resume from checkpoint if specified
    processed_images = set()
    if args.resume and os.path.exists(args.resume):
        try:
            with open(args.resume, 'r') as f:
                checkpoint_data = json.load(f)
                processed_images = set(checkpoint_data.get('processed_images', []))
                print(f"Resuming from checkpoint: {len(processed_images)} images already processed")
        except Exception as e:
            print(f"Warning: Could not load checkpoint {args.resume}: {e}")

    # Filter out already processed images
    image_files = [img for img in image_files if os.path.basename(img) not in processed_images]

    # Limit number of images if specified
    if args.max_images:
        image_files = image_files[:args.max_images]

    if not image_files:
        print("All images already processed according to checkpoint!")
        return

    print(f"Found {len(image_files)} image(s) to process")
    print(f"Processing in batches of {args.batch_size}")
    print(f"Progress will be shown every {args.progress_interval} images")

    start_time = time.time()
    checkpoint_data = {
        'start_time': datetime.now().isoformat(),
        'total_images': len(image_files),
        'processed_images': list(processed_images),
        'batch_size': args.batch_size,
        'model': args.model
    }

    # Create interpreter
    Interpreter = tf.lite.Interpreter
    try:
        interp = Interpreter(model_path=args.model, num_threads=args.threads)
    except TypeError:
        interp = Interpreter(model_path=args.model)

    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out_details = interp.get_output_details()
    out_detail = get_best_output_detail(out_details)

    _, h, w, _ = inp["shape"]
    quantization = inp.get("quantization")

    # Process each image
    all_times = []
    results = []
    batch_start_time = time.time()

    for batch_start in range(0, len(image_files), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(image_files))
        batch_files = image_files[batch_start:batch_end]

        print(f"\nProcessing batch {batch_start//args.batch_size + 1}/{(len(image_files)-1)//args.batch_size + 1} "
              f"(images {batch_start+1}-{batch_end})")

        batch_times = []

        for i, image_path in enumerate(batch_files):
            global_idx = batch_start + i

            try:
                # Load and preprocess image
                x = load_image(image_path, (w, h), np.dtype(inp["dtype"]), quantization)

                # Warmup (only for first image)
                if global_idx == 0 and not processed_images:
                    interp.set_tensor(inp["index"], x)
                    interp.invoke()

                # Timed runs for this image
                image_times = []
                for _ in range(args.runs):
                    t0 = time.perf_counter()
                    interp.set_tensor(inp["index"], x)
                    interp.invoke()
                    _ = interp.get_tensor(out_detail["index"])
                    image_times.append((time.perf_counter() - t0) * 1000.0)

                avg_time = sum(image_times) / len(image_times)
                batch_times.extend(image_times)

                # Get classification result
                y, is_scalar = get_output_vector(interp, out_detail)
                if is_scalar:
                    top_pred = f"Scalar: {float(y[0]):.6f}"
                else:
                    top_idx = np.argmax(y)
                    top_score = float(y[top_idx])
                    top_pred = f"Class {top_idx}: {top_score:.6f}"

                results.append({
                    'image': os.path.basename(image_path),
                    'avg_latency_ms': avg_time,
                    'top_prediction': top_pred,
                    'batch': batch_start//args.batch_size + 1
                })

                processed_images.add(os.path.basename(image_path))

                # Progress reporting
                if (global_idx + 1) % args.progress_interval == 0:
                    elapsed = time.time() - start_time
                    rate = (global_idx + 1) / elapsed
                    eta = (len(image_files) - global_idx - 1) / rate if rate > 0 else 0
                    print(f"  Progress: {global_idx+1}/{len(image_files)} images "
                          f"({(global_idx+1)/len(image_files)*100:.1f}%) - "
                          f"Rate: {rate:.1f} img/sec - ETA: {eta/60:.1f} min")

            except Exception as e:
                print(f"  Error processing {os.path.basename(image_path)}: {e}")
                continue

        # Batch statistics
        if batch_times:
            batch_avg = sum(batch_times) / len(batch_times)
            batch_elapsed = time.time() - batch_start_time
            batch_rate = len(batch_files) / batch_elapsed if batch_elapsed > 0 else 0
            print(f"  Batch completed: {len(batch_files)} images, "
                  f"Avg latency: {batch_avg:.2f}ms, Rate: {batch_rate:.1f} img/sec")

        # Save checkpoint after each batch
        if args.checkpoint:
            checkpoint_data['processed_images'] = list(processed_images)
            checkpoint_data['last_batch'] = batch_start//args.batch_size + 1
            checkpoint_data['elapsed_time'] = time.time() - start_time

            try:
                with open(args.checkpoint, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
                print(f"  Checkpoint saved: {args.checkpoint}")
            except Exception as e:
                print(f"  Warning: Could not save checkpoint: {e}")

        batch_start_time = time.time()

    # Final summary statistics
    total_time = time.time() - start_time
    if results:
        avg_latency = sum(r['avg_latency_ms'] for r in results) / len(results)
        min_latency = min(r['avg_latency_ms'] for r in results)
        max_latency = max(r['avg_latency_ms'] for r in results)
        fps = 1000.0 / avg_latency if avg_latency > 0 else 0
        total_inference_time = sum(all_times) / 1000.0 if all_times else 0

        print(f"\n{'='*60}")
        print("FINAL PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total images processed: {len(results)}")
        print(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"Average latency per image: {avg_latency:.2f} ms")
        print(f"Min latency: {min_latency:.2f} ms")
        print(f"Max latency: {max_latency:.2f} ms")
        print(f"Average FPS: {fps:.2f}")
        print(f"Total inference time: {total_inference_time:.2f} seconds")
        print(f"Processing rate: {len(results)/total_time:.2f} images/second")
        print(f"Images per minute: {len(results)/total_time*60:.1f}")

        # Class distribution summary
        if results:
            class_counts = {}
            for result in results:
                pred = result['top_prediction']
                if 'Class' in pred:
                    class_id = pred.split(':')[0].replace('Class ', '')
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1

            print("\nTop 10 predicted classes:")
            sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
            for class_id, count in sorted_classes[:10]:
                percentage = (count / len(results)) * 100
                print(f"  Class {class_id}: {count} images ({percentage:.1f}%)")

        # Save detailed results
        results_file = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(results_file, 'w') as f:
                json.dump({
                    'summary': {
                        'total_images': len(results),
                        'total_time_seconds': total_time,
                        'avg_latency_ms': avg_latency,
                        'fps': fps,
                        'processing_rate_img_per_sec': len(results)/total_time
                    },
                    'results': results
                }, f, indent=2)
            print(f"\nDetailed results saved to: {results_file}")
        except Exception as e:
            print(f"Warning: Could not save results: {e}")

        print(f"\n{'='*60}")
        print("PROCESSING COMPLETE!")
        print(f"Successfully processed {len(results)} images in {total_time:.1f} minutes")
        print(f"Average performance: {fps:.1f} FPS, {avg_latency:.1f}ms per image")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()


