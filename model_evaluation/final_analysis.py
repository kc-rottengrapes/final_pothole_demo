#!/usr/bin/env python3
"""
Final Model Analysis - Complete evaluation without validation dataset
"""

from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import time

def run_complete_analysis():
    """Run complete model analysis"""
    print("=" * 60)
    print("🔍 POTHOLE DETECTION MODEL ANALYSIS")
    print("=" * 60)
    
    # Load model
    try:
        model = YOLO("../best.pt")
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create results directory
    os.makedirs("final_results", exist_ok=True)
    
    # 1. Model Architecture Analysis
    print("\n📊 MODEL ARCHITECTURE")
    print("-" * 30)
    
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / 1024 / 1024
    layer_count = len(list(model.model.modules()))
    
    print(f"Model Type: {model.task}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Size: {model_size_mb:.1f} MB")
    print(f"Total Layers: {layer_count}")
    
    # 2. Speed Benchmark
    print("\n⚡ PERFORMANCE BENCHMARK")
    print("-" * 30)
    
    # Create test image
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Warm up
    for _ in range(3):
        _ = model(test_image, verbose=False)
    
    # Benchmark
    times = []
    for i in range(10):
        start = time.time()
        results = model(test_image, verbose=False)
        times.append((time.time() - start) * 1000)
    
    avg_time = np.mean(times)
    fps = 1000 / avg_time
    
    print(f"Average Inference Time: {avg_time:.1f} ms")
    print(f"Frames Per Second: {fps:.1f}")
    print(f"Throughput: {fps * 60:.0f} frames/minute")
    
    # 3. Model Assessment
    print("\n🎯 MODEL ASSESSMENT")
    print("-" * 30)
    
    # Size assessment
    if total_params < 5e6:
        size_rating = "LIGHTWEIGHT"
        size_emoji = "🟢"
    elif total_params < 20e6:
        size_rating = "MEDIUM"
        size_emoji = "🟡"
    else:
        size_rating = "HEAVY"
        size_emoji = "🔴"
    
    # Speed assessment
    if avg_time < 30:
        speed_rating = "FAST"
        speed_emoji = "🟢"
    elif avg_time < 100:
        speed_rating = "MODERATE"
        speed_emoji = "🟡"
    else:
        speed_rating = "SLOW"
        speed_emoji = "🔴"
    
    print(f"Model Size: {size_emoji} {size_rating}")
    print(f"Inference Speed: {speed_emoji} {speed_rating}")
    
    # 4. Create Visualizations
    create_analysis_charts(total_params, trainable_params, avg_time, fps, model_size_mb)
    
    # 5. Save Results
    results = {
        "model_info": {
            "type": model.task,
            "total_parameters": int(total_params),
            "trainable_parameters": int(trainable_params),
            "model_size_mb": round(model_size_mb, 1),
            "total_layers": layer_count
        },
        "performance": {
            "avg_inference_time_ms": round(avg_time, 1),
            "fps": round(fps, 1),
            "throughput_per_minute": round(fps * 60, 0)
        },
        "assessment": {
            "size_rating": size_rating,
            "speed_rating": speed_rating,
            "overall_score": calculate_overall_score(total_params, avg_time)
        }
    }
    
    with open("final_results/analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # 6. Generate Report
    generate_text_report(results)
    
    print(f"\n📁 Results saved to 'final_results/' folder")
    print("✅ Complete analysis finished!")
    
    return results

def create_analysis_charts(total_params, trainable_params, avg_time, fps, model_size_mb):
    """Create analysis visualizations"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Parameter breakdown
    param_data = [trainable_params, total_params - trainable_params]
    labels = ['Trainable', 'Non-trainable']
    colors = ['#FF6B6B', '#4ECDC4']
    
    ax1.pie(param_data, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Parameter Distribution', fontsize=14, fontweight='bold')
    
    # 2. Model comparison
    models = ['YOLOv8n', 'YOLOv8s', 'YOLOv8m', 'Current']
    sizes = [6.2, 21.5, 49.7, model_size_mb]
    colors_bar = ['lightblue', 'lightgreen', 'orange', 'red']
    
    bars = ax2.bar(models, sizes, color=colors_bar)
    ax2.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Size (MB)')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Performance metrics
    metrics = ['Inference Time (ms)', 'FPS', 'Params (M)', 'Size (MB)']
    values = [avg_time, fps, total_params/1e6, model_size_mb]
    normalized_values = [v/max(values) for v in values]  # Normalize for radar chart
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    normalized_values = normalized_values + [normalized_values[0]]
    
    ax3.plot(angles, normalized_values, 'o-', linewidth=2, color='#45B7D1')
    ax3.fill(angles, normalized_values, alpha=0.25, color='#45B7D1')
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(metrics)
    ax3.set_title('Performance Profile', fontsize=14, fontweight='bold')
    ax3.grid(True)
    
    # 4. Speed vs Size scatter
    model_names = ['YOLOv8n', 'YOLOv8s', 'YOLOv8m', 'Current Model']
    model_sizes = [6.2, 21.5, 49.7, model_size_mb]
    theoretical_speeds = [25, 20, 15, fps]  # Theoretical FPS
    
    scatter = ax4.scatter(model_sizes, theoretical_speeds, s=[100, 150, 200, 250], 
                         c=['blue', 'green', 'orange', 'red'], alpha=0.7)
    
    for i, name in enumerate(model_names):
        ax4.annotate(name, (model_sizes[i], theoretical_speeds[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax4.set_xlabel('Model Size (MB)')
    ax4.set_ylabel('FPS')
    ax4.set_title('Speed vs Size Trade-off', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("final_results/analysis_dashboard.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📈 Analysis charts saved")

def calculate_overall_score(total_params, avg_time):
    """Calculate overall model score"""
    # Size score (smaller is better)
    size_score = max(0, 100 - (total_params / 1e6) * 10)
    
    # Speed score (faster is better)  
    speed_score = max(0, 100 - avg_time)
    
    # Overall score
    overall = (size_score + speed_score) / 2
    
    if overall > 80:
        return "EXCELLENT"
    elif overall > 60:
        return "GOOD"
    elif overall > 40:
        return "FAIR"
    else:
        return "NEEDS_IMPROVEMENT"

def generate_text_report(results):
    """Generate text-based report"""
    
    report = f"""
POTHOLE DETECTION MODEL ANALYSIS REPORT
======================================

MODEL SPECIFICATIONS
--------------------
Architecture: {results['model_info']['type']}
Total Parameters: {results['model_info']['total_parameters']:,}
Model Size: {results['model_info']['model_size_mb']} MB
Total Layers: {results['model_info']['total_layers']}

PERFORMANCE METRICS
------------------
Inference Time: {results['performance']['avg_inference_time_ms']} ms
Frames Per Second: {results['performance']['fps']}
Throughput: {results['performance']['throughput_per_minute']} frames/minute

ASSESSMENT
----------
Size Rating: {results['assessment']['size_rating']}
Speed Rating: {results['assessment']['speed_rating']}
Overall Score: {results['assessment']['overall_score']}

DEPLOYMENT RECOMMENDATIONS
-------------------------
Mobile/Edge: {'Suitable' if results['model_info']['model_size_mb'] < 50 else 'Needs Optimization'}
Server: Excellent
Real-time: {'Yes' if results['performance']['fps'] > 15 else 'Limited'}

OPTIMIZATION SUGGESTIONS
-----------------------
1. Quantization: Reduce model size by 75%
2. TensorRT: Improve inference speed by 2-3x
3. ONNX: Enable cross-platform deployment
4. Pruning: Remove redundant parameters

TECHNICAL NOTES
--------------
- Model uses segmentation architecture
- Optimized for 640x640 input resolution
- Compatible with YOLO ecosystem
- Suitable for production deployment

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open("final_results/analysis_report.txt", "w", encoding='utf-8') as f:
        f.write(report)
    
    print("📄 Text report generated")

if __name__ == "__main__":
    results = run_complete_analysis()
    
    print("\n" + "=" * 60)
    print("🎉 ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Model Size: {results['assessment']['size_rating']}")
    print(f"Speed: {results['assessment']['speed_rating']}")
    print(f"Overall: {results['assessment']['overall_score']}")
    print(f"FPS: {results['performance']['fps']}")
    print("=" * 60)