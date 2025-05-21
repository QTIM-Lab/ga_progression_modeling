import yaml
import subprocess
import time
from datetime import datetime
import os, sys
from pathlib import Path
import threading
import queue
import json
from rich.console import Console, Group
from rich.progress import Progress, BarColumn, TextColumn
from rich.live import Live
from rich.panel import Panel
from shutil import copy

import warnings
warnings.filterwarnings('ignore')

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run_script(script_path, args=None):
    args = args or []
    cmd = ['python', script_path] + [str(arg) for arg in args]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running script {script_path}: {e}")
        return

def read_output(script_name, process, output_queue):
    for line in iter(process.stdout.readline, ''):
        output_queue.put((script_name, line.strip()))
    process.stdout.close()

def run_parallel_scripts(console, script_configs, output_queues, script_titles):
    procs = {}
    threads = {}
    progress_bars = {}
    task_ids = {}

    for script, (desc, args) in script_configs.items():
        proc = subprocess.Popen(
            ["python", script] + [str(arg) for arg in args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        procs[script] = proc
        threads[script] = threading.Thread(target=read_output, args=(script, proc, output_queues[script]), daemon=True)
        threads[script].start()

        progress_bars[script] = Progress(TextColumn("{task.description}"), BarColumn(), TextColumn("[green]{task.completed}/{task.total}"))
        task_ids[script] = {}

    with Live(console=console, refresh_per_second=10) as live:
        while any(proc.poll() is None for proc in procs.values()):
            renderables = []
            for script, q in output_queues.items():
                progress = progress_bars[script]
                while not q.empty():
                    _, line = q.get()
                    try:
                        msg = json.loads(line)
                        task = msg["task"]
                        completed = msg["progress"]
                        total = msg["total"]
                        if task not in task_ids[script]:
                            task_ids[script][task] = progress.add_task(task, total=total)
                        progress.update(task_ids[script][task], completed=completed)
                    
                    except json.JSONDecodeError:
                        continue
                
                renderables.append(Panel(progress, title=script_titles.get(script, script)))
            live.update(Group(*renderables))

    for script, proc in procs.items():
        proc.wait()
        if proc.returncode != 0:
            console.print(f"[bold red]❌ {script} failed with return code {proc.returncode}")

def main(args):
    # Load config
    config_path = args.config
    config = load_config(config_path)
    modality = config['dataset']['modality']

    # Prepare the results directories
    results_folder = 'results' / Path(config['results']['folder'])
    results_folder.mkdir(parents=True, exist_ok=True)
    (results_folder / 'pipeline_files').mkdir(parents=True, exist_ok=True)
    (results_folder / 'pipeline_files' / 'segmentations' / 'ga').mkdir(parents=True, exist_ok=True)
    (results_folder / 'pipeline_files' / 'segmentations' / 'vessels').mkdir(parents=True, exist_ok=True)
    (results_folder / 'pipeline_files' / 'registrations').mkdir(parents=True, exist_ok=True)
    (results_folder / 'pipeline_files' / 'presentations' / 'longitudinal').mkdir(parents=True, exist_ok=True)
    (results_folder / 'pipeline_files' / 'presentations' / 'sequential').mkdir(parents=True, exist_ok=True)

    # Run pipeline scripts
    start_time = time.time()

    # Startup console
    console = Console()

    # BEGIN PIPELINE
    console.rule("[bold bright_magenta]🚀 Starting GA Progression Modeling Pipeline")

    # EXTRACT MODALITY DATA
    console.print(f"[bold yellow]>> 'Extracting {modality} data'")
    modality_csv = results_folder / 'pipeline_files' / 'images_1.csv'
    if not modality_csv.exists():
        args1 = [
            "--csv",
            Path(config['dataset']['folder']) / config['dataset']['csv_path'],
            "--image_col", 
            config['dataset']['image_col'],
            "--modality_col",
            config['dataset']['modality_col'],
            "--modality_val",
            config['dataset']['modality'],
            "--save_as",
            modality_csv,
            # "--n_images", # only for testing
            # 10
        ]
        run_script(script_path='src/pipeline_scripts/1_get_modality_data.py', args=args1)

    # RUN SEGMENTATION PIPELINES
    console.print(f"[bold yellow]>> 'Segmenting GA/Vessels from {modality} data'")
    
    script_configs = {}
    output_queues = {}
    script_titles = {}
    ga_output_csv = results_folder / 'pipeline_files' / 'segmentations' / 'ga_segs.csv'
    if not ga_output_csv.exists():
        script_configs["src/pipeline_scripts/2_segment_ga.py"] = (
            f"Segment GA on {modality} data",
            [
                "--csv", results_folder / 'pipeline_files' / 'images_1.csv',
                "--image_col", config['dataset']['image_col'],
                "--ga_col", config['dataset']['ga_col'],
                "--hist_eq", config['models']['segmentation']['ga']['hist_eq'],
                "--weights_path", config['models']['segmentation']['ga']['weights'],
                "--output_folder", results_folder / 'pipeline_files' / 'segmentations' / 'ga',
                "--device", config['models']['device'],
                "--save_as", ga_output_csv
            ]
        )
        output_queues["src/pipeline_scripts/2_segment_ga.py"] = queue.Queue()
        script_titles["src/pipeline_scripts/2_segment_ga.py"] = "Segment GA"

    vessel_output_csv = results_folder / 'pipeline_files' / 'segmentations' / 'vessel_segs.csv'
    if not vessel_output_csv.exists():
        script_configs['src/pipeline_scripts/3_segment_vessels.py'] = (
            f"Segment Vessels on {modality} data",
            [
                "--csv", results_folder / 'pipeline_files' / 'images_1.csv',
                "--image_col", config['dataset']['image_col'],
                "--hist_eq", config['models']['segmentation']['vessel']['hist_eq'],
                "--weights_path", config['models']['segmentation']['vessel']['weights'],
                "--output_folder", results_folder / 'pipeline_files' / 'segmentations' / 'vessels',
                "--device", config['models']['device'],
                "--save_as", vessel_output_csv
            ]
        )
        output_queues["src/pipeline_scripts/3_segment_vessels.py"] = queue.Queue()
        script_titles["src/pipeline_scripts/3_segment_vessels.py"] = "Segment Vessels"
    run_parallel_scripts(console, script_configs, output_queues, script_titles)
    
    # run_script(
    #     'src/pipeline_scripts/2_segment_ga.py', 
    #     args=script_configs["src/pipeline_scripts/2_segment_ga.py"][1]
    #     )
    # sys.exit(0)

    # MERGE SEGMENTATION PIPELINE RESULTS
    merged_csv = results_folder / 'pipeline_files' / 'images_2.csv'
    if not merged_csv.exists():
        args2 = [
            "--csv",
            results_folder / 'pipeline_files' / 'images_1.csv',
            results_folder / 'pipeline_files' / 'segmentations' / 'ga_segs.csv',
            results_folder / 'pipeline_files' / 'segmentations' / 'vessel_segs.csv',
            "--join_on",
            config['dataset']['image_col'],
            "--save_as",
            merged_csv,
        ]
        run_script(script_path='src/pipeline_scripts/4_merge_datasets.py', args=args2)

    # COMPUTE GA AREAS
    console.print(f"[bold yellow]>> 'Computing GA areas data'")
    areas_csv = results_folder / 'pipeline_files' / 'images_3.csv'
    if not areas_csv.exists():
        args3 = [
            "--csv",
            results_folder / 'pipeline_files' / 'images_2.csv',
            "--ga_col",
            config['dataset']['ga_col'],
            "--size_x_col",
            config['dataset']['size_x_col'],
            "--size_y_col",
            config['dataset']['size_y_col'],
            "--scale_x_col",
            config['dataset']['scale_x_col'],
            "--scale_y_col",
            config['dataset']['scale_y_col'],
            "--save_as",
            areas_csv,
        ]
        run_script(script_path='src/pipeline_scripts/5_compute_ga_area.py', args=args3)

    # REGISTER GA IMAGES
    console.print(f"[bold yellow]>> 'Register {modality} data'")
    registered_csv = results_folder / 'pipeline_files' / 'images_4.csv'
    if not registered_csv.exists():
        args4 = [
            "--csv",
            results_folder / 'pipeline_files' / 'images_3.csv',
            "--img_col",
            config['dataset']['image_col'],
            "--vessel_col",
            config['dataset']['vessel_col'],
            "--pid_col",
            config['dataset']['patient_id_col'],
            "--lat_col",
            config['dataset']['laterality_col'],
            "--sequence_col",
            config['dataset']['examdate_col'],
            "--size",
            config['models']['registration']['size'],
            "--input",
            config['models']['registration']['input'],
            "--reg_method",
            config['models']['registration']['method'],
            "--reg2start",
            config['models']['registration']['reg2start'],
            "--lambda_tps",
            config['models']['registration']['tps'],
            "--device",
            config['models']['device'],
            "--output_folder",
            results_folder / 'pipeline_files' / 'registrations',
            "--save_as",
            registered_csv
        ]
        run_script(script_path='src/pipeline_scripts/6_register_images.py', args=args4)

    # RUN PRESENTATION PIPELINES
    console.print(f"[bold yellow]>> 'Making presentations'")
    script_configs = {}
    output_queues = {}
    script_titles = {}

    # Sequential ppt
    sequential_output_ppt = results_folder / 'pipeline_files' / 'presentations' / 'sequential.pptx'
    if not sequential_output_ppt.exists():
        script_configs['src/pipeline_scripts/7_create_ppt_sequential.py'] = (
            f"Generate Sequntial PPT",
            [
                "--csv", results_folder / 'pipeline_files' / 'images_4.csv',
                "--image_col", config['dataset']['image_col'],
                "--ga_col", config['dataset']['ga_col'],
                "--patient_col", config['dataset']['patient_id_col'],
                "--laterality_col", config['dataset']['laterality_col'],
                "--date_col", config['dataset']['examdate_col'],
                "--area_col", 'mm_area',
                "--output_folder", results_folder / 'pipeline_files' / 'presentations' / 'sequential',
                "--save_as", sequential_output_ppt,
                "--register"
            ]
        )
        output_queues["src/pipeline_scripts/7_create_ppt_sequential.py"] = queue.Queue()
        script_titles["src/pipeline_scripts/7_create_ppt_sequential.py"] = "Generate Sequential PPT"

    # Longitudinal ppt
    longitudinal_output_ppt = results_folder / 'pipeline_files' / 'presentations' / 'longitudinal.pptx'
    if not longitudinal_output_ppt.exists():
        script_configs["src/pipeline_scripts/8_create_ppt_longitudinal.py"] = (
            f"Generate Longitudinal PPT",
            [
                "--csv", results_folder / 'pipeline_files' / 'images_4.csv',
                "--image_col", config['dataset']['image_col'],
                "--ga_col", config['dataset']['ga_col'],
                "--patient_col", config['dataset']['patient_id_col'],
                "--laterality_col", config['dataset']['laterality_col'],
                "--date_col", config['dataset']['examdate_col'],
                "--area_col", 'mm_area',
                "--output_folder", results_folder / 'pipeline_files' / 'presentations' / 'longitudinal',
                "--save_as", longitudinal_output_ppt, 
                *(["--deidentify"] if args.deidentify else []),
                "--gompertz_path", args.gompertz_path
            ]
        )
        output_queues["src/pipeline_scripts/8_create_ppt_longitudinal.py"] = queue.Queue()
        script_titles["src/pipeline_scripts/8_create_ppt_longitudinal.py"] = "Generate Longitudinal PPT"
    
    # run scripts in parallel
    run_parallel_scripts(console, script_configs, output_queues, script_titles)

    # run_script(
    #     'src/pipeline_scripts/8_create_ppt_longitudinal.py', 
    #     args=script_configs["src/pipeline_scripts/8_create_ppt_longitudinal.py"][1]
    #     )
    # sys.exit(0)

    # copy results csv over
    final_csv = results_folder / 'images_processed.csv'
    if not final_csv.exists():
        copy(registered_csv, final_csv)

    # copy sequential presentation over
    seg_ppt = results_folder / 'sequential.pptx'
    if not seg_ppt.exists():
        copy(sequential_output_ppt, seg_ppt)

    # copy sequential presentation over
    long_ppt = results_folder / 'longitudinal.pptx'
    if not long_ppt.exists():
        copy(longitudinal_output_ppt, long_ppt)

    # calculate runtime
    end_time = time.time()
    elapsed = end_time - start_time
    console.print(f"\n[bold green]✅ Finished pipeline in {elapsed:.2f} seconds")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('')
    parser.add_argument('--config', default='src/configs/config.yaml')
    parser.add_argument('--deidentify', action='store_true')
    parser.add_argument('--gompertz_path', default=None)
    args = parser.parse_args()
    main(args)