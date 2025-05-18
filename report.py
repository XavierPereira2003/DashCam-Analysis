import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd

def extract_tensorboard_data(log_dir):
    """
    Extract scalar data from TensorBoard log files
    
    Args:
        log_dir: Path to TensorBoard log directory
        
    Returns:
        Dictionary of tag names and their scalar values
    """
    # Load the event data
    ea = event_accumulator.EventAccumulator(
        log_dir,
        size_guidance={
            event_accumulator.SCALARS: 0,  # Load all scalar data
            event_accumulator.TENSORS: 0,  # Load all tensor data
        }
    )
    ea.Reload()  # Load all data

    # Get list of tags (metrics)
    tags = ea.Tags()['scalars']
    
    # Extract scalar data
    scalar_data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        steps = [event.step for event in events]
        values = [event.value for event in events]
        scalar_data[tag] = {'steps': steps, 'values': values}
    
    return scalar_data, tags

def export_to_pdf(log_dir, title, include_avg=False, dpi=150):
    """
    Export TensorBoard data to PDF
    
    Args:
        log_dir: Path to TensorBoard log directory or list of directories
        output_pdf: Output PDF file path
        include_avg: Whether to include average plots when multiple logs are provided
        dpi: DPI for plot images
    """
    # Handle single directory or list of directories
    multiple_logs = False
    if isinstance(log_dir, list):
        if len(log_dir) > 1:
            multiple_logs = True
        else:
            log_dir = log_dir[0]
    
    # Create PDF
    with PdfPages(title+".pdf") as pdf:
        # Title page
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        title_text = title
        plt.text(0.5, 0.5, title_text, horizontalalignment='center',
                fontsize=24, fontweight='bold')
        if isinstance(log_dir, list):
            log_paths = '\n'.join(log_dir)
        else:
            log_paths = log_dir
        plt.text(0.5, 0.4, f"Log Sources:\n{log_paths}", horizontalalignment='center',
                fontsize=12)
        plt.tight_layout()
        pdf.savefig(dpi=dpi)
        plt.close()
        
        if multiple_logs:
            # Process multiple logs
            all_data = []
            all_tags = set()
            
            # First collect all data and identify common tags
            for directory in log_dir:
                data, tags = extract_tensorboard_data(directory)
                all_data.append(data)
                all_tags.update(tags)
            
            # Sort tags to ensure consistent order
            all_tags = sorted(all_tags)
            
            # Create a plot for each tag, showing all runs
            for tag in all_tags:
                plt.figure(figsize=(10, 6))
                legend_labels = []
                
                # Plot individual runs
                for i, (data, log_path) in enumerate(zip(all_data, log_dir)):
                    if tag in data:
                        plt.plot(data[tag]['steps'], data[tag]['values'])
                        legend_labels.append(os.path.basename(log_path))
                
                # Plot average if requested and if we have more than one run with this tag
                if include_avg:
                    tag_exists = [tag in data for data in all_data]
                    if sum(tag_exists) > 1:
                        # Need to interpolate to get same x-axis for averaging
                        # Get all steps from all runs
                        all_steps = set()
                        for data in all_data:
                            if tag in data:
                                all_steps.update(data[tag]['steps'])
                        all_steps = sorted(list(all_steps))
                        
                        # Interpolate values for each run
                        interp_values = []
                        for data in all_data:
                            if tag in data:
                                steps = data[tag]['steps']
                                values = data[tag]['values']
                                interp_values.append(np.interp(
                                    all_steps, 
                                    steps, 
                                    values, 
                                    left=np.nan, 
                                    right=np.nan
                                ))
                        
                        # Calculate average ignoring NaNs
                        avg_values = np.nanmean(interp_values, axis=0)
                        plt.plot(all_steps, avg_values, 'k--', linewidth=2)
                        legend_labels.append('Average')
                
                plt.title(f'Metric: {tag}')
                plt.xlabel('Step')
                plt.ylabel('Value')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(legend_labels)
                plt.tight_layout()
                pdf.savefig(dpi=dpi)
                plt.close()
                
        else:
            # Process single log directory
            data, tags = extract_tensorboard_data(log_dir)
            
            # Create a plot for each tag
            for tag in sorted(tags):
                plt.figure(figsize=(10, 6))
                plt.plot(data[tag]['steps'], data[tag]['values'])
                plt.title(f'Metric: {tag}')
                plt.xlabel('Step')
                plt.ylabel('Value')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                pdf.savefig(dpi=dpi)
                plt.close()
                
        # Create a summary table
        if multiple_logs:
            # Create summary of final values for each run and metric
            plt.figure(figsize=(12, len(all_tags) * 0.5 + 2))
            plt.axis('off')
            
            cell_text = []
            rows = all_tags
            columns = ['Run'] + [os.path.basename(path) for path in log_dir]
            
            for tag in all_tags:
                row = [tag]
                for data in all_data:
                    if tag in data:
                        # Get the last value
                        last_value = data[tag]['values'][-1]
                        row.append(f"{last_value:.4f}")
                    else:
                        row.append("N/A")
                cell_text.append(row)
            
            table = plt.table(
                cellText=cell_text,
                colLabels=columns,
                loc='center',
                cellLoc='center',
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
            plt.title("Summary of Final Values", fontsize=16, pad=20)
            plt.tight_layout()
            pdf.savefig(dpi=dpi)
            plt.close()
            
        # Add summary stats to PDF
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0.5, 0.95, "Export Summary", fontsize=16, fontweight='bold', ha='center')
        
        if multiple_logs:
            summary_text = f"- Number of runs: {len(log_dir)}\n"
            summary_text += f"- Total metrics: {len(all_tags)}\n"
            summary_text += f"- Metrics exported: {', '.join(all_tags[:5])}"
            if len(all_tags) > 5:
                summary_text += f" and {len(all_tags) - 5} more"
        else:
            summary_text = f"- Log path: {log_dir}\n"
            summary_text += f"- Total metrics: {len(tags)}\n"
            summary_text += f"- Metrics exported: {', '.join(sorted(tags)[:5])}"
            if len(tags) > 5:
                summary_text += f" and {len(tags) - 5} more"
                
        plt.text(0.5, 0.5, summary_text, fontsize=12, ha='center', va='center')
        plt.tight_layout()
        pdf.savefig(dpi=dpi)
        plt.close()
        
    print(f"PDF report generated successfully: {title}")
    

def main():
    parser = argparse.ArgumentParser(description='Export TensorBoard logs to PDF')
    parser.add_argument('--logdir', type=str, nargs='+', required=True,
                        help='Path(s) to TensorBoard log directory')
    parser.add_argument('--output', type=str, default='tensorboard_export',
                        help='Output PDF file path')
    parser.add_argument('--include-avg', action='store_true',
                        help='Include average plots when multiple logs are provided')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI for exported figures')
    
    args = parser.parse_args()
    
    export_to_pdf(args.logdir, args.output, args.include_avg, args.dpi)

if __name__ == '__main__':
    main()