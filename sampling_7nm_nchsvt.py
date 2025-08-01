import os
import re
import csv
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from scipy.stats import qmc
import random
import itertools

# ================= Configuration Section =================
CONFIG = {
    "LIB_PATH": r"/home/shenshan/mos_model_nn/FinFet_pdk/Model_all/models/cln7_1d8_sp_v1d2_2p2.l",
    "PARAM_DATA": "/home/shenshan/mos_model_nn/FinFet_data/SIM/param_data.dat",
    "CONVERTED_PARAM": "/home/shenshan/mos_model_nn/FinFet_data/SIM/converted_param.dat",
    "HSPICE_SCRIPT": "/home/shenshan/mos_model_nn/FinFet_data/SIM/generated_simulation.sp",
    "HSPICE_OUTPUT": "/home/shenshan/mos_model_nn/FinFet_data/SIM/transistor.lis",
    "FINAL_CSV": "/home/shenshan/mos_model_nn/FinFet_data/SIM/output.csv",
    "NUM_SAMPLES": 1400,  # Number of samples required
    "VGS_SAMPLES": 20,  # Number of vgs samples
    "VDS_SAMPLES": 20,  # Number of vds samples
    "VBS_SAMPLES": 20,  # Number of vbs samples
    "LENGTH_SAMPLES": 40,  # Number of length samples
    "SCALE_FACTOR": 1,
    "HSPICE_CMD": "hspice -i {} -o {}",
    "MODEL_PARAMS": ['PHIG', 'CFS', 'TOXP', 'CGSL', 'CIT', 'U0', 'UA', 'EU', 'ETA0'],
    "FIELD_ORDER": [
        'group', 'model',  # Add model name field
        'PHIG', 'CFS', 'TOXP', 'CGSL', 'CIT', 'U0', 'UA', 'EU', 'ETA0',
        'vgs', 'vds', 'vbs', 'length', 'nfin',
        'temp', 'qg', 'qb', 'qd', 'ids', 'didsdvg', 'didsdvd',
        'cgg', 'cgd', 'cgs'
    ]
}


def main():
    try:
        # First parse the model library
        model_ranges = parse_model_lib(CONFIG["LIB_PATH"])
        print(f"✅ Model library parsed, found {len(model_ranges)} model definitions")

        # Pass model ranges to data generation function
        gen_data_samples(
            num_samples=CONFIG["NUM_SAMPLES"],
            voltage_bounds=([0, 0, -1], [1.0, 1, 1]),  # vgs, vds, vbs ranges
            model_ranges=model_ranges,
            vgs_samples=CONFIG["VGS_SAMPLES"],
            vds_samples=CONFIG["VDS_SAMPLES"],
            vbs_samples=CONFIG["VBS_SAMPLES"],
            length_samples=CONFIG["LENGTH_SAMPLES"]
        )

        convert_para_data(CONFIG["PARAM_DATA"], CONFIG["CONVERTED_PARAM"])
        generate_hspice_script(CONFIG["CONVERTED_PARAM"], CONFIG["HSPICE_SCRIPT"])

        code = os.system(CONFIG["HSPICE_CMD"].format(
            CONFIG["HSPICE_SCRIPT"],
            CONFIG["HSPICE_OUTPUT"]
        ))
        if code != 0:
            raise RuntimeError("HSPICE simulation failed")

        # Pass model ranges to result processing function
        process_simulation_results(
            CONFIG["CONVERTED_PARAM"],
            CONFIG["HSPICE_OUTPUT"],
            CONFIG["FINAL_CSV"],
            model_ranges
        )
        print("🎉 Automation process completed! Results saved to:", CONFIG["FINAL_CSV"])

    except Exception as e:
        print(f"❌ Process execution failed: {str(e)}")
        raise


def parse_model_lib(lib_path):
    """Parse .lib file to extract all models and their L/NFIN ranges"""
    model_ranges = {}
    current_model = None

    try:
        with open(lib_path, 'r') as f:
            for line in f:
                # Only parse nch_svt_mac models
                model_match = re.search(r'\.model\s+(nch_svt_mac\.\d+)\s+nmos\s*\(', line)
                if model_match:
                    current_model = model_match.group(1)
                    model_ranges[current_model] = {
                        'lmin': None,
                        'lmax': None,
                        'nfinmin': None,
                        'nfinmax': None
                    }
                    continue

                # Process only current model
                if current_model:
                    # Extract L range
                    lmin_match = re.search(r'lmin\s*=\s*([\d\.eE+-]+)', line)
                    if lmin_match:
                        model_ranges[current_model]['lmin'] = float(lmin_match.group(1))

                    lmax_match = re.search(r'lmax\s*=\s*([\d\.eE+-]+)', line)
                    if lmax_match:
                        model_ranges[current_model]['lmax'] = float(lmax_match.group(1))

                    # Extract NFIN range
                    nfinmin_match = re.search(r'nfinmin\s*=\s*(\d+)', line)
                    if nfinmin_match:
                        model_ranges[current_model]['nfinmin'] = int(nfinmin_match.group(1))

                    nfinmax_match = re.search(r'nfinmax\s*=\s*(\d+)', line)
                    if nfinmax_match:
                        model_ranges[current_model]['nfinmax'] = int(nfinmax_match.group(1))

    except Exception as e:
        raise RuntimeError(f"Failed to parse model library: {str(e)}")

    # Filter out models with incomplete ranges
    valid_models = {}
    for model, ranges in model_ranges.items():
        if None not in (ranges['lmin'], ranges['lmax'], ranges['nfinmin'], ranges['nfinmax']):
            valid_models[model] = ranges
        else:
            print(f"⚠️ Skipping incomplete model: {model} (missing range values)")

    # Print parsing results for debugging
    print("Parsed valid model ranges:")
    for model, ranges in valid_models.items():
        print(f"{model}: L=[{ranges['lmin']:.2e}, {ranges['lmax']:.2e}], "
              f"NFIN=[{ranges['nfinmin']}, {ranges['nfinmax']}]")

    return valid_models


def gen_data_samples(num_samples, voltage_bounds, model_ranges,
                     vgs_samples, vds_samples, vbs_samples, length_samples):
    """
    Generate data samples - strictly follow process rules

    Parameters:
    num_samples: Final number of samples needed
    voltage_bounds: Voltage ranges (vgs_min, vds_min, vbs_min), (vgs_max, vds_max, vbs_max)
    model_ranges: Model range dictionary
    vgs_samples: vgs sample count
    vds_samples: vds sample count
    vbs_samples: vbs sample count
    length_samples: length sample count
    """
    # 1. Generate LHS samples for independent dimensions
    print(f"📊 Generating independent dimension samples:")
    print(f"  vgs: {vgs_samples} samples, range: [{voltage_bounds[0][0]}, {voltage_bounds[1][0]}]")
    print(f"  vds: {vds_samples} samples, range: [{voltage_bounds[0][1]}, {voltage_bounds[1][1]}]")
    print(f"  vbs: {vbs_samples} samples, range: [{voltage_bounds[0][2]}, {voltage_bounds[1][2]}]")
    print(f"  length: {length_samples} samples")

    # Generate voltage samples
    vgs_values = generate_lhs_samples(vgs_samples, voltage_bounds[0][0], voltage_bounds[1][0])
    vds_values = generate_lhs_samples(vds_samples, voltage_bounds[0][1], voltage_bounds[1][1])
    vbs_values = generate_lhs_samples(vbs_samples, voltage_bounds[0][2], voltage_bounds[1][2])

    # Generate length samples (mix of fixed points and continuous points)
    length_values = generate_length_samples(length_samples, model_ranges)

    # 2. Generate all possible combinations
    all_combinations = list(itertools.product(vgs_values, vds_values, vbs_values, length_values))
    total_combinations = len(all_combinations)
    print(
        f"🔢 Generated all combinations: {vgs_samples}×{vds_samples}×{vbs_samples}×{length_samples} = {total_combinations} combinations")

    # 3. Generate corresponding nfin for each combination
    print("🔧 Generating nfin for each combination...")
    full_params = []
    for vgs, vds, vbs, length in tqdm(all_combinations, desc="Generating combinations", unit="combinations"):
        # Find models matching the length
        matched_models = []
        for model_name, ranges in model_ranges.items():
            lmin = ranges['lmin']
            lmax = ranges['lmax']
            if lmin <= length <= lmax:
                matched_models.append((model_name, ranges))

        if matched_models:
            # Randomly select a matching model
            selected_model, model_range = random.choice(matched_models)
            nfinmin = model_range['nfinmin']
            nfinmax = model_range['nfinmax']

            # Ensure NFIN is within process limits
            nfinmin = max(2, nfinmin)  # Minimum 2 fins
            nfinmax = min(20, nfinmax)  # Maximum 20 fins

            # Special handling: For L=0.008/0.011um, allow NFIN=20-24
            if length in [0.008e-6, 0.011e-6]:
                nfinmax = min(24, nfinmax)  # Extend upper limit to 24

            nfin = random.randint(nfinmin, nfinmax)
        else:
            # If no matching model, use safe range [2,20]
            nfin = random.randint(2, 20)

        full_params.append([vgs, vds, vbs, length, nfin])

    # 4. Perform LHS sampling from all combinations
    if total_combinations < num_samples:
        print(
            f"⚠️ Warning: Total combinations ({total_combinations}) less than required samples ({num_samples}), using all combinations")
        selected_indices = range(total_combinations)
    else:
        # Create LHS sampler
        sampler = qmc.LatinHypercube(d=1)
        # Generate samples in [0,1) range
        samples = sampler.random(n=num_samples)
        # Map samples to combination index range [0, total_combinations-1]
        selected_indices = (samples.flatten() * (total_combinations - 1)).astype(int)
        # Ensure indices are unique and sorted
        selected_indices = np.unique(selected_indices)

    selected_params = [full_params[i] for i in selected_indices]
    print(f"🎯 Sampled {len(selected_params)} samples from {total_combinations} combinations using LHS")

    # 5. Save data
    os.makedirs(os.path.dirname(CONFIG["PARAM_DATA"]), exist_ok=True)
    with open(CONFIG["PARAM_DATA"], "w") as data_f:
        data_f.write(".DATA para\n")
        data_f.write("+ vgs vds vbs length nfin\n")
        for params in selected_params:
            # Special handling: nfin as integer
            formatted_params = [
                f"{params[0]:.3e}",
                f"{params[1]:.3e}",
                f"{params[2]:.3e}",
                f"{params[3]:.3e}",
                f"{int(params[4])}"  # Write nfin as integer
            ]
            data_f.write(f"+ {' '.join(formatted_params)}\n")
        data_f.write(".ENDDATA\n")
    print(f"💾 Parameter data saved to: {CONFIG['PARAM_DATA']}")


def generate_lhs_samples(n_samples, low, high):
    """Generate 1D LHS samples"""
    sampler = qmc.LatinHypercube(d=1)
    samples = sampler.random(n=n_samples)
    return qmc.scale(samples, [low], [high]).flatten()


def generate_length_samples(n_samples, model_ranges):
    """Generate length samples - mix of fixed points and continuous points"""
    # Fixed L values (8nm/11nm/20nm/36nm)
    fixed_L_values = [0.008e-6, 0.011e-6, 0.020e-6, 0.036e-6]
    num_fixed = int(n_samples * 0.2)  # 20% fixed L values
    num_fixed_per_value = max(1, num_fixed // len(fixed_L_values))  # Samples per fixed value

    # Continuous L range (72nm-240nm)
    continuous_L_min, continuous_L_max = 0.072e-6, 0.240e-6
    num_continuous = n_samples - num_fixed_per_value * len(fixed_L_values)  # Remaining 80% continuous L

    # 1. Generate fixed L samples
    fixed_L_samples = []
    for l in fixed_L_values:
        fixed_L_samples.extend([l] * num_fixed_per_value)

    # 2. Generate continuous L samples (using Latin Hypercube Sampling)
    if num_continuous > 0:
        l_sampler = qmc.LatinHypercube(d=1)
        l_samples = l_sampler.random(n=num_continuous)
        continuous_L_samples = qmc.scale(l_samples, [continuous_L_min], [continuous_L_max]).flatten()
    else:
        continuous_L_samples = []

    # 3. Combine all L values
    all_L_values = np.concatenate([fixed_L_samples, continuous_L_samples])

    # Ensure correct sample count
    if len(all_L_values) != n_samples:
        # If insufficient, supplement with continuous samples
        additional_needed = n_samples - len(all_L_values)
        if additional_needed > 0:
            l_sampler = qmc.LatinHypercube(d=1)
            additional_samples = l_sampler.random(n=additional_needed)
            additional_L = qmc.scale(additional_samples, [continuous_L_min], [continuous_L_max]).flatten()
            all_L_values = np.concatenate([all_L_values, additional_L])

    return all_L_values


def find_nfin_range_for_length(length, model_ranges):
    """Find matching NFIN range based on length"""
    nfin_ranges = []

    for model, ranges in model_ranges.items():
        lmin = ranges['lmin']
        lmax = ranges['lmax']
        nfinmin = ranges['nfinmin']
        nfinmax = ranges['nfinmax']

        if lmin <= length <= lmax:
            nfin_ranges.append((nfinmin, nfinmax))

    if not nfin_ranges:
        return None

    # Merge all matching NFIN ranges
    min_nfin = min(r[0] for r in nfin_ranges)
    max_nfin = max(r[1] for r in nfin_ranges)

    return (min_nfin, max_nfin)


def convert_para_data(input_file, output_file):
    param_names = ['vgs', 'vds', 'vbs', 'length', 'nfin']
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    start_index = None
    end_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith('.DATA'):
            start_index = i + 2
        elif line.strip() == '.ENDDATA':
            end_index = i
            break

    data_lines = lines[start_index:end_index]
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as outfile:
        for idx, line in enumerate(data_lines, 1):
            values = line.strip().split()[1:]
            for name, value in zip(param_names, values):
                outfile.write(f".param {name}{idx}={value}\n")
            outfile.write("\n")


def generate_hspice_script(data_file, output_file):
    """Generate HSPICE script - remove width parameter"""
    # Fixed template section
    header = f'''*HSPICE*
*** LIB ***
.lib '/home/shenshan/mos_model_nn/FinFet_pdk/Model_all/models/cln7_1d8_sp_v1d2_2p2_usage.l'  TTMacro_MOS_MOSCAP
.lib '/home/shenshan/mos_model_nn/FinFet_pdk/Model_all/models/cln7_1d8_sp_v1d2_2p2_usage.l'  pre_simu

.inc '{data_file}'  // Include parameter definition file

.TEMP=25  // Initial temperature setting (will be overridden by .step)

*** INSTANCES ***\n'''

    voltage_sources_template = '''
*** V sources for each instance ***{voltage_sources}'''

    print_statements_template = '''
*** PROBE ***{print_statements}'''

    temperature_command = '''
*** STEP COMMAND FOR TEMPERATURE ***
.dc temp -25 125 50  // Temperature sweep range: -25°C to 125°C, step 50°C'''

    footer = '''
*** OPTIONS ***
.option list post=2 INGOLD=2

.end'''

    try:
        with open(data_file, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        raise Exception(f'❌ Failed to read converted parameter file: {str(e)}')

    instances = []
    voltages = []
    prints = []
    param_dict = {}

    for line in lines:
        if line.startswith('.param'):
            key_value = line.split()[1].split('=')
            param_dict[key_value[0]] = key_value[1]

    indices = sorted({int(''.join(filter(str.isdigit, k))) for k in param_dict if k[-1].isdigit()})

    for i in indices:
        # Instantiate using only L and NFIN parameters, remove W parameter
        instances.append(f"xmdut{i} D{i} G{i} S{i} B{i} nch_svt_mac L=length{i} NFIN=nfin{i}")

        # Voltage sources
        voltages.append(f'''
VD{i} D{i} 0 dc vds{i}
VS{i} S{i} 0 dc 0
VG{i} G{i} 0 dc vgs{i}
VB{i} B{i} 0 dc vbs{i}''')

        # Print outputs
        prints.append(f'''
*** data name = para , index = {i} ***
.print QG(xmdut{i}) QB(xmdut{i}) QD(xmdut{i}) IDS(xmdut{i}) DIDSDVG(xmdut{i}) DIDSDVD(xmdut{i}) CGG(xmdut{i}) CGD(xmdut{i}) CGS(xmdut{i})''')

    # Assemble complete script
    full_script = (
            header +
            '\n'.join(instances) + '\n\n' +
            voltage_sources_template.format(voltage_sources='\n'.join(voltages)) + '\n\n' +
            temperature_command + '\n\n' +
            print_statements_template.format(print_statements='\n'.join(prints)) + '\n\n' +
            footer
    )

    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(full_script)
        print(f'✅ HSPICE script generated successfully: {output_file}')
    except Exception as e:
        raise Exception(f'❌ Failed to write HSPICE script: {str(e)}')


def parse_param_data(file_path):
    params = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

    current_group = {}
    for line in lines:
        match = re.match(r'\.param\s+(\w+)=(.+)', line.strip())
        if match:
            key = match.group(1)
            value = match.group(2).strip()
            base_key = ''.join(filter(str.isalpha, key))
            num_suffix = ''.join(filter(str.isdigit, key))
            group_num = int(num_suffix) if num_suffix else len(params) + 1

            if not current_group or current_group.get('group') != group_num:
                current_group = {'group': group_num}
                params.append(current_group)

            if base_key == 'nfin':
                current_group['nfin'] = int(float(value))
            else:
                current_group[base_key] = value

    return params


def parse_lis_data_blocks(content):
    """Parse simulation data blocks in HSPICE output file"""
    pattern = r'x\s*\n\s*\n(.*?)\n\s*y'
    blocks = re.findall(pattern, content, flags=re.DOTALL)

    data_dict = defaultdict(dict)

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue

        headers = [h.strip() for h in lines[0].split()]
        device_line = lines[1].strip()
        group_match = re.search(r'xmdut(\d+)', device_line)
        if not group_match:
            continue
        group = int(group_match.group(1))

        for data_line in lines[2:]:
            if not data_line.strip():
                continue

            values = data_line.split()
            if not values:
                continue

            try:
                temp = float(values[0])
            except ValueError:
                continue

            for i in range(1, min(len(values), len(headers))):
                key = headers[i].lower()
                try:
                    value = float(values[i])
                    data_dict[(group, temp)][key] = value
                except (ValueError, IndexError):
                    data_dict[(group, temp)][key] = values[i] if i < len(values) else 'N/A'

    result = []
    for (group, temp), data in data_dict.items():
        row = {'group': group, 'temp': temp}
        row.update(data)
        result.append(row)

    return result


def process_simulation_results(param_file, lis_file, output_csv, model_ranges):
    """Process simulation results - match parameters based on model ranges"""
    params = parse_param_data(param_file)

    try:
        with open(lis_file, 'r', encoding='utf-8') as f:
            lis_content = f.read()
    except Exception as e:
        raise Exception(f'❌ Failed to read simulation result file: {str(e)}')

    try:
        model_params = extract_model_params_from_lis(lis_content)
        if not model_params:
            raise Exception("No model parameters extracted")
    except Exception as e:
        raise Exception(f'❌ Failed to extract model parameters: {str(e)}')

    lis_data = parse_lis_data_blocks(lis_content)
    param_by_group = {p['group']: p for p in params}

    combined = []
    match_stats = defaultdict(int)

    for item in tqdm(lis_data, desc="Processing simulation results"):
        group = item['group']
        if group not in param_by_group:
            continue

        instance_params = param_by_group[group]
        length = float(instance_params['length'])
        nfin = int(instance_params['nfin'])

        matched_model = find_matching_model(length, nfin, model_ranges)

        row = {
            'group': group,
            'temp': item['temp'],
            'vgs': instance_params['vgs'],
            'vds': instance_params['vds'],
            'vbs': instance_params['vbs'],
            'length': length,
            'nfin': nfin,
            'model': matched_model or 'N/A'
        }

        if matched_model and matched_model in model_params:
            match_stats['matched'] += 1
            for param in CONFIG["MODEL_PARAMS"]:
                param_found = False
                for key in model_params[matched_model]:
                    if key.lower() == param.lower():
                        row[param] = model_params[matched_model][key]
                        param_found = True
                        break
                if not param_found:
                    row[param] = 'N/A'
        else:
            match_stats['unmatched'] += 1
            for param in CONFIG["MODEL_PARAMS"]:
                row[param] = 'N/A'

        for key in ['qg', 'qb', 'qd', 'ids', 'didsdvg', 'didsdvd', 'cgg', 'cgd', 'cgs']:
            row[key] = item.get(key, 'N/A')

        combined.append(row)

    print(f"\n🔍 Model matching statistics:")
    print(f"  Matched: {match_stats['matched']} instances")
    print(f"  Unmatched: {match_stats['unmatched']} instances")

    try:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=CONFIG["FIELD_ORDER"])
            writer.writeheader()
            for row in tqdm(combined, desc="Writing CSV", unit="rows"):
                writer.writerow(row)
        print(f'✅ Results saved to: {output_csv}')
    except Exception as e:
        raise Exception(f'❌ Failed to write CSV file: {str(e)}')


def extract_model_params_from_lis(content):
    """Extract model parameters from .lis file"""
    model_params = {}
    current_model = None
    param_block = False

    for line in content.splitlines():
        if "Model Name:" in line:
            match = re.search(r'Model Name:\s*(\w+\.\d+)', line)
            if match:
                current_model = match.group(1)
                model_params[current_model] = {}
                param_block = True
                continue

        if param_block and current_model:
            if re.search(r'^\s*$', line) or "Model Name:" in line:
                param_block = False
                if "Model Name:" in line:
                    match = re.search(r'Model Name:\s*(\w+\.\d+)', line)
                    if match:
                        current_model = match.group(1)
                        model_params[current_model] = {}
                        continue

            match = re.match(r'^\s*(\w+)\s*=\s*([\d\.Ee+-]+)', line.strip())
            if match:
                key = match.group(1)
                value = match.group(2)
                try:
                    if 'e' in value.lower():
                        model_params[current_model][key] = float(value)
                    elif '.' not in value:
                        model_params[current_model][key] = int(value)
                    else:
                        model_params[current_model][key] = float(value)
                except ValueError:
                    model_params[current_model][key] = value

    print(f"✅ Extracted parameters for {len(model_params)} models")
    for model, params in model_params.items():
        print(f"  {model}: {list(params.keys())}")

    return model_params


def find_matching_model(length, nfin, model_ranges):
    """Find matching model name based on length and NFIN"""
    for model_name, ranges in model_ranges.items():
        lmin = ranges.get('lmin')
        lmax = ranges.get('lmax')
        nfinmin = ranges.get('nfinmin')
        nfinmax = ranges.get('nfinmax')

        if None in (lmin, lmax, nfinmin, nfinmax):
            continue

        if (lmin <= length <= lmax) and (nfinmin <= nfin <= nfinmax):
            print(f"Matched model: {model_name} (L={length:.2e}, NFIN={nfin}) "
                  f"in range L[{lmin:.2e}-{lmax:.2e}], NFIN[{nfinmin}-{nfinmax}]")
            return model_name

    print(f"⚠️ No matching model found: L={length:.2e}, NFIN={nfin}")
    return None


if __name__ == '__main__':
    os.makedirs(os.path.dirname(CONFIG["PARAM_DATA"]), exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG["CONVERTED_PARAM"]), exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG["HSPICE_SCRIPT"]), exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG["FINAL_CSV"]), exist_ok=True)
    main()