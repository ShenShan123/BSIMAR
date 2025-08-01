import csv


def read_csv(csv_path: str):
    """
    Read CSV file and return the data in the file according to specified rules.
    Args:
        csv_path: path to the CSV file.
    """
    try:
        with open(csv_path, mode='r') as file:
            csv_reader = csv.DictReader(file)

            input_head = ['PHIG', 'CFS', 'TOXP', 'CGSL',  'CIT', 'U0', 'UA', 'EU','ETA0', 'vgs', 'vds', 'vbs', 'length', 'nfin',
                          'temp']
            output_head = ['qg', 'qb', 'qd', 'ids', 'didsdvg', 'didsdvd', 'cgg', 'cgd', 'cgs']

            input_sequence_reshape = []
            output_sequence_print = []

            for row in csv_reader:
                # Extracting inputs and outputs based on headers
                input_row = [float(row[key]) for key in input_head]
                output_row = [float(row[key]) for key in output_head]

                input_sequence_reshape.append(input_row)
                output_sequence_print.append(output_row)

            print("total num of sequences:", len(input_sequence_reshape))
            print("input sequence length:", len(input_sequence_reshape[0]), input_head)
            print("output sequence length:", len(output_sequence_print[0]), output_head)
            return input_head, input_sequence_reshape, output_head, output_sequence_print

    except Exception as e:
        raise Exception(f"Failed in reading CSV file. Error: {e}")


if __name__ == '__main__':
    read_csv("/data/huangl/py_projects/mos_model_nn/SIM/output.csv")