import os
import json
import csv

cro_dict = {
    1: 'S',
    2: 'MS_0',
    3: 'MS_1',
    4: 'MS_2',
    5: 'M2',
}
revin_dict = {
    0: 'False',
    1: 'True'
}


def update_min_list(min_list, value, file_path):
    """
    Update the list storing the minimum values
    """
    if len(min_list) < 1:
        min_list.append((value, file_path))
        min_list.sort(key=lambda x: x[0])
    else:
        if value < min_list[-1][0]:
            i = len(min_list) - 2
            while i >= 0 and min_list[i][0] > value:
                min_list[i + 1] = min_list[i]
                i -= 1
            min_list[i + 1] = (value, file_path)


def get_min_mae_mse_weighted(seed, task, alpha=1.0, cro='S', revin=False):
    seq_len, pre_len = task

    path_str = f"./test_dict/{seed}/{seq_len}->{pre_len}/a1_{alpha}/{cro}/revin_{revin_dict[revin]}"

    sub_folders = [os.path.join(path_str, sub_path) for sub_path in os.listdir(path_str)]

    min_mae_files = []
    min_mse_files = []

    for folder in sub_folders:
        for root, dirs, files in os.walk(folder):
            for file_name in files:
                if not file_name.endswith('.json'):
                    continue
                file_path = os.path.join(root, file_name)
                try:
                    with open(file_path, "r") as f:
                        json_data = json.load(f)
                except Exception as e:
                    # Skip and continue if there's an error reading the file
                    continue

                mae = json_data.get("mae")
                mse = json_data.get("mse")

                if mae is not None:
                    update_min_list(min_mae_files, mae, file_path)

                if mse is not None:
                    update_min_list(min_mse_files, mse, file_path)

    best_mae_file = min_mae_files[0]
    best_mae = best_mae_file[0]
    best_mae_file = best_mae_file[1].split(path_str)[1]

    best_mse_file = min_mse_files[0]
    best_mse = best_mse_file[0]
    best_mse_file = best_mse_file[1].split(path_str)[1]

    return best_mae, best_mae_file, best_mse, best_mse_file


if __name__ == "__main__":
    csv_filename = "uptst_results.csv"

    random_seeds = [2021, 2022, 2023, 2024]
    seq_opt = [48, 96, 192]
    pre_opt = [7, 14, 28, 84]

    with open(csv_filename, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        header = [
            "Tasks", "Recon", "Pre_len", "MSE",
            "MAE",
        ]
        csv_writer.writerow(header)

        for i in range(5):
            for alpha_1_list in [[0.7, 0.6, 0.5, 0.4, 0.3],[1.0]]:
                for pre in pre_opt:
                    avg_mae_list = []
                    avg_mse_list = []
                    for seed in random_seeds:
                        best_mae, best_mse = 1e+4, 1e+4
                        for alpha_1 in alpha_1_list:
                            for seq in seq_opt:
                                task = [seq, pre]
                                mae, _, mse, _ = get_min_mae_mse_weighted(seed=seed, task=task, alpha=alpha_1,
                                                                           cro=cro_dict[i + 1], )
                                if best_mae > mae:
                                    best_mae = mae

                                if best_mse > mse:
                                    best_mse = mse
                        avg_mae_list.append(best_mae)
                        avg_mse_list.append(best_mse)

                    avg_mae = sum(avg_mae_list) / len(avg_mae_list)
                    avg_mse = sum(avg_mse_list) / len(avg_mse_list)
                        # Write the current iteration's results to the CSV file
                    csv_row = [
                            cro_dict[i + 1],
                            'w r' if alpha_1_list != [1.0] else 'w/o r',
                            pre,
                            float(round(avg_mse,4)),  # Changed to avg_mse
                            float(round(avg_mae,4)),  # Changed to avg_mae
                        ]
                    csv_writer.writerow(csv_row)
