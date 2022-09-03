import csv
import numpy as np


def export_mini_dataset(
    dataset_path, mini_dataset_path, data_limit=10, file_format="csv"
):
    with open(dataset_path, "r") as old_file:
        reader = csv.reader(old_file)

        with open(f"{mini_dataset_path}.{file_format}", "w") as new_file:
            writer = csv.writer(new_file, delimiter=",")

            for idx, line in enumerate(reader):
                writer.writerow(line)
                if idx == data_limit:
                    break

    print("Sucessfully exported as mini file")


def load_az_dataset(dataset_path):
    data = []
    labels = []

    with open(dataset_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file)

        for line in csv_reader:
            label = int(line[0])
            image = np.array([int(x) for x in line[1:]], dtype="uint8")
            image = image.reshape((28, 28))

            data.append(image)
            labels.append(label)

    data = np.array(data, dtype="float32")
    labels = np.array(labels, dtype="int")

    return (data, labels)
