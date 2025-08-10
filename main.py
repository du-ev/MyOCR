from preprocess_datasets import create_combined_data

force_regenerate_data = True

def main():
    create_combined_data(force_regenerate_data)

if __name__ == "__main__":
    main()