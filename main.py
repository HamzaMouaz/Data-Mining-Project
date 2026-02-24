from soil_fertility.components.data_ingestion import DataIngestion
from soil_fertility.components.data_transformation.data_transformation import (
    DataTransformation,
)
from soil_fertility.components.data_transformation.data_transformation_part2 import (
    DataTransformationTwo,
)
from soil_fertility.components.data_transformation.data_transformation_part3 import (
    DataTransformationThree,
)


def run_full_pipeline():
    """Run the complete data processing pipeline"""

    # Step 1: Data Ingestion for Dataset1
    print("Starting Dataset1 processing...")
    ingestion1 = DataIngestion()
    part, train_path1, test_path1 = ingestion1.init_ingestion(
        path="data/Dataset1.csv", option="csv"
    )
    # Step 4: Additional Transformations (Part 2)
    print("Running Part 2 transformations...")
    transformer2 = DataTransformation()
    values = transformer2.transform(train_path=train_path1, test_path=test_path1)

    # Step 3: Data Ingestion for Dataset3
    print("Starting Dataset3 processing...")
    ingestion3 = DataIngestion()
    part, train_path3, test_path3 = ingestion3.init_ingestion(
        path="data/Dataset3.xlsx", option="xlsx"
    )

    # Step 5: Additional Transformations (Part 3)
    print("Running Part 3 transformations...")
    transformer3 = DataTransformationThree()
    values = transformer3.transform(train_path=train_path3, test_path=test_path3)

    # Step 2: Data Ingestion for Dataset2
    print("Starting Dataset2 processing...")
    ingestion2 = DataIngestion()
    part, train_path2, test_path2 = ingestion2.init_ingestion(
        path="data/Dataset2.csv", option="csv"
    )
    # Step 4: Additional Transformations (Part 2)
    print("Running Part 2 transformations...")
    transformer2 = DataTransformationTwo()
    values = transformer2.transform(
        data_path=r"D:\2026 projects\DataMining-project\data\Dataset2.csv"
    )


if __name__ == "__main__":
    run_full_pipeline()
