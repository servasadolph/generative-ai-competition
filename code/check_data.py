import os
data_path = "/datasets/competition"
print("✅ Dataset exists:", os.path.exists(data_path))
if os.path.exists(data_path):
    print("📁 Files:", os.listdir(data_path))