import torch

import os

def get_username():

    return os.getlogin()

def check_pytorch():

    print("\nChecking PyTorch...")

    try:

        # 检查 PyTorch 版本

        print("PyTorch version:", torch.__version__)

        # 检查是否可以使用 GPU

        if torch.cuda.is_available():

            print("GPU is available for PyTorch.")

        else:

            print("GPU is not available for PyTorch.")

    except Exception as e:

        print("PyTorch test failed:", e)

 

if __name__ == "__main__":

    username = get_username()

    print(f"Running as user: {username}\n")

    check_pytorch()

